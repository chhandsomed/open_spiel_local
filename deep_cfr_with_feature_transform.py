"""
DeepCFR with Custom Feature Transformation Layer

This example demonstrates how to add a feature transformation layer
before the MLP network to convert raw information states into custom features.

Features:
- Hand strength calculation (pair, flush, straight, etc.)
- Position advantage features
- Feature normalization for training stability
- Poker domain knowledge integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel
from open_spiel.python.pytorch import deep_cfr


def card_index_to_rank_suit(card_idx, num_suits=4, num_ranks=13):
    """将牌索引转换为牌面和花色"""
    rank = card_idx // num_suits
    suit = card_idx % num_suits
    return rank, suit


# 德州扑克起手牌强度表（Preflop Hand Strength）
# 基于标准起手牌排名，归一化到[0, 1]
# 1.0 = 最强（AA），0.0 = 最弱（72o）

def get_preflop_hand_strength(rank1, rank2, is_suited, num_ranks=13):
    """计算起手牌强度（Preflop）
    
    Args:
        rank1: 第一张牌的rank (0-12, 0=2, 12=A)
        rank2: 第二张牌的rank (0-12, 0=2, 12=A)
        is_suited: 是否同花
        num_ranks: 牌面数量
    
    Returns:
        strength: 归一化的强度值 [0, 1]
    """
    # 确保 rank1 >= rank2
    if rank1 < rank2:
        rank1, rank2 = rank2, rank1
    
    # 对子（Pairs）
    if rank1 == rank2:
        # AA=1.0, KK=0.98, QQ=0.96, ..., 22=0.5
        pair_strength = 0.5 + 0.5 * (rank1 / (num_ranks - 1))
        return pair_strength
    
    # 非对子
    # 创建标准化的强度值
    # 高牌组合：AK > AQ > AJ > ... > 32
    
    # 基础强度：高牌值
    high_rank_strength = rank1 / (num_ranks - 1)  # 0-1
    
    # 低牌值影响（差距越小，强度越高）
    gap = rank1 - rank2
    gap_penalty = gap / (num_ranks - 1)  # 差距越大，惩罚越大
    
    # 同花加成
    suited_bonus = 0.15 if is_suited else 0.0
    
    # 计算最终强度
    # 对子已经处理，这里处理非对子
    base_strength = high_rank_strength * 0.7 + (1 - gap_penalty) * 0.3
    final_strength = base_strength + suited_bonus
    
    # 确保在[0, 1]范围内，且对子总是最强
    final_strength = min(final_strength, 0.95)  # 非对子最高0.95，对子可以到1.0
    
    return final_strength


# 更精确的起手牌强度表（基于标准德州扑克起手牌排名）
# 
# 参考来源：
# 1. 德州扑克起手牌强度排名: https://www.dpskill.com/jinjie/765.html
# 2. 起手牌强度排名详解: https://www.dpgod.com/teach/56.html
# 3. 起手牌图表: https://coinpoker.com/cn/help/poker-cheat-sheet/
#
# 说明：
# - 强度值基于标准排名进行归一化，范围 [0, 1]
# - 1.0 = 最强（AA），0.0 = 最弱（72o等）
# - 对子：线性映射，AA=1.0, KK=0.98, ..., 22=0.6
# - 同花组合：基于标准排名，归一化到 [0.6, 0.95]
# - 不同花组合：同花强度减去 0.18（经验值）
# - 这些数值是相对强度，用于特征工程，不是绝对胜率
#
# 详细说明请参考: PREFLOP_STRENGTH_REFERENCES.md
PREFLOP_HAND_STRENGTH_TABLE = {
    # 对子（Pairs）- 最强
    (12, 12): 1.00,  # AA - 最强
    (11, 11): 0.98,  # KK
    (10, 10): 0.96,  # QQ
    (9, 9): 0.94,    # JJ
    (8, 8): 0.92,    # TT
    (7, 7): 0.88,    # 99
    (6, 6): 0.84,    # 88
    (5, 5): 0.80,    # 77
    (4, 4): 0.76,    # 66
    (3, 3): 0.72,    # 55
    (2, 2): 0.68,    # 44
    (1, 1): 0.64,    # 33
    (0, 0): 0.60,    # 22
    
    # 同花（Suited）- 高牌组合
    (12, 11): 0.95,  # AKs
    (12, 10): 0.93,  # AQs
    (12, 9): 0.91,   # AJs
    (12, 8): 0.89,   # ATs
    (12, 7): 0.87,   # A9s
    (12, 6): 0.85,   # A8s
    (12, 5): 0.83,   # A7s
    (12, 4): 0.81,   # A6s
    (12, 3): 0.79,   # A5s
    (12, 2): 0.77,   # A4s
    (12, 1): 0.75,   # A3s
    (12, 0): 0.73,   # A2s
    
    (11, 10): 0.90,  # KQs
    (11, 9): 0.88,   # KJs
    (11, 8): 0.86,   # KTs
    (11, 7): 0.84,   # K9s
    (11, 6): 0.82,   # K8s
    (11, 5): 0.80,   # K7s
    (11, 4): 0.78,   # K6s
    (11, 3): 0.76,   # K5s
    (11, 2): 0.74,   # K4s
    (11, 1): 0.72,   # K3s
    (11, 0): 0.70,   # K2s
    
    (10, 9): 0.85,   # QJs
    (10, 8): 0.83,   # QTs
    (10, 7): 0.81,   # Q9s
    (10, 6): 0.79,   # Q8s
    (10, 5): 0.77,   # Q7s
    (10, 4): 0.75,   # Q6s
    (10, 3): 0.73,   # Q5s
    (10, 2): 0.71,   # Q4s
    (10, 1): 0.69,   # Q3s
    (10, 0): 0.67,   # Q2s
    
    (9, 8): 0.80,    # JTs
    (9, 7): 0.78,   # J9s
    (9, 6): 0.76,   # J8s
    (9, 5): 0.74,   # J7s
    (9, 4): 0.72,   # J6s
    (9, 3): 0.70,   # J5s
    (9, 2): 0.68,   # J4s
    (9, 1): 0.66,   # J3s
    (9, 0): 0.64,   # J2s
    
    (8, 7): 0.75,    # T9s
    (8, 6): 0.73,    # T8s
    (8, 5): 0.71,    # T7s
    (8, 4): 0.69,    # T6s
    (8, 3): 0.67,    # T5s
    (8, 2): 0.65,    # T4s
    (8, 1): 0.63,    # T3s
    (8, 0): 0.61,    # T2s
    
    # 注意：不同花（Offsuit）的强度通过同花强度减去0.18计算
    # 不需要在表中单独列出，避免重复定义
}


def calculate_preflop_hand_strength(hole_cards_bits, num_suits=4, num_ranks=13):
    """计算起手牌强度（Preflop Hand Strength）
    
    根据标准德州扑克起手牌强度表，计算两张手牌的强度。
    
    Args:
        hole_cards_bits: [batch_size, 52] 手牌的one-hot编码
        num_suits: 花色数量
        num_ranks: 牌面数量
    
    Returns:
        strength: [batch_size, 1] 归一化的起手牌强度 [0, 1]
    """
    batch_size = hole_cards_bits.shape[0]
    strengths = []
    
    for i in range(batch_size):
        # 找到两张手牌
        card_indices = torch.nonzero(hole_cards_bits[i] > 0, as_tuple=False).squeeze()
        
        # 处理维度问题：检查是否有足够的牌
        if card_indices.numel() == 0:
            # 没有手牌
            strengths.append(0.5)
            continue
        
        # 处理单样本情况
        if card_indices.dim() == 0:
            card_indices = card_indices.unsqueeze(0)
        
        if card_indices.numel() < 2:
            # 如果手牌不足2张，返回中等强度
            strengths.append(0.5)
            continue
        
        # 获取两张牌的rank和suit
        rank1, suit1 = card_index_to_rank_suit(card_indices[0].item(), num_suits, num_ranks)
        rank2, suit2 = card_index_to_rank_suit(card_indices[1].item(), num_suits, num_ranks)
        
        # 确保 rank1 >= rank2
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1
        
        # 是否同花
        is_suited = (suit1 == suit2)
        
        # 查找强度表
        hand_key = (rank1, rank2)
        
        # 对子：直接查表（对子最强，不区分花色）
        if rank1 == rank2:
            strength = PREFLOP_HAND_STRENGTH_TABLE.get(hand_key, 0.5)
        else:
            # 非对子：根据是否同花选择不同的强度
            if hand_key in PREFLOP_HAND_STRENGTH_TABLE:
                # 表中存储的是同花强度
                if is_suited:
                    strength = PREFLOP_HAND_STRENGTH_TABLE[hand_key]
                else:
                    # 不同花：同花强度减去0.15-0.20
                    strength = max(0.0, PREFLOP_HAND_STRENGTH_TABLE[hand_key] - 0.18)
            else:
                # 如果表中没有，使用计算函数
                strength = get_preflop_hand_strength(rank1, rank2, is_suited, num_ranks)
        
        strengths.append(strength)
    
    return torch.tensor(strengths, dtype=torch.float32, device=hole_cards_bits.device).unsqueeze(1)


def calculate_hand_strength_features(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """计算手牌强度特征（基于领域知识）
    
    注意：只保留起手牌强度特征，其他特征（对子、同花、顺子等）可以从原始信息状态中学习
    
    Args:
        hole_cards_bits: [batch_size, 52] 手牌的one-hot编码
        board_cards_bits: [batch_size, 52] 公共牌的one-hot编码（未使用，保留以兼容接口）
        num_suits: 花色数量
        num_ranks: 牌面数量
    
    Returns:
        features: [batch_size, 1] 起手牌强度特征
    """
    # 起手牌强度（Preflop Hand Strength）
    # 这是最重要的特征之一，直接反映初始两张手牌的强度
    # 其他手牌强度特征（对子、同花、顺子、公共牌信息等）可以从原始信息状态中学习
    preflop_strength = calculate_preflop_hand_strength(hole_cards_bits, num_suits, num_ranks)
    
    return preflop_strength


def calculate_position_advantage(player_idx, num_players=6):
    """计算位置优势特征
    
    位置优势（从大到小）：
    - BTN (Button): 最强位置
    - CO (Cutoff): 强位置
    - MP (Middle Position): 中等位置
    - UTG (Under The Gun): 最弱位置
    - SB (Small Blind): 盲注位置
    - BB (Big Blind): 盲注位置
    
    Args:
        player_idx: 玩家索引 [batch_size, 1] 或 [batch_size]
        num_players: 玩家总数
    
    Returns:
        position_features: [batch_size, 4] 位置特征（position_advantage, is_early, is_late, is_blind）
    """
    if isinstance(player_idx, torch.Tensor):
        if len(player_idx.shape) == 2:
            player_idx = player_idx.squeeze(1)
        batch_size = player_idx.shape[0]
    else:
        batch_size = 1
        player_idx = torch.tensor([player_idx])
    
    features = []
    
    # 位置优势值（6人场：UTG=0, MP=1, CO=2, BTN=3, SB=4, BB=5）
    # 位置优势：BTN > CO > MP > UTG，SB和BB在盲注轮有位置优势
    position_values = {
        0: 0.0,   # UTG - 最弱
        1: 0.3,   # MP
        2: 0.6,   # CO
        3: 1.0,   # BTN - 最强
        4: 0.7,   # SB
        5: 0.8,   # BB
    }
    
    # 计算位置优势值
    position_advantage = torch.zeros(batch_size, 1, device=player_idx.device)
    for i, idx in enumerate(player_idx):
        idx_int = int(idx.item())
        position_advantage[i, 0] = position_values.get(idx_int, 0.0)
    features.append(position_advantage)
    
    # 位置编码（one-hot，但只取关键位置）
    # 注意：is_early/is_late/is_blind 和 distance_to_btn 与 position_advantage 有重叠
    # 但保留 is_early/is_late/is_blind 因为它们提供离散分类信息，可能对网络学习有帮助
    # 移除 distance_to_btn，因为它与 position_advantage 高度相关，信息冗余
    player_idx_expanded = player_idx.unsqueeze(1)  # [batch_size, 1]
    is_early = ((player_idx_expanded == 0) | (player_idx_expanded == 1)).float()  # UTG, MP
    is_late = ((player_idx_expanded == 2) | (player_idx_expanded == 3)).float()    # CO, BTN
    is_blind = ((player_idx_expanded == 4) | (player_idx_expanded == 5)).float()  # SB, BB
    
    features.extend([is_early, is_late, is_blind])
    
    # 注意：distance_to_btn 已移除，因为与 position_advantage 高度相关，信息冗余
    
    return torch.cat(features, dim=1)


class FeatureTransformLayer(nn.Module):
    """特征转换层：将原始信息状态转换为自定义特征
    
    这个层可以：
    1. 降维或增维
    2. 添加领域知识
    3. 特征归一化
    4. 提取结构化特征
    """
    
    def __init__(self, raw_input_size, transformed_size, dropout=0.1):
        """
        Args:
            raw_input_size: 原始信息状态大小 (例如: 266)
            transformed_size: 转换后的特征大小 (例如: 200)
            dropout: Dropout 概率
        """
        super(FeatureTransformLayer, self).__init__()
        self.raw_input_size = raw_input_size
        self.transformed_size = transformed_size
        
        # 方案1: 简单的线性转换（端到端可学习）
        self.transform = nn.Sequential(
            nn.Linear(raw_input_size, transformed_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformed_size * 2, transformed_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(transformed_size, transformed_size),
            nn.ReLU()
        )
        
        # 方案2: 添加 Batch Normalization（可选，提高训练稳定性）
        self.bn = nn.BatchNorm1d(transformed_size)
        self.use_bn = True
    
    def forward(self, x):
        """
        Args:
            x: 原始信息状态张量 [batch_size, raw_input_size]
        
        Returns:
            transformed: 转换后的特征 [batch_size, transformed_size]
        """
        # 如果输入是 1D，添加 batch 维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # 通过转换层
        transformed = self.transform(x)
        
        # 可选：Batch Normalization
        if self.use_bn and transformed.shape[0] > 1:
            transformed = self.bn(transformed)
        
        return transformed


class MLPWithFeatureTransform(nn.Module):
    """带特征转换层的 MLP 网络
    
    架构：
    原始信息状态 -> 特征转换层 -> MLP -> 输出
    """
    
    def __init__(self,
                 raw_input_size,
                 transformed_size,
                 hidden_sizes,
                 output_size,
                 activate_final=False,
                 dropout=0.1):
        """
        Args:
            raw_input_size: 原始信息状态大小 (266)
            transformed_size: 转换后的特征大小 (200)
            hidden_sizes: MLP 隐藏层大小列表 (例如: [256, 256])
            output_size: 输出大小 (4个动作)
            activate_final: 最后一层是否使用激活函数
            dropout: Dropout 概率
        """
        super(MLPWithFeatureTransform, self).__init__()
        
        # 特征转换层
        self.feature_transform = FeatureTransformLayer(
            raw_input_size, transformed_size, dropout
        )
        
        # 原始 MLP（使用转换后的特征大小作为输入）
        self.mlp = deep_cfr.MLP(
            transformed_size,
            hidden_sizes,
            output_size,
            activate_final=activate_final
        )
    
    def forward(self, x):
        """
        Args:
            x: 原始信息状态张量 [batch_size, raw_input_size]
        
        Returns:
            output: 网络输出 [batch_size, output_size]
        """
        # 先通过特征转换层
        transformed = self.feature_transform(x)
        # 再通过 MLP
        return self.mlp(transformed)
    
    def reset(self):
        """重置网络（用于 advantage network 重新初始化）"""
        # 重置 MLP，但保留转换层（可选）
        self.mlp.reset()
        # 如果需要重置转换层，可以添加：
        # for layer in self.feature_transform.transform:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)


class HybridFeatureTransform(nn.Module):
    """混合特征转换：手动特征工程 + 可学习转换
    
    结合领域知识和端到端学习
    
    包含：
    1. 手牌强度特征（对子、同花、顺子等）
    2. 位置优势特征
    3. 下注历史统计特征
    4. 特征归一化
    """
    
    def __init__(self, raw_input_size, transformed_size, num_players=6, max_game_length=52, 
                 use_normalization=True):
        """
        Args:
            raw_input_size: 原始信息状态大小（自动检测，通常为266-281，取决于游戏配置）
            transformed_size: 转换后的特征大小（推荐150-200）
            num_players: 玩家数量
            max_game_length: 最大游戏长度
            use_normalization: 是否使用特征归一化
        
        维度流程：
        - 输入：raw_input_size维（例如281维，保留全部）
        - 手动特征：从raw_input_size中提取7维（位置、起手牌强度、下注统计，新增）
        - 可学习特征：从raw_input_size中提取64维（新增）
        - 合并：raw_input_size + 7 + 64 = 281 + 7 + 64 = 352维
        - 最终输出：352维 -> transformed_size维（例如150维）
        """
        super(HybridFeatureTransform, self).__init__()
        self.num_players = num_players
        self.max_game_length = max_game_length
        self.use_normalization = use_normalization
        
        # 手动特征维度：
        # - 位置优势特征: 4维（position_advantage, is_early, is_late, is_blind，不包含distance_to_btn）
        # - 手牌强度特征: 1维（只保留起手牌强度，其他特征可以从原始信息状态中学习）
        # - 下注统计特征: 2维（最大下注、总下注，不包含平均下注和动作数量）
        # 总计: 4 + 1 + 2 = 7维
        self.manual_feature_size = 7  # 手动特征维度
        
        # 可学习的特征提取器（处理原始信息状态）
        layers = []
        layers.append(nn.Linear(raw_input_size, 128))
        if use_normalization:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(128, 64))
        if use_normalization:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        self.learned_transform = nn.Sequential(*layers)
        self.learned_transform_layers = layers  # 保存层引用以便单样本处理
        
        # 合并原始信息状态、手动特征和可学习特征
        # 保留原始信息状态的全部维度，然后添加提取的特征
        combined_size = raw_input_size + self.manual_feature_size + 64
        final_layers = []
        final_layers.append(nn.Linear(combined_size, transformed_size))
        if use_normalization:
            final_layers.append(nn.BatchNorm1d(transformed_size))
        final_layers.append(nn.ReLU())
        final_layers.append(nn.Dropout(0.1))
        self.final_transform = nn.Sequential(*final_layers)
        self.final_transform_layers = final_layers  # 保存层引用以便单样本处理
    
    def extract_manual_features(self, x):
        """手动提取特征（领域知识）
        
        提取的特征包括：
        1. 手牌强度特征（对子、同花、顺子等）
        2. 位置优势特征
        3. 下注历史统计特征
        4. 其他统计特征
        """
        batch_size = x.shape[0]
        features = []
        
        # ========== 1. 玩家位置特征 ==========
        player_pos = x[:, 0:self.num_players]
        player_idx = torch.argmax(player_pos, dim=1, keepdim=True).float()
        
        # 位置优势特征（包含领域知识）
        position_features = calculate_position_advantage(player_idx, self.num_players)
        features.append(position_features)  # 4维（不包含distance_to_btn）
        
        # ========== 2. 手牌强度特征 ==========
        hole_cards = x[:, self.num_players:self.num_players+52]
        board_cards = x[:, self.num_players+52:self.num_players+104]
        
        # 计算手牌强度特征（只保留起手牌强度）
        # 注意：其他手牌强度特征（对子、同花、顺子、公共牌信息等）可以从原始信息状态中学习
        # 起手牌强度是最重要的先验知识，直接反映初始两张手牌的强度
        hand_strength_features = calculate_hand_strength_features(
            hole_cards, board_cards
        )
        features.append(hand_strength_features)  # 1维（只包含起手牌强度）
        
        # ========== 3. 下注历史统计特征 ==========
        action_seq_start = self.num_players + 104
        action_seq_end = action_seq_start + self.max_game_length * 2
        
        # 提取动作序列（每2个bit一个动作）
        action_seq = x[:, action_seq_start:action_seq_end]
        # 计算动作数量（非零动作）
        num_actions = torch.sum(
            (action_seq[:, 0::2] != 0) | (action_seq[:, 1::2] != 0),
            dim=1, keepdim=True
        ).float()
        
        # 动作大小统计
        action_sizings_start = action_seq_end
        action_sizings = x[:, action_sizings_start:action_sizings_start+self.max_game_length]
        
        # 归一化下注金额（避免数值过大）
        total_bet = torch.sum(action_sizings, dim=1, keepdim=True)
        max_bet = torch.max(action_sizings, dim=1, keepdim=True)[0]
        avg_bet = total_bet / (num_actions + 1e-8)  # 避免除零
        
        # 归一化到合理范围（假设最大下注为20000）
        max_bet_norm = max_bet / 20000.0
        total_bet_norm = total_bet / 20000.0
        avg_bet_norm = avg_bet / 20000.0
        
        # 动作类型统计（简化：fold, call, raise, all-in）
        # 注意：
        # 1. 归一化动作数量已移除（与game_stage和下注金额特征重叠）
        # 2. avg_bet_norm 已移除，因为 avg_bet = total_bet / num_actions，可以从total_bet推导
        #    但 max_bet 和 total_bet 提供不同信息，都保留
        features.extend([
            max_bet_norm,  # 归一化最大下注（单次最大下注，反映激进程度）
            total_bet_norm,  # 归一化总下注（累计下注，反映投入程度）
        ])  # 2维
        
        # ========== 4. 其他统计特征 ==========
        # 注意：游戏阶段特征已移除，因为：
        # 1. 游戏阶段信息已经隐含在其他特征中：
        #    - max_board_rank: Preflop=0, Flop/Turn/River>0
        #    - pair_count: Preflop=0, 后续阶段可能>0
        #    - flush_potential: Preflop=0, 后续阶段可能>0
        # 2. 这些特征本身就能反映游戏阶段，不需要额外的game_stage特征
        # 3. 归一化（board_count / 5.0）也不必要，因为board_count本身就是离散值（0, 3, 4, 5）
        
        # ========== 组合所有特征 ==========
        manual_features = torch.cat(features, dim=1)
        
        # 确保特征维度正确
        if manual_features.shape[1] < self.manual_feature_size:
            padding = torch.zeros(
                batch_size,
                self.manual_feature_size - manual_features.shape[1],
                device=x.device
            )
            manual_features = torch.cat([manual_features, padding], dim=1)
        elif manual_features.shape[1] > self.manual_feature_size:
            manual_features = manual_features[:, :self.manual_feature_size]
        
        # 注意：不对手动特征进行全局归一化，原因：
        # 1. 布尔特征（is_pair, is_flush_draw等）应该保持0/1，归一化会破坏语义
        # 2. 已经归一化的特征（如[0,1]范围的特征）已经具有正确的语义和范围
        # 3. 各个特征已经单独归一化到合理范围（如除以最大值、归一化到[0,1]等）
        # 4. 网络中的BatchNorm已经提供了必要的归一化，不需要对手动特征再次归一化
        
        return manual_features
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 原始信息状态 [batch_size, raw_input_size]
        
        Returns:
            transformed: 转换后的特征 [batch_size, transformed_size]
        """
        # 如果输入是1D，添加batch维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # 保留原始信息状态
        original_features = x
        
        # 手动特征（领域知识）
        manual_features = self.extract_manual_features(x)
        
        # 可学习特征（端到端学习）
        # BatchNorm在单样本时会出错，使用LayerNorm替代或跳过
        if x.shape[0] == 1 and self.use_normalization:
            # 单样本时手动执行（跳过BatchNorm）
            learned_features = x
            for layer in self.learned_transform_layers:
                if isinstance(layer, nn.BatchNorm1d):
                    # 跳过BatchNorm，使用LayerNorm替代
                    learned_features = F.layer_norm(learned_features.unsqueeze(0), 
                                                    (learned_features.shape[-1],)).squeeze(0)
                else:
                    learned_features = layer(learned_features)
        else:
            learned_features = self.learned_transform(x)
        
        # 合并：原始信息状态 + 手动特征 + 可学习特征
        combined = torch.cat([original_features, manual_features, learned_features], dim=1)
        
        # 最终转换（带归一化）
        if x.shape[0] == 1 and self.use_normalization:
            # 单样本时手动执行（跳过BatchNorm）
            transformed = combined
            for layer in self.final_transform_layers:
                if isinstance(layer, nn.BatchNorm1d):
                    # 跳过BatchNorm，使用LayerNorm替代
                    transformed = F.layer_norm(transformed.unsqueeze(0), 
                                               (transformed.shape[-1],)).squeeze(0)
                else:
                    transformed = layer(transformed)
        else:
            transformed = self.final_transform(combined)
        
        return transformed


class DeepCFRWithFeatureTransform(deep_cfr.DeepCFRSolver):
    """带特征转换层的 DeepCFR Solver
    
    这个类继承 DeepCFRSolver，但使用带特征转换层的网络。
    只需要重写网络创建部分，其他方法自动继承。
    """
    
    def __init__(self,
                 game,
                 policy_network_layers=(256, 256),
                 advantage_network_layers=(128, 128),
                 transformed_size=200,  # 新增：转换后的特征大小
                 use_hybrid_transform=False,  # 是否使用混合转换
                 num_iterations: int = 100,
                 num_traversals: int = 20,
                 learning_rate: float = 1e-4,
                 batch_size_advantage=None,
                 batch_size_strategy=None,
                 memory_capacity: int = int(1e6),
                 policy_network_train_steps: int = 1,
                 advantage_network_train_steps: int = 1,
                 reinitialize_advantage_networks: bool = True,
                 device=None):
        """
        Args:
            transformed_size: 转换后的特征大小（默认200，小于原始信息状态大小）
            use_hybrid_transform: 是否使用混合特征转换（手动+可学习）
            其他参数同 DeepCFRSolver
        
        注意：
        - 原始信息状态大小会自动检测（通常为266-281，取决于游戏配置）
        - 特征转换流程：保留原始信息状态 + 添加手动特征（7维）+ 添加可学习特征（64维）-> transformed_size（推荐150-200）
        - 合并后维度：raw_input_size + 7 + 64（例如：281 + 7 + 64 = 352维）
        - 最终输出：352维 -> transformed_size维
        """
        # 先初始化基础属性（不创建网络）
        import pyspiel
        from open_spiel.python import policy
        all_players = list(range(game.num_players()))
        policy.Policy.__init__(self, game, all_players)
        self._game = game
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError("Simultaneous games are not supported.")
        
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._transformed_size = transformed_size
        self._use_hybrid_transform = use_hybrid_transform
        
        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        
        # 创建带转换层的策略网络
        self._strategy_memories = deep_cfr.ReservoirBuffer(memory_capacity)
        
        if use_hybrid_transform:
            # 使用混合特征转换
            transform_layer = HybridFeatureTransform(
                self._embedding_size,
                transformed_size,
                num_players=self._num_players,
                max_game_length=game.max_game_length()
            )
            # 创建 MLP（使用转换后的尺寸）
            mlp = deep_cfr.MLP(transformed_size, list(policy_network_layers), self._num_actions)
            # 组合成完整网络
            self._policy_network = nn.Sequential(transform_layer, mlp)
        else:
            # 使用简单的特征转换层
            self._policy_network = MLPWithFeatureTransform(
                self._embedding_size,
                transformed_size,
                list(policy_network_layers),
                self._num_actions
            )
        
        self._policy_network = self._policy_network.to(self._device)
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)
        
        # 创建带转换层的优势网络（每个玩家一个）
        self._advantage_memories = [
            deep_cfr.ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        
        if use_hybrid_transform:
            self._advantage_networks = []
            for _ in range(self._num_players):
                transform_layer = HybridFeatureTransform(
                    self._embedding_size,
                    transformed_size,
                    num_players=self._num_players,
                    max_game_length=game.max_game_length()
                )
                mlp = deep_cfr.MLP(transformed_size, list(advantage_network_layers), self._num_actions)
                self._advantage_networks.append(nn.Sequential(transform_layer, mlp))
        else:
            self._advantage_networks = [
                MLPWithFeatureTransform(
                    self._embedding_size,
                    transformed_size,
                    list(advantage_network_layers),
                    self._num_actions
                ) for _ in range(self._num_players)
            ]
        
        # Move advantage networks to device
        for i in range(self._num_players):
            self._advantage_networks[i] = self._advantage_networks[i].to(self._device)
        
        self._loss_advantages = nn.MSELoss(reduction="mean")
        self._optimizer_advantages = []
        for p in range(self._num_players):
            self._optimizer_advantages.append(
                torch.optim.Adam(
                    self._advantage_networks[p].parameters(), lr=learning_rate))
        self._learning_rate = learning_rate
    
    def reinitialize_advantage_network(self, player):
        """重新初始化优势网络（保留转换层，重置 MLP）"""
        # 如果是 Sequential，需要访问内部的 MLP
        if isinstance(self._advantage_networks[player], nn.Sequential):
            # 找到 MLP 层并重置
            for module in self._advantage_networks[player]:
                if isinstance(module, deep_cfr.MLP):
                    module.reset()
        elif hasattr(self._advantage_networks[player], 'mlp'):
            # MLPWithFeatureTransform
            self._advantage_networks[player].mlp.reset()
        else:
            # 普通 MLP
            self._advantage_networks[player].reset()
        
        # 确保网络在正确的设备上
        self._advantage_networks[player] = self._advantage_networks[player].to(self._device)
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self._learning_rate)
    
    # 继承其他方法（_traverse_game_tree, _sample_action_from_advantage 等）
    # 这些方法不需要修改，因为它们只是调用网络，不关心网络内部结构


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("DeepCFR with Feature Transformation - 使用示例")
    print("=" * 60)
    
    # 创建游戏
    game_config = {
        "numPlayers": 6,
        "numBoardCards": "0 3 1 1",  # Preflop, Flop, Turn, River
        "numRanks": 13,
        "numSuits": 4,
        "firstPlayer": "2",  # UTG
        "stack": "20000 20000 20000 20000 20000 20000",
        "blind": "50 100 0 0 0 0",  # SB, BB, UTG, MP, CO, BTN
        "numHoleCards": 2,
        "numRounds": 4,
        "betting": "nolimit",
        "maxRaises": "3",
    }
    game = pyspiel.load_game("universal_poker", game_config)
    
    print(f"\n游戏: {game.get_type().short_name}")
    print(f"玩家数量: {game.num_players()}")
    
    # 获取原始信息状态大小
    state = game.new_initial_state()
    raw_info_state = state.information_state_tensor(0)
    raw_size = len(raw_info_state)
    print(f"原始信息状态大小: {raw_size}")
    
    # 创建带特征转换层的 DeepCFR Solver
    print("\n" + "-" * 60)
    print("创建 DeepCFR with Feature Transform")
    print("-" * 60)
    
    solver = DeepCFRWithFeatureTransform(
        game,
        policy_network_layers=(256, 256),
        advantage_network_layers=(128, 128),
        transformed_size=200,  # 从 266 降到 200
        use_hybrid_transform=False,  # 使用简单转换层
        num_iterations=10,  # 示例：少量迭代
        num_traversals=5,
        learning_rate=1e-4,
        device=torch.device("cpu")
    )
    
    print(f"转换后的特征大小: {solver._transformed_size}")
    print(f"策略网络参数数量: {sum(p.numel() for p in solver._policy_network.parameters())}")
    print(f"优势网络参数数量: {sum(p.numel() for p in solver._advantage_networks[0].parameters())}")
    
    # 测试前向传播
    print("\n" + "-" * 60)
    print("测试前向传播")
    print("-" * 60)
    
    # 获取一个信息状态
    while not state.is_terminal() and not state.is_chance_node():
        legal_actions = state.legal_actions()
        if legal_actions:
            state.apply_action(legal_actions[0])
        else:
            break
    
    if not state.is_terminal():
        info_state = state.information_state_tensor(0)
        info_state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0))
        
        # 测试策略网络
        with torch.no_grad():
            output = solver._policy_network(info_state_tensor)
            probs = solver._policy_sm(output)
        
        print(f"输入形状: {info_state_tensor.shape}")
        print(f"输出形状: {output.shape}")
        print(f"动作概率: {probs[0].numpy()}")
        
        # 测试优势网络
        with torch.no_grad():
            advantage_output = solver._advantage_networks[0](info_state_tensor)
        print(f"优势网络输出形状: {advantage_output.shape}")
        print(f"优势值: {advantage_output[0].numpy()}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("\n使用说明：")
    print("1. 在 train_deep_cfr_texas.py 中导入 DeepCFRWithFeatureTransform")
    print("2. 替换原来的 DeepCFRSolver")
    print("3. 可以调整 transformed_size 来改变特征维度")
    print("4. 设置 use_hybrid_transform=True 使用混合特征转换")


if __name__ == "__main__":
    example_usage()

 