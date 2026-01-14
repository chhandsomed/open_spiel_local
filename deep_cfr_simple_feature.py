"""
简化版本：直接拼接手动特征到原始信息状态

流程：
信息状态(raw_input_size维) + 手动特征(25维) = (raw_input_size+25)维 -> MLP

手动特征组成（25维）：
1. 起手牌强度 (1维，加权2倍以提高重要性，范围0-2)
2. 当前手牌强度（考虑公共牌）(1维，范围0-1)
3. 是否中牌 (1维，0/1布尔值)
4. 成牌特征：是否中了对子、三条、两对 (3维，0/1布尔值)
5. 听牌特征：同花听牌、outs数量、顺子听牌、outs数量、成牌概率 (5维，范围0-1)
6. 公共牌特征：强度、是否同花面、是否顺子面 (3维，范围0-1)
7. 游戏轮次：Preflop/Flop/Turn/River (4维 one-hot，0/1布尔值)
8. 手牌强度变化 (1维，使用tanh归一化到[0,1])
9. 下注统计：最大下注、总下注 (2维，归一化到0-1)
10. 动作特征：是否有人加注、是否有人全押 (2维，0/1布尔值)

注意：OpenSpiel 1.6.9中，信息状态维度包含sizings部分，Bet动作后sizings包含实际下注金额

对于6人fchpa: 281 + 25 = 306维 (OpenSpiel 1.6.9)
对于6人fcpa: 266 + 25 = 291维
对于2人fchpa: 175 + 25 = 200维

支持多 GPU 并行训练（DataParallel）
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from open_spiel.python.pytorch import deep_cfr
from deep_cfr_with_feature_transform import (
    calculate_hand_strength_features,
    calculate_preflop_hand_strength,
    card_index_to_rank_suit
)


# ========== 向量化特征计算函数（高性能版本） ==========
# 优化：使用 PyTorch 批量操作代替 Python 循环，提速 100x+

def vectorized_preflop_strength(hole_cards_bits, num_suits=4, num_ranks=13):
    """向量化计算起手牌强度（完全向量化，无Python循环）
    
    Args:
        hole_cards_bits: [batch_size, 52] 手牌的one-hot编码
        
    Returns:
        strength: [batch_size, 1] 归一化的起手牌强度 [0, 1]
    """
    batch_size = hole_cards_bits.shape[0]
    device = hole_cards_bits.device
    
    # 将 52 张牌的 one-hot 转换为 rank 和 suit 表示
    # 牌索引布局: rank * num_suits + suit，即 0-3 对应 rank=0(2)，4-7 对应 rank=1(3)，...
    card_indices = hole_cards_bits.view(batch_size, num_ranks, num_suits)  # [batch, 13, 4]
    
    # 统计每个 rank 的牌数
    rank_counts = card_indices.sum(dim=2)  # [batch, 13]
    has_card = (rank_counts > 0).float()
    # 修复：统计牌的总数，而不是有牌的 rank 数量
    card_count = hole_cards_bits.sum(dim=1, keepdim=True)  # [batch, 1]
    
    # 1. 对子检测
    is_pair = (rank_counts == 2).any(dim=1, keepdim=True).float()  # [batch, 1]
    
    # 2. 对子 rank（向量化）：使用加权索引找最大对子 rank
    pair_mask = (rank_counts == 2).float()  # [batch, 13]
    rank_indices = torch.arange(num_ranks, device=device).float().unsqueeze(0)  # [1, 13]
    # 对子 rank = 有对子的最大 rank 索引
    pair_rank = (pair_mask * rank_indices).max(dim=1, keepdim=True)[0]  # [batch, 1]
    pair_strength = pair_rank / 12.0
    
    # 3. 高牌 rank（向量化）
    weighted_ranks = has_card * rank_indices
    max_rank = weighted_ranks.max(dim=1, keepdim=True)[0]  # [batch, 1]
    
    # 第二高牌（向量化）：将最高牌位置清零后再取最大
    max_rank_idx = max_rank.long()  # [batch, 1]
    # 创建 one-hot 掩码移除最高牌
    one_hot_max = torch.zeros(batch_size, num_ranks, device=device)
    one_hot_max.scatter_(1, max_rank_idx, 1.0)
    second_rank_mask = has_card * (1 - one_hot_max)
    second_rank = (second_rank_mask * rank_indices).max(dim=1, keepdim=True)[0]  # [batch, 1]
    
    # 4. 是否同花
    suit_counts = card_indices.sum(dim=1)  # [batch, 4]
    is_suited = (suit_counts == 2).any(dim=1, keepdim=True).float()
    
    # 5. 连张检测
    rank_diff = torch.abs(max_rank - second_rank)
    is_connected = (rank_diff <= 1).float()
    is_gapper = ((rank_diff == 2) | (rank_diff == 3)).float()
    
    # 综合计算强度
    pair_base = 0.5 + pair_strength * 0.5  # 对子：0.5-1.0
    high_card_value = (max_rank / 12.0) * 0.4 + (second_rank / 12.0) * 0.2
    suited_bonus = is_suited * 0.1
    connected_bonus = is_connected * 0.05 + is_gapper * 0.02
    non_pair_strength = high_card_value + suited_bonus + connected_bonus
    
    # 最终强度
    strength = is_pair * pair_base + (1 - is_pair) * non_pair_strength
    
    # 处理无效输入
    invalid_mask = (card_count < 2)
    strength = torch.where(invalid_mask, torch.tensor(0.5, device=device), strength)
    
    return strength


def vectorized_hand_strength(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """向量化计算手牌强度（完全向量化，无Python循环）
    
    Returns:
        strength: [batch_size, 1] 手牌强度值 (0-1)
    """
    batch_size = hole_cards_bits.shape[0]
    device = hole_cards_bits.device
    
    # 合并手牌和公共牌
    all_cards = hole_cards_bits + board_cards_bits
    all_cards = (all_cards > 0.5).float()
    total_cards = all_cards.sum(dim=1, keepdim=True)
    
    # 转换为 rank/suit 表示
    cards_by_rank = all_cards.view(batch_size, num_ranks, num_suits)  # [batch, 13, 4]
    rank_counts = cards_by_rank.sum(dim=2)  # [batch, 13]
    suit_counts = cards_by_rank.sum(dim=1)  # [batch, 4]
    
    # 检测牌型
    has_four = (rank_counts == 4).any(dim=1, keepdim=True).float()
    has_three = (rank_counts >= 3).any(dim=1, keepdim=True).float()
    # 修复：区分纯对子和三条以上
    trips_count = (rank_counts >= 3).sum(dim=1, keepdim=True).float()  # 三条数量
    pair_only_count = (rank_counts == 2).sum(dim=1, keepdim=True).float()  # 纯对子数量（不含三条）
    total_pair_count = (rank_counts >= 2).sum(dim=1, keepdim=True).float()  # 所有>=2的rank数量
    has_pair = (total_pair_count >= 1).float()
    has_two_pair = (total_pair_count >= 2).float()
    has_flush = (suit_counts >= 5).any(dim=1, keepdim=True).float()
    
    # 顺子检测（向量化：使用卷积核）
    rank_present = (rank_counts > 0).float()  # [batch, 13]
    # A 可以做低牌 A-2-3-4-5
    rank_with_low_ace = torch.cat([rank_present[:, 12:13], rank_present], dim=1)  # [batch, 14]
    # 使用滑动求和检测顺子（5个连续rank）
    rank_expanded = rank_with_low_ace.unsqueeze(1)  # [batch, 1, 14]
    windows = rank_expanded.unfold(2, 5, 1)  # [batch, 1, 10, 5]
    window_sums = windows.sum(dim=3).squeeze(1)  # [batch, 10]
    has_straight = (window_sums >= 5).any(dim=1, keepdim=True).float()
    
    # 修复葫芦检测：三条 + 另一个对子（或另一个三条）
    # 葫芦 = (有三条) & (有纯对子 或 有两个三条)
    has_full_house = (has_three * ((pair_only_count >= 1) | (trips_count >= 2)).float()).float()
    has_straight_flush = (has_flush * has_straight).float()
    
    # 计算强度（使用条件累加避免多次 where）
    strength = torch.zeros(batch_size, 1, device=device)
    # 从低到高设置，后面的会覆盖前面的
    strength = torch.where(has_pair > 0, torch.tensor(0.3, device=device), strength)
    strength = torch.where(has_two_pair > 0, torch.tensor(0.4, device=device), strength)
    strength = torch.where(has_three > 0, torch.tensor(0.5, device=device), strength)
    strength = torch.where(has_straight > 0, torch.tensor(0.6, device=device), strength)
    strength = torch.where(has_flush > 0, torch.tensor(0.7, device=device), strength)
    strength = torch.where(has_full_house > 0, torch.tensor(0.8, device=device), strength)
    strength = torch.where(has_four > 0, torch.tensor(0.9, device=device), strength)
    strength = torch.where(has_straight_flush > 0, torch.tensor(1.0, device=device), strength)
    # 高牌
    strength = torch.where(strength == 0, torch.tensor(0.1, device=device), strength)
    
    # 对于总牌数不足5张的情况，使用起手牌强度
    use_preflop = (total_cards < 5).float()
    preflop_strength = vectorized_preflop_strength(hole_cards_bits, num_suits, num_ranks)
    strength = use_preflop * preflop_strength + (1 - use_preflop) * strength
    
    return strength


def vectorized_made_hands(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """向量化检查成牌类型
    
    Returns:
        hit_pair: [batch_size, 1] 是否有对子
        hit_trips: [batch_size, 1] 是否有三条
        hit_two_pair: [batch_size, 1] 是否有两对
    """
    batch_size = hole_cards_bits.shape[0]
    device = hole_cards_bits.device
    
    # 合并所有牌
    all_cards = hole_cards_bits + board_cards_bits
    all_cards = (all_cards > 0.5).float()
    
    # 检查是否有公共牌
    board_count = board_cards_bits.sum(dim=1, keepdim=True)
    has_board = (board_count > 0).float()
    
    # 统计 rank 计数
    cards_by_rank = all_cards.view(batch_size, num_ranks, num_suits)
    rank_counts = cards_by_rank.sum(dim=2)  # [batch, 13]
    
    # 检查成牌
    hit_pair = ((rank_counts >= 2).any(dim=1, keepdim=True).float() * has_board)
    hit_trips = ((rank_counts >= 3).any(dim=1, keepdim=True).float() * has_board)
    pair_count = (rank_counts >= 2).sum(dim=1, keepdim=True).float()
    hit_two_pair = ((pair_count >= 2).float() * has_board)
    
    return hit_pair, hit_trips, hit_two_pair


def vectorized_draws(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """向量化检查听牌（完全向量化，无Python循环）
    
    Returns:
        flush_draw: [batch_size, 1] 是否同花听牌 (4张同花)
        flush_outs: [batch_size, 1] 同花 outs 归一化
        straight_draw: [batch_size, 1] 是否顺子听牌
        straight_outs: [batch_size, 1] 顺子 outs 归一化
        draw_equity: [batch_size, 1] 听牌成牌概率
    """
    batch_size = hole_cards_bits.shape[0]
    device = hole_cards_bits.device
    
    # 合并所有牌
    all_cards = hole_cards_bits + board_cards_bits
    all_cards = (all_cards > 0.5).float()
    total_cards = all_cards.sum(dim=1, keepdim=True)
    
    # 检查是否有公共牌
    board_count = board_cards_bits.sum(dim=1, keepdim=True)
    has_board = (board_count > 0).float()
    
    # 转换为 rank/suit 表示
    cards_by_rank = all_cards.view(batch_size, num_ranks, num_suits)
    suit_counts = cards_by_rank.sum(dim=1)  # [batch, 4]
    max_suit_count = suit_counts.max(dim=1, keepdim=True)[0]
    
    # 同花听牌
    flush_draw = ((max_suit_count == 4).float() * has_board)
    flush_outs = torch.clamp((13 - max_suit_count) / 13.0, 0, 1) * has_board
    
    # 顺子听牌检测（向量化）
    rank_counts = cards_by_rank.sum(dim=2)  # [batch, 13]
    rank_present = (rank_counts > 0).float()
    rank_with_low_ace = torch.cat([rank_present[:, 12:13], rank_present], dim=1)  # [batch, 14]
    
    # 使用 unfold 创建滑动窗口
    rank_expanded = rank_with_low_ace.unsqueeze(1)  # [batch, 1, 14]
    windows = rank_expanded.unfold(2, 5, 1)  # [batch, 1, 10, 5]
    window_sums = windows.sum(dim=3).squeeze(1)  # [batch, 10]
    
    # 顺子听牌：恰好4张在5张窗口内
    is_straight_draw = (window_sums == 4).any(dim=1, keepdim=True).float()
    straight_draw = is_straight_draw * has_board
    
    # outs：找最大的窗口和（即最接近顺子的）
    max_window_count = window_sums.max(dim=1, keepdim=True)[0]
    # 如果有4张在窗口内，outs = 1（缺1张）
    straight_outs = ((max_window_count == 4).float() * 0.2) * has_board  # 1/5 = 0.2
    
    # 估算成牌概率
    remaining_cards = torch.clamp(52 - total_cards, min=1)
    flush_equity = flush_draw * 9 / remaining_cards
    straight_equity = straight_draw * 8 / remaining_cards
    draw_equity = torch.clamp(flush_equity + straight_equity, 0, 1)
    
    return flush_draw, flush_outs, straight_draw, straight_outs, draw_equity


def vectorized_board_features(board_cards_bits, num_suits=4, num_ranks=13):
    """向量化计算公共牌特征（完全向量化，无Python循环）
    
    Returns:
        board_strength: [batch_size, 1] 公共牌强度
        is_flush_board: [batch_size, 1] 是否同花面 (3+张同花)
        is_straight_board: [batch_size, 1] 是否顺子面
        game_round: [batch_size, 4] 游戏轮次 one-hot
    """
    batch_size = board_cards_bits.shape[0]
    device = board_cards_bits.device
    
    # 公共牌数量
    board_count = board_cards_bits.sum(dim=1, keepdim=True)  # [batch, 1]
    
    # 游戏轮次 one-hot
    game_round = torch.zeros(batch_size, 4, device=device)
    game_round[:, 0] = (board_count.squeeze() == 0).float()
    game_round[:, 1] = (board_count.squeeze() == 3).float()
    game_round[:, 2] = (board_count.squeeze() == 4).float()
    game_round[:, 3] = (board_count.squeeze() == 5).float()
    
    # 转换为 rank/suit 表示
    cards_by_rank = board_cards_bits.view(batch_size, num_ranks, num_suits)
    rank_counts = cards_by_rank.sum(dim=2)  # [batch, 13]
    suit_counts = cards_by_rank.sum(dim=1)  # [batch, 4]
    
    # 公共牌强度
    has_trips = (rank_counts >= 3).any(dim=1, keepdim=True).float()
    pair_count = (rank_counts >= 2).sum(dim=1, keepdim=True).float()
    has_two_pair = (pair_count >= 2).float()
    has_pair = (pair_count >= 1).float()
    
    board_strength = torch.zeros(batch_size, 1, device=device)
    board_strength = torch.where(has_pair > 0, torch.tensor(0.4, device=device), board_strength)
    board_strength = torch.where(has_two_pair > 0, torch.tensor(0.6, device=device), board_strength)
    board_strength = torch.where(has_trips > 0, torch.tensor(0.8, device=device), board_strength)
    board_strength = torch.where((board_strength == 0) & (board_count > 0), torch.tensor(0.2, device=device), board_strength)
    
    # 同花面
    max_suit_count = suit_counts.max(dim=1, keepdim=True)[0]
    is_flush_board = (max_suit_count >= 3).float()
    
    # 顺子面（向量化）
    rank_present = (rank_counts > 0).float()
    rank_with_low_ace = torch.cat([rank_present[:, 12:13], rank_present], dim=1)  # [batch, 14]
    
    # 使用 unfold 检测顺子面
    rank_expanded = rank_with_low_ace.unsqueeze(1)  # [batch, 1, 14]
    windows = rank_expanded.unfold(2, 5, 1)  # [batch, 1, 10, 5]
    window_sums = windows.sum(dim=3).squeeze(1)  # [batch, 10]
    
    # 3+张在5张窗口内算顺子面
    max_window = window_sums.max(dim=1, keepdim=True)[0]
    is_straight_board = ((max_window >= 3) & (board_count >= 3)).float()
    
    return board_strength, is_flush_board, is_straight_board, game_round


# ========== 旧版本函数（保留兼容性，但不再使用） ==========

def bits_to_card_indices(bits_tensor):
    """将bits tensor转换为牌索引列表（旧版本，已弃用）"""
    if bits_tensor.dim() == 1:
        bits_tensor = bits_tensor.unsqueeze(0)
    batch_size = bits_tensor.shape[0]
    card_indices_list = []
    for i in range(batch_size):
        indices = torch.nonzero(bits_tensor[i] > 0.5, as_tuple=False).squeeze()
        if indices.numel() == 0:
            card_indices_list.append([])
        elif indices.dim() == 0:
            card_indices_list.append([indices.item()])
        else:
            card_indices_list.append(indices.tolist())
    return card_indices_list


def evaluate_hand_strength_from_bits(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """评估手牌强度（使用向量化版本）"""
    return vectorized_hand_strength(hole_cards_bits, board_cards_bits, num_suits, num_ranks)


def check_made_hands(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """检查成牌（使用向量化版本）"""
    return vectorized_made_hands(hole_cards_bits, board_cards_bits, num_suits, num_ranks)


def check_draws(hole_cards_bits, board_cards_bits, num_suits=4, num_ranks=13):
    """检查听牌（使用向量化版本）"""
    return vectorized_draws(hole_cards_bits, board_cards_bits, num_suits, num_ranks)


def calculate_board_features(board_cards_bits, num_suits=4, num_ranks=13):
    """计算公共牌特征（使用向量化版本）"""
    return vectorized_board_features(board_cards_bits, num_suits, num_ranks)


def wrap_with_data_parallel(model, device, gpu_ids=None):
    """将模型包装为 DataParallel（如果有多个 GPU）
    
    Args:
        model: PyTorch 模型
        device: 主设备
        gpu_ids: GPU ID 列表
    
    Returns:
        包装后的模型（DataParallel 或原模型）
    """
    if gpu_ids is not None and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        model = model.to(device)
        return model
    else:
        return model.to(device)


def unwrap_data_parallel(model):
    """获取 DataParallel 包装的原始模型
    
    Args:
        model: 可能被 DataParallel 包装的模型
    
    Returns:
        原始模型
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def detect_manual_feature_size_from_state_dict(state_dict, raw_input_size):
    """从模型权重自动检测手动特征维度
    
    Args:
        state_dict: 模型权重字典
        raw_input_size: 原始信息状态大小
    
    Returns:
        manual_feature_size: 手动特征维度（1或7），如果无法检测则返回None
    """
    # 查找 MLP 第一层的权重
    # MLP 的第一层权重形状是 [hidden_size, input_size]
    # input_size = raw_input_size + manual_feature_size
    
    for key, value in state_dict.items():
        # 处理 DataParallel 的 key（可能包含 'module.' 前缀）
        clean_key = key.replace('module.', '')
        
        # 查找 MLP 的第一层（通常是 'mlp.model.0._linear._weight' 或类似）
        if 'mlp' in clean_key.lower() and ('weight' in clean_key.lower() or 'linear' in clean_key.lower()):
            if len(value.shape) == 2:
                # 第一层的输入维度
                mlp_input_size = value.shape[1]
                
                # 计算特征维度
                feature_size = mlp_input_size - raw_input_size
                
                if feature_size == 1:
                    return 1  # 新版本：1维特征
                elif feature_size == 7:
                    return 7  # 老版本：7维特征
                else:
                    # 如果检测到其他维度，返回None让用户手动指定
                    print(f"⚠️  检测到未知的特征维度: {feature_size} (MLP输入: {mlp_input_size}, 原始输入: {raw_input_size})")
                    # return None
                    return feature_size
    
    return None


class SimpleFeatureMLP(nn.Module):
    """简单的MLP，在输入前拼接手动特征
    
    兼容性说明：
    - 新版本：1维特征（手牌强度）
    - 老版本：7维特征（位置4维 + 手牌强度1维 + 下注统计2维）
    
    自动检测：根据 MLP 第一层的输入维度推断特征维度
    """
    
    def __init__(self, raw_input_size, hidden_sizes, output_size, 
                 num_players=6, max_game_length=52, activate_final=False, max_stack=2000,
                 manual_feature_size=None):
        """
        Args:
            raw_input_size: 原始信息状态大小（例如224，取决于游戏配置）
            hidden_sizes: MLP隐藏层大小列表（例如[256, 256]）
            output_size: 输出大小（动作数量）
            num_players: 玩家数量
            max_game_length: 最大游戏长度
            activate_final: 最后一层是否使用激活函数
            max_stack: 单个玩家的最大筹码量（用于归一化下注统计特征，默认2000）
            manual_feature_size: 手动特征维度（None时自动推断：1维或7维）
        """
        super(SimpleFeatureMLP, self).__init__()
        self.num_players = num_players
        self.max_game_length = max_game_length
        self.max_stack = float(max_stack)  # 保留用于兼容性
        # 预计算log归一化的分母，避免重复计算
        import numpy as np
        self.log_max_stack = np.log1p(self.max_stack)
        
        # 如果未指定，默认使用增强版本（23维）
        if manual_feature_size is None:
            self.manual_feature_size = 23  # 增强版本：23维特征
        else:
            self.manual_feature_size = manual_feature_size
        
        # 输入维度 = 原始信息状态 + 手动特征
        input_size = raw_input_size + self.manual_feature_size
        
        # 使用原始的MLP结构
        self.mlp = deep_cfr.MLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final=activate_final
        )
        
        # 保存原始输入大小用于调试
        self.raw_input_size = raw_input_size
    
    def extract_manual_features(self, x):
        """提取手动特征
        
        兼容性支持：
        - 增强版本（23维）：手牌强度、成牌、听牌、公共牌、游戏轮次等特征
        - 老版本（7维）：位置特征(4维) + 手牌强度(1维) + 下注统计(2维)
        - 旧版本（1维）：只提取手牌强度
        """
        batch_size = x.shape[0]
        features = []
        
        # 提取基础信息
        hole_cards = x[:, self.num_players:self.num_players+52]
        board_cards = x[:, self.num_players+52:self.num_players+104]
        
        if self.manual_feature_size == 23:
            # 增强版本：23维特征（使用向量化计算，高性能）
            # 1. 起手牌强度（1维）- 加权2倍以提高重要性
            preflop_strength = vectorized_preflop_strength(hole_cards)
            preflop_strength_weighted = preflop_strength * 2.0  # 加权2倍
            # 确保加权后仍在合理范围内（0-2，但会被归一化）
            features.append(preflop_strength_weighted)
            
            # 2. 当前手牌强度（考虑公共牌）（1维）
            current_hand_strength = vectorized_hand_strength(hole_cards, board_cards)
            features.append(current_hand_strength)
            
            # 3. 是否中牌（1维）
            hit_hand = (current_hand_strength > 0.2).float()
            features.append(hit_hand)
            
            # 4. 成牌特征（3维）
            hit_pair, hit_trips, hit_two_pair = check_made_hands(hole_cards, board_cards)
            features.extend([hit_pair, hit_trips, hit_two_pair])
            
            # 5. 听牌特征（5维）
            flush_draw, flush_outs, straight_draw, straight_outs, draw_equity = check_draws(hole_cards, board_cards)
            features.extend([flush_draw, flush_outs, straight_draw, straight_outs, draw_equity])
            
            # 6. 公共牌特征（7维）
            board_strength, is_flush_board, is_straight_board, game_round = calculate_board_features(board_cards)
            features.extend([board_strength, is_flush_board, is_straight_board])
            features.append(game_round)  # 4维 one-hot
            
            # 7. 手牌强度变化（1维）- 使用tanh归一化到[-1, 1]，然后映射到[0, 1]
            strength_change_raw = current_hand_strength - preflop_strength
            # 使用tanh归一化（缩放因子3.0），然后映射到[0, 1]
            strength_change_normalized = (torch.tanh(strength_change_raw * 3.0) + 1.0) / 2.0
            features.append(strength_change_normalized)
            
            # 8. 下注统计特征（2维）- 如果board_cards有值，说明已经进入下注阶段
            action_seq_start = self.num_players + 104
            action_seq_end = action_seq_start + self.max_game_length * 2
            action_sizings_start = action_seq_end
            action_sizings = x[:, action_sizings_start:action_sizings_start+self.max_game_length]
            
            total_bet = torch.sum(action_sizings, dim=1, keepdim=True)
            max_bet = torch.max(action_sizings, dim=1, keepdim=True)[0]
            # 使用log归一化：log(1 + amount) / log(1 + max_stack)
            # 优点：小注值会更大，更接近其他特征，避免被稀释
            max_bet_norm = torch.log1p(max_bet) / self.log_max_stack
            total_bet_norm = torch.log1p(total_bet) / self.log_max_stack
            features.extend([max_bet_norm, total_bet_norm])
            
            # 9. 是否有人加注/全押（2维）
            action_seq = x[:, action_seq_start:action_seq_end]
            # 检查是否有raise (01) 或 all-in (11)
            has_raise = torch.any(
                (action_seq[:, 0::2] < 0.5) & (action_seq[:, 1::2] > 0.5) |
                (action_seq[:, 0::2] > 0.5) & (action_seq[:, 1::2] > 0.5),
                dim=1, keepdim=True
            ).float()
            has_allin = torch.any(
                (action_seq[:, 0::2] > 0.5) & (action_seq[:, 1::2] > 0.5),
                dim=1, keepdim=True
            ).float()
            features.extend([has_raise, has_allin])
            
            # 总计：1 + 1 + 1 + 3 + 5 + 3 + 4 + 1 + 2 + 2 = 23维
            return torch.cat(features, dim=1)
        
        elif self.manual_feature_size == 1:
            # 旧版本：只有手牌强度特征（1维）
            hand_strength_features = calculate_hand_strength_features(
                hole_cards, board_cards
            )
            return hand_strength_features  # 1维：手牌强度
        
        elif self.manual_feature_size == 7:
            # 老版本：7维特征（位置4维 + 手牌强度1维 + 下注统计2维）
            # 1. 位置特征（4维）
            try:
                from deep_cfr_with_feature_transform import calculate_position_advantage
                player_pos = x[:, 0:self.num_players]
                player_idx = torch.argmax(player_pos, dim=1, keepdim=True).float()
                position_features = calculate_position_advantage(player_idx, self.num_players)
                features.append(position_features)  # 4维
            except ImportError:
                # 如果没有该函数，使用零填充
                features.append(torch.zeros(batch_size, 4, device=x.device))
            
            # 2. 手牌强度特征（1维）
            hole_cards = x[:, self.num_players:self.num_players+52]
            board_cards = x[:, self.num_players+52:self.num_players+104]
            hand_strength_features = calculate_hand_strength_features(
                hole_cards, board_cards
            )
            features.append(hand_strength_features)  # 1维
            
            # 3. 下注统计特征（2维）
            action_seq_start = self.num_players + 104
            action_seq_end = action_seq_start + self.max_game_length * 2
            action_seq = x[:, action_seq_start:action_seq_end]
            action_sizings_start = action_seq_end
            action_sizings = x[:, action_sizings_start:action_sizings_start+self.max_game_length]
            
            total_bet = torch.sum(action_sizings, dim=1, keepdim=True)
            max_bet = torch.max(action_sizings, dim=1, keepdim=True)[0]
            max_bet_norm = max_bet / self.max_stack
            total_bet_norm = total_bet / self.max_stack
            
            features.extend([max_bet_norm, total_bet_norm])  # 2维
            
            return torch.cat(features, dim=1)  # 总共7维
        else:
            raise ValueError(f"不支持的特征维度: {self.manual_feature_size}，只支持1维、7维或25维")
    
    def forward(self, x, precomputed_features=None):
        """
        Args:
            x: 原始信息状态 [batch_size, raw_input_size] 或 
               已经包含特征的输入 [batch_size, raw_input_size + manual_feature_size]
            precomputed_features: 可选，预计算的手动特征 [batch_size, manual_feature_size]
        
        Returns:
            output: 网络输出 [batch_size, output_size]
        """
        # 如果输入是1D，添加batch维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # 快速路径：如果输入已经包含特征（维度 = raw_input_size + manual_feature_size）
        expected_combined_size = self.raw_input_size + self.manual_feature_size
        if x.shape[1] == expected_combined_size:
            # 输入已经是完整的，直接通过 MLP
            # 对 sizings 部分进行归一化
            x_norm = x.clone()
            action_sizings_start = self.num_players + 52 + 52 + self.max_game_length * 2
            if action_sizings_start < self.raw_input_size:
                x_norm[:, action_sizings_start:self.raw_input_size] = torch.log1p(
                    x_norm[:, action_sizings_start:self.raw_input_size]) / self.log_max_stack
            return self.mlp(x_norm)
        
        # 快速路径：如果提供了预计算的特征
        if precomputed_features is not None:
            x_norm = x.clone()
            action_sizings_start = self.num_players + 52 + 52 + self.max_game_length * 2
            if action_sizings_start < x.shape[1]:
                x_norm[:, action_sizings_start:] = torch.log1p(
                    x_norm[:, action_sizings_start:]) / self.log_max_stack
            combined = torch.cat([x_norm, precomputed_features], dim=1)
            return self.mlp(combined)
        
        # 自动适配输入维度 (Auto-adapt input dimension)
        # 当游戏配置(如筹码)变化导致 max_game_length 变化时，输入维度会变
        # 我们需要将其适配回训练时的维度 raw_input_size
        current_dim = x.shape[1]
        if current_dim != self.raw_input_size:
            # 计算 Header 大小: Player(N) + Private(52) + Public(52)
            header_size = self.num_players + 52 + 52
            
            # 计算 Old 和 New 的 max_game_length (L)
            # Dim = Header + 2*L + L = Header + 3*L
            L_old = (self.raw_input_size - header_size) // 3
            L_new = (current_dim - header_size) // 3
            
            # 只有当计算出的 L 也是整数且合理时才进行适配
            if (self.raw_input_size - header_size) % 3 == 0 and \
               (current_dim - header_size) % 3 == 0:
                   
                # print(f"【维度适配】检测到维度变化: {current_dim} -> {self.raw_input_size} (L: {L_new} -> {L_old})")
                
                # 分解 New Tensor
                header = x[:, :header_size]
                
                action_seq_start = header_size
                action_seq_len_new = 2 * L_new
                action_seq_new = x[:, action_seq_start : action_seq_start + action_seq_len_new]
                
                sizings_start = action_seq_start + action_seq_len_new
                sizings_len_new = L_new
                sizings_new = x[:, sizings_start : sizings_start + sizings_len_new]
                
                # 构造 Old Parts
                if L_new > L_old:
                    # 检查是否有非零数据被截断
                    truncated_actions = action_seq_new[:, int(2*L_old):]
                    truncated_sizings = sizings_new[:, int(L_old):]
                    
                    if torch.sum(truncated_actions) > 0 or torch.sum(truncated_sizings) > 0:
                        print(f"【警告】超长对局截断: 当前游戏长度超过模型训练上限 ({int(L_old)}步)。部分动作历史已丢失，可能影响决策。")
                    
                    # 截断 (保留前 L_old 个动作，通常是按时间顺序的)
                    # OpenSpiel 通常按顺序填充，尾部是 Padding (0)
                    action_seq_old = action_seq_new[:, :int(2*L_old)]
                    sizings_old = sizings_new[:, :int(L_old)]
                else:
                    # 填充 (Padding)
                    batch_size = x.shape[0]
                    device = x.device
                    
                    # Pad Action Seq
                    pad_len_action = 2 * (L_old - L_new)
                    zeros_action = torch.zeros(batch_size, pad_len_action, device=device)
                    action_seq_old = torch.cat([action_seq_new, zeros_action], dim=1)
                    
                    # Pad Sizings
                    pad_len_sizing = L_old - L_new
                    zeros_sizing = torch.zeros(batch_size, pad_len_sizing, device=device)
                    sizings_old = torch.cat([sizings_new, zeros_sizing], dim=1)
                
                # 重组
                x = torch.cat([header, action_seq_old, sizings_old], dim=1)

        # 验证输入维度
        assert x.shape[1] == self.raw_input_size, \
            f"输入维度不匹配: 期望 {self.raw_input_size}，实际 {x.shape[1]}"
        
        # 提取手动特征（1维：手牌强度）
        manual_features = self.extract_manual_features(x)
        
        # 验证手动特征维度
        assert manual_features.shape[1] == self.manual_feature_size, \
            f"手动特征维度错误: 期望 {self.manual_feature_size}，实际 {manual_features.shape[1]}"
        
        # 2. 对原始输入 x 进行归一化处理 (Critical Fix!)
        # 原始 x 中的 sizings 部分包含绝对金额 (0-20000)，必须归一化
        x_norm = x.clone()
        
        # 计算 sizings 的起始索引
        # x 的结构: [Player(N) | PrivateCards(52) | PublicCards(52) | ActionSeq(2*Len) | Sizings(Len)]
        # N = num_players (6)
        # Len = max_game_length
        # 3. 拼接：归一化后的原始信息状态 + 手动特征（1维：手牌强度）
        action_sizings_start = self.num_players + 52 + 52 + self.max_game_length * 2
        
        # 对 sizings 部分进行归一化
        # 使用log归一化：log(1 + amount) / log(1 + max_stack)
        # 优点：小注值会更大（0.43-0.57），更接近其他特征（0.0-2.0），避免被稀释
        if action_sizings_start < x.shape[1]:
            x_norm[:, action_sizings_start:] = torch.log1p(x_norm[:, action_sizings_start:]) / self.log_max_stack
             
            # #打印归一化后的 Sizings (Debug)
            # if self._print_counter <= 3:
            #     print(f"  [Debug] 归一化前的 Sizings (前5个样本):")
            #     print(x[:5, action_sizings_start:])
            #     print(f"  [Debug] 归一化后的 Sizings (前5个样本):")
            #     print(x_norm[:5, action_sizings_start:])

        combined = torch.cat([x_norm, manual_features], dim=1)
        
        # 验证拼接后的维度
        expected_combined_size = self.raw_input_size + self.manual_feature_size
        assert combined.shape[1] == expected_combined_size, \
            f"拼接后维度错误: 期望 {expected_combined_size}，实际 {combined.shape[1]}"
        
        # 通过MLP
        return self.mlp(combined)
    
    def reset(self):
        """重置网络（用于advantage network重新初始化）"""
        self.mlp.reset()


class DeepCFRSimpleFeature(deep_cfr.DeepCFRSolver):
    """简化版本：直接拼接手动特征的DeepCFR
    
    支持多 GPU 并行训练（DataParallel）
    """
    
    def __init__(self,
                 game,
                 policy_network_layers=(256, 256),
                 advantage_network_layers=(128, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 20,
                 learning_rate: float = 1e-4,
                 batch_size_advantage=None,
                 batch_size_strategy=None,
                 memory_capacity: int = int(1e6),
                 policy_network_train_steps: int = 1,
                 advantage_network_train_steps: int = 1,
                 reinitialize_advantage_networks: bool = True,
                 device=None,
                 multi_gpu: bool = False,
                 gpu_ids=None,
                 manual_feature_size=None):
        """
        与原始DeepCFRSolver相同，但使用SimpleFeatureMLP
        
        Args:
            multi_gpu: 是否使用多 GPU 并行
            gpu_ids: GPU ID 列表（None 表示使用所有可用 GPU）
            manual_feature_size: 手动特征维度（None时使用默认1维，7表示老版本）
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
        
        # 保存手动特征维度（用于创建网络）
        self._manual_feature_size = manual_feature_size
        
        # 多 GPU 设置
        self._multi_gpu = multi_gpu
        self._gpu_ids = gpu_ids
        
        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        
        # 从游戏配置中解析max_stack（用于归一化下注统计特征）
        import re
        game_string = str(game)
        match = re.search(r'stack=([\d\s]+)', game_string)
        max_stack = 2000  # 默认值
        if match:
            stack_str = match.group(1).strip()
            stack_values = stack_str.split()
            if stack_values:
                try:
                    max_stack = int(stack_values[0])
                except ValueError:
                    pass
        
        # 创建策略网络（使用SimpleFeatureMLP）
        # 支持手动指定特征维度（用于兼容老模型）
        manual_feature_size = getattr(self, '_manual_feature_size', None)
        self._strategy_memories = deep_cfr.ReservoirBuffer(memory_capacity)
        policy_net = SimpleFeatureMLP(
            self._embedding_size,
            list(policy_network_layers),
            self._num_actions,
            num_players=self._num_players,
            max_game_length=game.max_game_length(),
            max_stack=max_stack,
            manual_feature_size=manual_feature_size  # 传递特征维度
        )
        
        # 多 GPU 包装
        if multi_gpu and gpu_ids is not None and len(gpu_ids) > 1:
            self._policy_network = wrap_with_data_parallel(policy_net, self._device, gpu_ids)
        else:
            self._policy_network = policy_net.to(self._device)
        
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)
        
        # 创建优势网络（每个玩家一个，使用SimpleFeatureMLP）
        self._advantage_memories = [
            deep_cfr.ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._advantage_networks = []
        for _ in range(self._num_players):
            adv_net = SimpleFeatureMLP(
                self._embedding_size,
                list(advantage_network_layers),
                self._num_actions,
                num_players=self._num_players,
                max_game_length=game.max_game_length(),
                max_stack=max_stack
            )
            # 多 GPU 包装
            if multi_gpu and gpu_ids is not None and len(gpu_ids) > 1:
                adv_net = wrap_with_data_parallel(adv_net, self._device, gpu_ids)
            else:
                adv_net = adv_net.to(self._device)
            self._advantage_networks.append(adv_net)
        
        self._loss_advantages = nn.MSELoss(reduction="mean")
        self._optimizer_advantages = []
        for p in range(self._num_players):
            self._optimizer_advantages.append(
                torch.optim.Adam(
                    self._advantage_networks[p].parameters(), lr=learning_rate))
        self._learning_rate = learning_rate
    
    def reinitialize_advantage_network(self, player):
        """重新初始化优势网络"""
        # 获取原始网络（如果被 DataParallel 包装）
        adv_net = unwrap_data_parallel(self._advantage_networks[player])
        adv_net.reset()
        
        # 重新包装（如果需要多 GPU）
        if self._multi_gpu and self._gpu_ids is not None and len(self._gpu_ids) > 1:
            self._advantage_networks[player] = wrap_with_data_parallel(
                adv_net, self._device, self._gpu_ids
            )
        else:
            self._advantage_networks[player] = adv_net.to(self._device)
        
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self._learning_rate)


if __name__ == "__main__":
    """测试简单版本"""
    import pyspiel
    import numpy as np
    
    # 创建游戏
    game_config = {
        "numPlayers": 6,
        "numBoardCards": "0 3 1 1",
        "numRanks": 13,
        "numSuits": 4,
        "firstPlayer": "2",
        "stack": "20000 20000 20000 20000 20000 20000",
        "blind": "50 100 0 0 0 0",
        "numHoleCards": 2,
        "numRounds": 4,
        "betting": "nolimit",
        "maxRaises": "3",
    }
    game = pyspiel.load_game("universal_poker", game_config)
    
    print("=" * 60)
    print("简化版本：直接拼接手动特征")
    print("=" * 60)
    
    state = game.new_initial_state()
    raw_size = len(state.information_state_tensor(0))
    print(f"原始信息状态大小: {raw_size}")
    print(f"手动特征大小: 1")
    print(f"合并后输入大小: {raw_size + 1}")
    
    # 创建solver
    solver = DeepCFRSimpleFeature(
        game,
        policy_network_layers=(256, 256),
        advantage_network_layers=(128, 128),
        num_iterations=10,
        num_traversals=5,
        learning_rate=1e-4,
        device=torch.device("cpu")
    )
    
    print(f"\n策略网络输入维度: {raw_size + 1}")
    print(f"优势网络输入维度: {raw_size + 1}")
    print("\n✓ 简化版本创建成功！")
    print("\n使用方法：")
    print("在 train_deep_cfr_texas.py 中：")
    print("  from deep_cfr_simple_feature import DeepCFRSimpleFeature")
    print("  solver = DeepCFRSimpleFeature(...)")

    # Add test forward pass to trigger feature printing
    print("\n" + "=" * 60)
    print("Testing Forward Pass (Trigger Feature Printing)")
    print("=" * 60)
    
    info_state = state.information_state_tensor(0)
    # Create a dummy batch of size 2 to test batch processing
    info_state_tensor = torch.FloatTensor(np.array([info_state, info_state]))
    
    print("Running policy network forward pass...")
    with torch.no_grad():
        # This should trigger the print inside extract_manual_features
        output = solver._policy_network(info_state_tensor)
        
    print("\nDone.")


