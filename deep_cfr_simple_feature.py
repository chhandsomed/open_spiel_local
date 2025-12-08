"""
简化版本：直接拼接手动特征到原始信息状态

流程：
信息状态(raw_input_size维) + 手动特征(7维) = (raw_input_size+7)维 -> MLP

手动特征组成：
- 位置特征 (4维): 位置优势分、是否前位、是否后位、是否盲注
- 手牌强度 (1维): 起手牌强度 (EHS)
- 下注统计 (2维): 最大下注、总下注（从sizings中提取）

注意：OpenSpiel 1.6.9中，信息状态维度包含sizings部分，Bet动作后sizings包含实际下注金额

对于6人fchpa: 281 + 7 = 288维 (OpenSpiel 1.6.9)
对于6人fcpa: 266 + 7 = 273维
对于2人fchpa: 175 + 7 = 182维

支持多 GPU 并行训练（DataParallel）
"""

import torch
import torch.nn as nn
import numpy as np
from open_spiel.python.pytorch import deep_cfr
from deep_cfr_with_feature_transform import (
    calculate_position_advantage,
    calculate_hand_strength_features
)


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


class SimpleFeatureMLP(nn.Module):
    """简单的MLP，在输入前拼接手动特征"""
    
    def __init__(self, raw_input_size, hidden_sizes, output_size, 
                 num_players=6, max_game_length=52, activate_final=False, max_stack=2000):
        """
        Args:
            raw_input_size: 原始信息状态大小（例如224，取决于游戏配置）
            hidden_sizes: MLP隐藏层大小列表（例如[256, 256]）
            output_size: 输出大小（动作数量）
            num_players: 玩家数量
            max_game_length: 最大游戏长度
            activate_final: 最后一层是否使用激活函数
            max_stack: 单个玩家的最大筹码量（用于归一化下注统计特征，默认2000）
        """
        super(SimpleFeatureMLP, self).__init__()
        self.num_players = num_players
        self.max_game_length = max_game_length
        self.max_stack = float(max_stack)  # 用于归一化下注统计特征
        self.manual_feature_size = 7  # 手动特征维度：位置(4) + 手牌强度(1) + 下注统计(2)
        
        # 输入维度 = 原始信息状态 + 手动特征（7维：位置4 + 手牌强度1 + 下注统计2）
        input_size = raw_input_size + self.manual_feature_size
        
        # 验证：确保没有意外添加额外的特征
        assert input_size == raw_input_size + 7, \
            f"输入维度错误: 期望 {raw_input_size + 7}，实际计算为 {input_size}"
        
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
        """提取手动特征（与HybridFeatureTransform中的相同）"""
        batch_size = x.shape[0]
        features = []
        
        # 1. 位置特征
        player_pos = x[:, 0:self.num_players]
        player_idx = torch.argmax(player_pos, dim=1, keepdim=True).float()
        position_features = calculate_position_advantage(player_idx, self.num_players)
        features.append(position_features)  # 4维
        
        # 2. 手牌强度特征
        hole_cards = x[:, self.num_players:self.num_players+52]
        board_cards = x[:, self.num_players+52:self.num_players+104]
        hand_strength_features = calculate_hand_strength_features(
            hole_cards, board_cards
        )
        features.append(hand_strength_features)  # 1维
        
        # 3. 下注统计特征
        # ⚠️ 重要说明：OpenSpiel 1.6.9中，信息状态维度包含sizings部分（max_game_length维）
        # - sizings存储每个动作对应的下注金额
        # - Bet动作后，sizings记录实际下注金额（例如：Bet350 → sizings=350.0）
        # - Call动作的sizings为0（根据源码注释）
        # - 训练时，大部分状态都会有下注动作，sizings会有实际值
        # 实际格式：玩家位置(6) + 手牌(52) + 公共牌(52) + 动作序列(2*max_game_length) + sizings(max_game_length)
        action_seq_start = self.num_players + 104
        action_seq_end = action_seq_start + self.max_game_length * 2
        action_seq = x[:, action_seq_start:action_seq_end]
        action_sizings_start = action_seq_end
        action_sizings = x[:, action_sizings_start:action_sizings_start+self.max_game_length]
        
        # 提取下注统计特征：最大下注和总下注（归一化）
        total_bet = torch.sum(action_sizings, dim=1, keepdim=True)
        max_bet = torch.max(action_sizings, dim=1, keepdim=True)[0]
        # 使用实际的max_stack进行归一化（All-in时max_bet = max_stack，归一化后 = 1.0）
        max_bet_norm = max_bet / self.max_stack
        total_bet_norm = total_bet / self.max_stack
        
        features.extend([max_bet_norm, total_bet_norm])  # 2维
        
        combined_features = torch.cat(features, dim=1)  # 总共7维：位置(4) + 手牌强度(1) + 下注统计(2)
        
        # 打印特征信息（用于调试）- 增强版，中文显示所有特征
        if not hasattr(self, '_print_counter'):
            self._print_counter = 0
        
        self._print_counter += 1
        # 前10次打印，之后每1000次打印一次
        should_print = (self._print_counter <= 10) or (self._print_counter % 1000 == 0)
        
        # if should_print:
        #     try:
        #         print("\n" + "="*80)
        #         print(f"【特征提取详情】第 {self._print_counter} 次调用")
        #         print("="*80)
        #         print(f"批次大小: {batch_size}")
        #         print(f"总输入维度: {x.shape[1]} (原始信息状态) + {self.manual_feature_size} (手动特征) = {x.shape[1] + self.manual_feature_size}")
        #         print(f"⚠️ 注意: OpenSpiel 1.6.9中维度包含sizings部分，Bet动作后sizings包含实际下注金额")
                
        #         # ========== 解析原始信息状态特征 ==========
        #         print(f"\n【原始信息状态特征解析】")
        #         idx = 0
                
        #         # A. 玩家位置 (One-hot)
        #         p_len = self.num_players
        #         player_pos_vec = x[0, idx:idx+p_len].detach().cpu().numpy()
        #         current_player = np.argmax(player_pos_vec)
        #         print(f"\n1. 当前玩家标识 (索引 {idx}-{idx+p_len-1}, 共{p_len}维)")
        #         print(f"   - One-hot向量: {player_pos_vec}")
        #         print(f"   - 当前玩家ID: {current_player}")
        #         if self.num_players == 6:
        #             position_names = ["SB(小盲)", "BB(大盲)", "UTG", "MP", "CO", "BTN(按钮)"]
        #             print(f"   - 位置名称: {position_names[current_player]}")
        #         idx += p_len
                
        #         # B. 私有手牌 (52张牌 One-hot)
        #         hc_len = 52
        #         hc_vec = x[0, idx:idx+hc_len].detach().cpu().numpy()
        #         hc_indices = np.where(hc_vec == 1)[0]
        #         print(f"\n2. 私有手牌 (索引 {idx}-{idx+hc_len-1}, 共{hc_len}维)")
        #         if len(hc_indices) > 0:
        #             print(f"   - 激活的牌索引: {hc_indices}")
        #             # 转换为牌面显示
        #             rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        #             suit_names = ['♠', '♥', '♦', '♣']
        #             cards = []
        #             for card_idx in hc_indices:
        #                 rank = card_idx // 4
        #                 suit = card_idx % 4
        #                 cards.append(f"{rank_names[rank]}{suit_names[suit]}")
        #             print(f"   - 手牌: {', '.join(cards)}")
        #         else:
        #             print(f"   - 当前状态无私有手牌信息（可能是游戏开始前）")
        #         idx += hc_len
                
        #         # C. 公共牌 (52张牌 One-hot)
        #         bc_len = 52
        #         bc_vec = x[0, idx:idx+bc_len].detach().cpu().numpy()
        #         bc_indices = np.where(bc_vec == 1)[0]
        #         print(f"\n3. 公共牌 (索引 {idx}-{idx+bc_len-1}, 共{bc_len}维)")
        #         if len(bc_indices) > 0:
        #             print(f"   - 激活的牌索引: {bc_indices}")
        #             rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        #             suit_names = ['♠', '♥', '♦', '♣']
        #             cards = []
        #             for card_idx in bc_indices:
        #                 rank = card_idx // 4
        #                 suit = card_idx % 4
        #                 cards.append(f"{rank_names[rank]}{suit_names[suit]}")
        #             print(f"   - 公共牌: {', '.join(cards)}")
        #         else:
        #             print(f"   - 当前状态无公共牌（Preflop阶段）")
        #         idx += bc_len
                
        #         # D. 动作序列 (Action Sequence)
        #         act_seq_len = 2 * self.max_game_length
        #         action_seq = x[0, idx:idx+act_seq_len].detach().cpu().numpy()
        #         print(f"\n4. 动作序列 (索引 {idx}-{idx+act_seq_len-1}, 共{act_seq_len}维)")
        #         print(f"   - 动作序列长度: {act_seq_len} (每2个bit表示一个动作)")
        #         # 统计非零动作
        #         non_zero_actions = 0
        #         for i in range(0, act_seq_len, 2):
        #             if action_seq[i] != 0 or action_seq[i+1] != 0:
        #                 non_zero_actions += 1
        #         print(f"   - 已执行动作数量: {non_zero_actions}/{self.max_game_length}")
        #         idx += act_seq_len
                
        #         # E. Sizings (下注金额)
        #         sizings_len = self.max_game_length
        #         sizings = x[0, idx:idx+sizings_len].detach().cpu().numpy()
        #         non_zero_sizings = sizings[sizings != 0]
        #         print(f"\n5. Sizings下注金额 (索引 {idx}-{idx+sizings_len-1}, 共{sizings_len}维)")
        #         print(f"   - 非零下注数量: {len(non_zero_sizings)}")
        #         if len(non_zero_sizings) > 0:
        #             print(f"   - 下注金额列表: {non_zero_sizings[:10]}")  # 显示前10个
        #             print(f"   - 最大下注: {np.max(non_zero_sizings):.1f}")
        #             print(f"   - 总下注: {np.sum(non_zero_sizings):.1f}")
        #         else:
        #             print(f"   - 当前状态无下注动作（sizings全为0，可能是游戏开始或Call动作）")

        #         # ========== 手动提取的特征 ==========
        #         print(f"\n【手动提取的特征】({self.manual_feature_size}维，将拼接到原始特征后)")
                
        #         # Position
        #         pos_vals = position_features[0].detach().cpu().numpy()
        #         print(f"\nA. 位置特征 (4维): {pos_vals}") 
        #         print(f"   - 位置优势分: {pos_vals[0]:.4f} (0.0=最弱位置, 1.0=最强位置)")
        #         print(f"   - 是否前位:   {pos_vals[1]:.1f} (1.0=是前位UTG/MP, 0.0=不是)")
        #         print(f"   - 是否后位:   {pos_vals[2]:.1f} (1.0=是后位CO/BTN, 0.0=不是)")
        #         print(f"   - 是否盲注:   {pos_vals[3]:.1f} (1.0=是盲注SB/BB, 0.0=不是)")
                
        #         # Hand Strength
        #         hs_vals = hand_strength_features[0].detach().cpu().numpy()
        #         print(f"\nB. 手牌强度特征 (1维): {hs_vals}")
        #         print(f"   - 起手牌强度(EHS): {hs_vals[0]:.4f} (0.0=最弱如72o, 1.0=最强如AA)")
                
        #         # Betting Statistics
        #         max_bet_val = max_bet_norm[0].item()
        #         total_bet_val = total_bet_norm[0].item()
        #         max_bet_raw = max_bet[0].item()
        #         total_bet_raw = total_bet[0].item()
        #         print(f"\nC. 下注统计特征 (2维):")
        #         print(f"   - 最大下注(归一化): {max_bet_val:.4f} (原始值: {max_bet_raw:.1f} 筹码)")
        #         print(f"   - 总下注(归一化):   {total_bet_val:.4f} (原始值: {total_bet_raw:.1f} 筹码)")
        #         print(f"   - 归一化基准: {self.max_stack:.0f}筹码 = 1.0 (All-in时最大下注=1.0)")
                
        #         # ========== 特征汇总 ==========
        #         print(f"\n【特征汇总】")
        #         print(f"原始信息状态维度: {x.shape[1]}")
        #         print(f"  ├─ 玩家位置: {self.num_players}维")
        #         print(f"  ├─ 私有手牌: 52维")
        #         print(f"  ├─ 公共牌: 52维")
        #         print(f"  ├─ 动作序列: {2 * self.max_game_length}维")
        #         print(f"  └─ Sizings: {self.max_game_length}维")
        #         print(f"手动特征维度: {self.manual_feature_size}")
        #         print(f"  ├─ 位置特征: 4维")
        #         print(f"  ├─ 手牌强度: 1维")
        #         print(f"  └─ 下注统计: 2维")
        #         print(f"最终输入MLP维度: {x.shape[1] + self.manual_feature_size}")
                
        #         print("="*80 + "\n")
        #     except Exception as e:
        #         print(f"【错误】打印特征时出错: {e}")
        #         import traceback
        #         traceback.print_exc()

        return combined_features
    
    def forward(self, x):
        """
        Args:
            x: 原始信息状态 [batch_size, raw_input_size]
        
        Returns:
            output: 网络输出 [batch_size, output_size]
        """
        # 如果输入是1D，添加batch维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # 验证输入维度
        assert x.shape[1] == self.raw_input_size, \
            f"输入维度不匹配: 期望 {self.raw_input_size}，实际 {x.shape[1]}"
        
        # 提取手动特征（7维：位置4 + 手牌强度1 + 下注统计2）
        manual_features = self.extract_manual_features(x)
        
        # 验证手动特征维度（应该是7维：位置4 + 手牌强度1 + 下注统计2）
        assert manual_features.shape[1] == self.manual_feature_size, \
            f"手动特征维度错误: 期望 {self.manual_feature_size}，实际 {manual_features.shape[1]}"
        assert self.manual_feature_size == 7, \
            f"手动特征维度应该是7维（位置4 + 手牌强度1 + 下注统计2），当前为 {self.manual_feature_size}"
        
        # 2. 对原始输入 x 进行归一化处理 (Critical Fix!)
        # 原始 x 中的 sizings 部分包含绝对金额 (0-20000)，必须归一化
        x_norm = x.clone()
        
        # 计算 sizings 的起始索引
        # x 的结构: [Player(N) | PrivateCards(52) | PublicCards(52) | ActionSeq(2*Len) | Sizings(Len)]
        # N = num_players (6)
        # Len = max_game_length
        # 3. 拼接：归一化后的原始信息状态 + 手动特征（7维：位置4 + 手牌强度1 + 下注统计2）
        action_sizings_start = self.num_players + 52 + 52 + self.max_game_length * 2
        
        # 对 sizings 部分进行归一化
        if action_sizings_start < x.shape[1]:
            x_norm[:, action_sizings_start:] = x_norm[:, action_sizings_start:] / self.max_stack
             
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
                 gpu_ids=None):
        """
        与原始DeepCFRSolver相同，但使用SimpleFeatureMLP
        
        Args:
            multi_gpu: 是否使用多 GPU 并行
            gpu_ids: GPU ID 列表（None 表示使用所有可用 GPU）
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
        self._strategy_memories = deep_cfr.ReservoirBuffer(memory_capacity)
        policy_net = SimpleFeatureMLP(
            self._embedding_size,
            list(policy_network_layers),
            self._num_actions,
            num_players=self._num_players,
            max_game_length=game.max_game_length(),
            max_stack=max_stack
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
    print(f"手动特征大小: 7")
    print(f"合并后输入大小: {raw_size + 7}")
    
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
    
    print(f"\n策略网络输入维度: {raw_size + 7}")
    print(f"优势网络输入维度: {raw_size + 7}")
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


