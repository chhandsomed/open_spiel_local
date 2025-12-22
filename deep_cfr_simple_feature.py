"""
简化版本：直接拼接手动特征到原始信息状态

流程：
信息状态(raw_input_size维) + 手动特征(1维) = (raw_input_size+1)维 -> MLP

手动特征组成：
- 手牌强度 (1维): 起手牌强度 (EHS)

注意：OpenSpiel 1.6.9中，信息状态维度包含sizings部分，Bet动作后sizings包含实际下注金额

对于6人fchpa: 281 + 1 = 282维 (OpenSpiel 1.6.9)
对于6人fcpa: 266 + 1 = 267维
对于2人fchpa: 175 + 1 = 176维

支持多 GPU 并行训练（DataParallel）
"""

import torch
import torch.nn as nn
import numpy as np
from open_spiel.python.pytorch import deep_cfr
from deep_cfr_with_feature_transform import calculate_hand_strength_features


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
                    return None
    
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
        
        # 如果未指定，默认使用新版本（1维）
        if manual_feature_size is None:
            self.manual_feature_size = 1  # 新版本：只有手牌强度(1维)
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
        - 新版本（1维）：只提取手牌强度
        - 老版本（7维）：位置特征(4维) + 手牌强度(1维) + 下注统计(2维)
        """
        batch_size = x.shape[0]
        features = []
        
        if self.manual_feature_size == 1:
            # 新版本：只有手牌强度特征（1维）
            hole_cards = x[:, self.num_players:self.num_players+52]
            board_cards = x[:, self.num_players+52:self.num_players+104]
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
            raise ValueError(f"不支持的特征维度: {self.manual_feature_size}，只支持1维或7维")
    
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


