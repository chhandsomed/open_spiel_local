"""
简化版本：直接拼接手动特征到原始信息状态

流程：
信息状态(281维) + 手动特征(7维) = 288维 -> MLP(288, [256, 256], num_actions)
"""

import torch
import torch.nn as nn
import numpy as np
from open_spiel.python.pytorch import deep_cfr
from deep_cfr_with_feature_transform import (
    calculate_position_advantage,
    calculate_hand_strength_features
)


class SimpleFeatureMLP(nn.Module):
    """简单的MLP，在输入前拼接手动特征"""
    
    def __init__(self, raw_input_size, hidden_sizes, output_size, 
                 num_players=6, max_game_length=52, activate_final=False):
        """
        Args:
            raw_input_size: 原始信息状态大小（例如281）
            hidden_sizes: MLP隐藏层大小列表（例如[256, 256]）
            output_size: 输出大小（动作数量）
            num_players: 玩家数量
            max_game_length: 最大游戏长度
            activate_final: 最后一层是否使用激活函数
        """
        super(SimpleFeatureMLP, self).__init__()
        self.num_players = num_players
        self.max_game_length = max_game_length
        self.manual_feature_size = 7  # 手动特征维度
        
        # 输入维度 = 原始信息状态 + 手动特征
        input_size = raw_input_size + self.manual_feature_size
        
        # 使用原始的MLP结构
        self.mlp = deep_cfr.MLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final=activate_final
        )
    
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
        action_seq_start = self.num_players + 104
        action_seq_end = action_seq_start + self.max_game_length * 2
        action_seq = x[:, action_seq_start:action_seq_end]
        action_sizings_start = action_seq_end
        action_sizings = x[:, action_sizings_start:action_sizings_start+self.max_game_length]
        
        total_bet = torch.sum(action_sizings, dim=1, keepdim=True)
        max_bet = torch.max(action_sizings, dim=1, keepdim=True)[0]
        max_bet_norm = max_bet / 20000.0
        total_bet_norm = total_bet / 20000.0
        
        features.extend([max_bet_norm, total_bet_norm])  # 2维
        
        return torch.cat(features, dim=1)  # 总共7维
    
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
        
        # 提取手动特征
        manual_features = self.extract_manual_features(x)
        
        # 拼接：原始信息状态 + 手动特征
        combined = torch.cat([x, manual_features], dim=1)
        
        # 通过MLP
        return self.mlp(combined)
    
    def reset(self):
        """重置网络（用于advantage network重新初始化）"""
        self.mlp.reset()


class DeepCFRSimpleFeature(deep_cfr.DeepCFRSolver):
    """简化版本：直接拼接手动特征的DeepCFR"""
    
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
                 device=None):
        """
        与原始DeepCFRSolver相同，但使用SimpleFeatureMLP
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
        
        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        
        # 创建策略网络（使用SimpleFeatureMLP）
        self._strategy_memories = deep_cfr.ReservoirBuffer(memory_capacity)
        self._policy_network = SimpleFeatureMLP(
            self._embedding_size,
            list(policy_network_layers),
            self._num_actions,
            num_players=self._num_players,
            max_game_length=game.max_game_length()
        )
        self._policy_network = self._policy_network.to(self._device)
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)
        
        # 创建优势网络（每个玩家一个，使用SimpleFeatureMLP）
        self._advantage_memories = [
            deep_cfr.ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._advantage_networks = [
            SimpleFeatureMLP(
                self._embedding_size,
                list(advantage_network_layers),
                self._num_actions,
                num_players=self._num_players,
                max_game_length=game.max_game_length()
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
        """重新初始化优势网络"""
        self._advantage_networks[player].reset()
        self._advantage_networks[player] = self._advantage_networks[player].to(self._device)
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

