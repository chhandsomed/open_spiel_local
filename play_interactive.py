#!/usr/bin/env python3
"""交互式对局脚本 - 与训练好的模型对打"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import sys
import argparse
import torch
import numpy as np
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

# 尝试导入简单特征版本
try:
    from deep_cfr_simple_feature import DeepCFRSimpleFeature
    USE_SIMPLE_FEATURE = True
except ImportError:
    USE_SIMPLE_FEATURE = False
    from open_spiel.python.pytorch.deep_cfr import MLP


def load_model(model_dir, num_players=None, device='cpu'):
    """加载训练好的模型"""
    print(f"\n[1/2] 加载模型: {model_dir}")
    
    # 读取配置文件
    config_path = os.path.join(model_dir, 'config.json')
    config = {}
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  ✓ 读取配置文件")
        
        # 从配置获取玩家数量
        if num_players is None:
            num_players = config.get('num_players', 6)
        
        # 从配置获取模型类型
        use_simple_feature = config.get('use_simple_feature', False)
        use_feature_transform = config.get('use_feature_transform', False)
        policy_layers = tuple(config.get('policy_layers', [64, 64]))
        
        # 获取保存前缀
        save_prefix = config.get('save_prefix', 'deepcfr_texas')
        
        # 获取 betting_abstraction
        betting_abstraction = config.get('betting_abstraction', 'fcpa')
        
        # 获取 game_string
        game_string = config.get('game_string', None)
    else:
        print(f"  ⚠️ 配置文件不存在，使用默认值")
        if num_players is None:
            num_players = 6
        use_simple_feature = False
        use_feature_transform = False
        policy_layers = (64, 64)
        save_prefix = 'deepcfr_texas'
        betting_abstraction = 'fcpa'
        game_string = None
    
    # 创建游戏（必须与训练时一致）
    game = None
    
    # 优先使用 game_string
    if game_string:
        try:
            print(f"  使用 game_string 创建游戏: {game_string}")
            game = pyspiel.load_game(game_string)
        except Exception as e:
            print(f"  ⚠️ 使用 game_string 创建游戏失败: {e}，尝试手动配置")
            game = None
    
    if game is None:
        game_config = {
            'numPlayers': num_players,
            'numBoardCards': '0 3 1 1',
            'numRanks': 13,
            'numSuits': 4,
            'firstPlayer': '2',
            'stack': '2000 2000 2000 2000 2000 2000',
            'blind': '100 100 100 100 100 100',
            'numHoleCards': 2,
            'numRounds': 4,
            'betting': 'nolimit',
            'maxRaises': '3',
            'bettingAbstraction': betting_abstraction, # 使用读取到的配置
        }
        
        # 修正盲注配置（如果 num_players 是 6）
        if num_players == 6:
            # P0=SB(50), P1=BB(100)
            game_config['blind'] = "50 100 0 0 0 0"
            # P2=UTG acts first preflop (index 3), P0=SB acts first postflop (index 1)
            game_config['firstPlayer'] = "3 1 1 1"
        elif num_players == 2:
            game_config['blind'] = "100 50"
            game_config['firstPlayer'] = "2 1 1 1"
        
        game = pyspiel.load_game('universal_poker', game_config)
    
    # 加载模型
    # 优先使用 config 中的 prefix，否则尝试默认名称
    policy_filename = f"{save_prefix}_policy_network.pt"
    policy_path = os.path.join(model_dir, policy_filename)
    
    # 如果找不到，尝试旧的默认名称作为回退
    if not os.path.exists(policy_path):
        fallback_path = os.path.join(model_dir, 'deepcfr_texas_policy_network.pt')
        if os.path.exists(fallback_path):
            print(f"  ⚠️ 未找到 {policy_filename}，尝试加载 {os.path.basename(fallback_path)}")
            policy_path = fallback_path
    
    if not os.path.exists(policy_path):
        print(f"  ✗ 模型文件不存在: {policy_path}")
        return None, None
    
    # 根据配置选择模型类型
    if use_simple_feature and USE_SIMPLE_FEATURE:
        # 使用简单特征版本
        print(f"  使用简单特征版本（266维 + 7维特征）")
        solver = DeepCFRSimpleFeature(
            game,
            policy_network_layers=policy_layers,
            advantage_network_layers=(32, 32),
            num_iterations=1,
            num_traversals=1,
            learning_rate=1e-4,
            device=device
        )
        solver._policy_network.load_state_dict(
            torch.load(policy_path, map_location=device)
        )
        solver._policy_network.eval()
        print(f"  ✓ 模型加载成功（简单特征版本）")
        return game, solver
    elif use_feature_transform and USE_SIMPLE_FEATURE:
        # 使用复杂特征转换版本
        print(f"  使用复杂特征转换版本")
        try:
            from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
            transformed_size = config.get('transformed_size', 150)
            use_hybrid_transform = config.get('use_hybrid_transform', True)
            
            solver = DeepCFRWithFeatureTransform(
                game,
                policy_network_layers=policy_layers,
                advantage_network_layers=(32, 32),
                num_iterations=1,
                num_traversals=1,
                learning_rate=1e-4,
                transformed_size=transformed_size,
                use_hybrid_transform=use_hybrid_transform,
                device=device
            )
            solver._policy_network.load_state_dict(
                torch.load(policy_path, map_location=device)
            )
            solver._policy_network.eval()
            print(f"  ✓ 模型加载成功（复杂特征转换版本）")
            return game, solver
        except ImportError:
            print(f"  ⚠️ 无法导入复杂特征转换版本，尝试标准版本")
    
    # 使用标准版本
    print(f"  使用标准版本")
    state = game.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    network = MLP(embedding_size, list(policy_layers), num_actions)
    network = network.to(device)
    
    try:
        network.load_state_dict(torch.load(policy_path, map_location=device))
        network.eval()
        print(f"  ✓ 模型加载成功（标准版本）")
        return game, network
    except RuntimeError as e:
        print(f"  ✗ 模型加载失败: {e}")
        print(f"  提示: 模型类型可能不匹配，请检查配置文件")
        return None, None


def get_model_action(state, model, device, player):
    """获取模型的动作"""
    # 检查是否是 DeepCFRSolver 类型（有 action_probabilities 方法）
    if hasattr(model, 'action_probabilities'):
        # 使用求解器的 action_probabilities
        try:
            probs = model.action_probabilities(state, player)
            actions = list(probs.keys())
            probabilities = np.array([probs[a] for a in actions])
            probabilities = probabilities / probabilities.sum()
            action = np.random.choice(actions, p=probabilities)
            return action, probs
        except Exception as e:
            print(f"  ⚠️ 使用 action_probabilities 失败: {e}，尝试直接使用网络")
    
    # 使用网络直接预测
    info_state = state.information_state_tensor(player)
    legal_actions = state.legal_actions(player)
    
    info_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
    
    with torch.no_grad():
        # 如果是求解器，使用其策略网络
        if hasattr(model, '_policy_network'):
            network = model._policy_network
        else:
            network = model
        
        logits = network(info_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    action_probs = {a: float(probs[a]) for a in legal_actions}
    total = sum(action_probs.values())
    if total > 1e-10:
        action_probs = {a: p/total for a, p in action_probs.items()}
    else:
        action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
    
    actions = list(action_probs.keys())
    probabilities = np.array([action_probs[a] for a in actions])
    probabilities = probabilities / probabilities.sum()
    action = np.random.choice(actions, p=probabilities)
    
    return action, action_probs


def format_card(card_idx):
    """格式化牌面"""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♥', '♦', '♣']
    
    rank = ranks[card_idx // 4]
    suit = suits[card_idx % 4]
    return f"{rank}{suit}"


def get_game_state_info(state, board_cards_history=None):
    """获取游戏状态信息
    
    Args:
        state: 游戏状态
        board_cards_history: 公共牌历史（用于跟踪发牌顺序）
    """
    try:
        state_struct = state.to_struct()
        
        # 使用 getattr 访问属性（state_struct 是对象，不是字典）
        board_cards_str = getattr(state_struct, 'board_cards', '')
        player_hands = getattr(state_struct, 'player_hands', [])
        pot = getattr(state_struct, 'pot_size', 0)
        betting_history = getattr(state_struct, 'betting_history', '')
        
        info = {
            'round': None,
            'board_cards': [],
            'board_cards_by_round': {},  # 按轮次分组的公共牌
            'player_hands': {},
            'pot': pot,
            'current_player': state.current_player() if not state.is_chance_node() else None,
            'betting_history': betting_history,
        }
        
        # 获取公共牌（board_cards 是字符串格式，每张牌2个字符，如 "Kh2d2c"）
        if board_cards_str:
            # 每张牌是2个字符（如 "Kh", "2d", "2c"）
            # 需要按2个字符一组分割
            cards = []
            for i in range(0, len(board_cards_str), 2):
                if i + 2 <= len(board_cards_str):
                    cards.append(board_cards_str[i:i+2])
            info['board_cards'] = cards
            
            # 按轮次分组公共牌（如果有历史记录）
            if board_cards_history is not None:
                prev_count = len(board_cards_history) if board_cards_history else 0
                current_count = len(cards)
                if current_count > prev_count:
                    # 新发的牌
                    new_cards = cards[prev_count:]
                    if prev_count == 0:
                        info['board_cards_by_round']['Flop'] = new_cards
                    elif prev_count == 3:
                        info['board_cards_by_round']['Turn'] = new_cards
                    elif prev_count == 4:
                        info['board_cards_by_round']['River'] = new_cards
        
        # 获取玩家手牌（player_hands 是字符串列表，每个元素如 "AhTc"）
        if player_hands:
            for player, hand_str in enumerate(player_hands):
                if hand_str:
                    # hand_str 是字符串格式，每张牌2个字符，如 "AhTc"
                    hand_cards = []
                    for i in range(0, len(hand_str), 2):
                        if i + 2 <= len(hand_str):
                            hand_cards.append(hand_str[i:i+2])
                    info['player_hands'][player] = hand_cards
        
        # 判断轮次（根据公共牌数量）
        board_count = len(info['board_cards'])
        if board_count == 0:
            info['round'] = "Preflop"
        elif board_count == 3:
            info['round'] = "Flop"
        elif board_count == 4:
            info['round'] = "Turn"
        elif board_count == 5:
            info['round'] = "River"
        else:
            # 如果数量异常，尝试从 betting_history 判断
            # betting_history 格式: "r100c/r200c/..." 用 "/" 分隔轮次
            if betting_history:
                rounds = betting_history.split('/')
                round_num = len(rounds) - 1  # 减1因为第一轮可能没有 "/"
                if round_num == 0:
                    info['round'] = "Preflop"
                elif round_num == 1:
                    info['round'] = "Flop"
                elif round_num == 2:
                    info['round'] = "Turn"
                elif round_num == 3:
                    info['round'] = "River"
                else:
                    info['round'] = f"Round {round_num}"
            else:
                info['round'] = f"Round {board_count}"
        
        return info
    except Exception as e:
        # 如果获取失败，返回基本信息
        return {
            'round': "Unknown",
            'board_cards': [],
            'board_cards_by_round': {},
            'player_hands': {},
            'pot': 0,
            'current_player': state.current_player() if not state.is_chance_node() else None,
            'betting_history': '',
        }


def parse_betting_history(history_str):
    """解析下注历史"""
    # 简化解析，显示基本信息
    if not history_str:
        return "无下注历史"
    
    # 统计下注次数
    bet_count = history_str.count('r') + history_str.count('c') + history_str.count('f')
    return f"下注轮数: {bet_count}"


def display_game_state(state, human_player=0, model_player=1, action_history=None, 
                       board_cards_by_round=None, board_cards_ordered=None):
    """显示游戏状态"""
    info = get_game_state_info(state, None)
    
    print("\n" + "=" * 70)
    print(f"当前轮次: {info['round']}")
    print("=" * 70)
    
    # 显示公共牌（按轮次分组显示）
    if board_cards_by_round and board_cards_ordered:
        # 按轮次显示
        print(f"\n公共牌:")
        if 'Flop' in board_cards_by_round:
            print(f"  Flop: {' '.join(board_cards_by_round['Flop'])}")
        if 'Turn' in board_cards_by_round:
            print(f"  Turn: {' '.join(board_cards_by_round['Turn'])}")
        if 'River' in board_cards_by_round:
            print(f"  River: {' '.join(board_cards_by_round['River'])}")
        # 显示所有公共牌（按发牌顺序）
        print(f"  全部: {' '.join(board_cards_ordered)}")
    elif info['board_cards']:
        # 如果没有按轮次分组，直接显示
        print(f"\n公共牌: {' '.join(info['board_cards'])}")
    else:
        print("\n公共牌: (未发牌)")
    
    # 显示玩家手牌（只显示人类玩家的）
    if human_player in info['player_hands']:
        hand = info['player_hands'][human_player]
        print(f"\n你的手牌: {' '.join(hand)}")
    
    # 显示底池和玩家投入
    print(f"\n底池: {info['pot']}")
    
    # 显示玩家投入（如果有）
    try:
        state_struct = state.to_struct()
        player_contributions = getattr(state_struct, 'player_contributions', [])
        if player_contributions:
            print(f"\n玩家投入:")
            for p, contrib in enumerate(player_contributions):
                if p == human_player:
                    print(f"  你: {contrib}")
                else:
                    print(f"  玩家 {p}: {contrib}")
    except:
        pass
    
    # 显示动作历史（如果有）
    if action_history:
        print(f"\n本轮动作历史:")
        for player, action, action_str in action_history[-5:]:  # 只显示最近5个动作
            if player == human_player:
                print(f"  你: {action_str}")
            else:
                print(f"  玩家 {player}: {action_str}")
    
    # 显示当前玩家
    if info['current_player'] is not None:
        if info['current_player'] == human_player:
            print("\n>>> 轮到你行动 <<<")
        else:
            print(f"\n>>> 轮到玩家 {info['current_player']} 行动 <<<")


def action_to_string(action):
    """将动作编号转换为清晰的中文描述"""
    action_map = {
        0: "弃牌 (Fold)",
        1: "跟注/过牌 (Call/Check)",
        2: "底池加注 (Pot Raise)",
        3: "全押 (All-in)",
        4: "半池加注 (Half-pot)"  # 注意：在某些配置下可能不可用
    }
    return action_map.get(action, f"动作 {action}")


def get_human_action(state, player):
    """获取人类玩家的动作"""
    legal_actions = state.legal_actions(player)
    
    print(f"\n可选动作:")
    action_map = {}
    idx = 1
    
    for action in legal_actions:
        action_name = action_to_string(action)
        print(f"  {idx}. {action_name} (动作编号: {action})")
        action_map[idx] = action
        idx += 1
    
    while True:
        try:
            choice = input("\n请选择动作 (输入数字): ").strip()
            choice = int(choice)
            if choice in action_map:
                selected_action = action_map[choice]
                print(f"\n✓ 你选择了: {action_to_string(selected_action)}")
                return selected_action
            else:
                print(f"无效选择，请输入 1-{len(legal_actions)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n游戏被中断")
            sys.exit(0)


def play_interactive_game(game, model, device, human_player=0, model_player=1):
    """进行一局交互式游戏"""
    state = game.new_initial_state()
    action_history = []  # 记录动作历史
    last_round = None
    board_cards_by_round = {}  # 按轮次存储公共牌（保持发牌顺序）
    board_cards_ordered = []  # 按发牌顺序存储所有公共牌
    
    print("\n" + "=" * 70)
    print("开始新游戏")
    print("=" * 70)
    
    while not state.is_terminal():
        if state.is_chance_node():
            # 处理随机节点（发牌等）
            outcomes = state.chance_outcomes()
            if outcomes:
                action = np.random.choice([a for a, _ in outcomes], 
                                         p=[p for _, p in outcomes])
                prev_state = state
                state = state.child(action)
                
                # 检查是否发了新公共牌
                try:
                    prev_struct = prev_state.to_struct()
                    prev_board_cards_str = getattr(prev_struct, 'board_cards', '')
                    
                    state_struct = state.to_struct()
                    board_cards_str = getattr(state_struct, 'board_cards', '')
                    
                    if board_cards_str and board_cards_str != prev_board_cards_str:
                        # 解析当前公共牌
                        current_cards = []
                        for i in range(0, len(board_cards_str), 2):
                            if i + 2 <= len(board_cards_str):
                                current_cards.append(board_cards_str[i:i+2])
                        
                        # 解析之前的公共牌
                        prev_cards = []
                        if prev_board_cards_str:
                            for i in range(0, len(prev_board_cards_str), 2):
                                if i + 2 <= len(prev_board_cards_str):
                                    prev_cards.append(prev_board_cards_str[i:i+2])
                        
                        # 找出新发的牌（通过集合差集）
                        prev_set = set(prev_cards)
                        new_cards = [card for card in current_cards if card not in prev_set]
                        
                        # 如果找到了新牌，按轮次存储
                        if new_cards:
                            prev_count = len(prev_cards)
                            current_count = len(current_cards)
                            
                            # 根据之前和当前的公共牌数量判断轮次
                            if prev_count == 0:
                                # Flop 开始：可能是1张、2张或3张
                                if 'Flop' not in board_cards_by_round:
                                    board_cards_by_round['Flop'] = []
                                board_cards_by_round['Flop'].extend(new_cards)
                                board_cards_ordered.extend(new_cards)
                            elif prev_count < 3:
                                # Flop 继续发牌（第2或第3张）
                                if 'Flop' not in board_cards_by_round:
                                    board_cards_by_round['Flop'] = []
                                board_cards_by_round['Flop'].extend(new_cards)
                                board_cards_ordered.extend(new_cards)
                            elif prev_count == 3:
                                # Turn: 1张牌
                                if 'Turn' not in board_cards_by_round:
                                    board_cards_by_round['Turn'] = []
                                board_cards_by_round['Turn'].extend(new_cards)
                                board_cards_ordered.extend(new_cards)
                            elif prev_count == 4:
                                # River: 1张牌
                                if 'River' not in board_cards_by_round:
                                    board_cards_by_round['River'] = []
                                board_cards_by_round['River'].extend(new_cards)
                                board_cards_ordered.extend(new_cards)
                except Exception as e:
                    # 如果解析失败，尝试直接使用当前状态
                    try:
                        state_struct = state.to_struct()
                        board_cards_str = getattr(state_struct, 'board_cards', '')
                        if board_cards_str:
                            current_cards = []
                            for i in range(0, len(board_cards_str), 2):
                                if i + 2 <= len(board_cards_str):
                                    current_cards.append(board_cards_str[i:i+2])
                            
                            # 如果还没有记录，按数量判断轮次
                            if len(current_cards) == 3 and 'Flop' not in board_cards_by_round:
                                board_cards_by_round['Flop'] = current_cards
                                board_cards_ordered = current_cards.copy()
                            elif len(current_cards) == 4 and 'Turn' not in board_cards_by_round:
                                board_cards_by_round['Turn'] = current_cards[3:]
                                board_cards_ordered = current_cards.copy()
                            elif len(current_cards) == 5 and 'River' not in board_cards_by_round:
                                board_cards_by_round['River'] = current_cards[4:]
                                board_cards_ordered = current_cards.copy()
                    except:
                        pass
            else:
                break
        else:
            current_player = state.current_player()
            current_info = get_game_state_info(state, None)
            current_round = current_info['round']
            
            # 如果进入新轮次，显示提示
            if current_round != last_round and last_round is not None:
                print(f"\n{'='*70}")
                print(f"进入新轮次: {current_round}")
                print(f"{'='*70}")
                action_history = []  # 新轮次清空历史
            
            last_round = current_round
            
            # 显示游戏状态（传入按轮次分组的公共牌）
            display_game_state(state, human_player, model_player, action_history, board_cards_by_round, board_cards_ordered)
            
            if current_player == human_player:
                # 人类玩家行动
                action = get_human_action(state, current_player)
                action_str = action_to_string(action)
                action_history.append((current_player, action, action_str))
            else:
                # 模型行动
                action, probs = get_model_action(state, model, device, current_player)
                action_str = action_to_string(action)
                action_history.append((current_player, action, action_str))
                
                # 显示模型的选择
                print(f"\n玩家 {current_player} (模型) 选择了: {action_str}")
                
                # 显示动作概率（简化显示）
                if len(probs) <= 5:
                    prob_str = ", ".join([f"{action_to_string(a)}: {p:.2%}" 
                                         for a, p in sorted(probs.items(), 
                                                           key=lambda x: x[1], reverse=True)])
                    print(f"  动作概率: {prob_str}")
            
            state = state.child(action)
    
    # 游戏结束，显示结果
    returns = state.returns()
    info = get_game_state_info(state, None)
    
    print("\n" + "=" * 70)
    print("游戏结束")
    print("=" * 70)
    
    # 显示最终状态（按轮次显示公共牌）
    if board_cards_by_round and board_cards_ordered:
        print(f"\n最终公共牌:")
        if 'Flop' in board_cards_by_round:
            print(f"  Flop: {' '.join(board_cards_by_round['Flop'])}")
        if 'Turn' in board_cards_by_round:
            print(f"  Turn: {' '.join(board_cards_by_round['Turn'])}")
        if 'River' in board_cards_by_round:
            print(f"  River: {' '.join(board_cards_by_round['River'])}")
        print(f"  全部 (按发牌顺序): {' '.join(board_cards_ordered)}")
    elif info['board_cards']:
        print(f"\n最终公共牌: {' '.join(info['board_cards'])}")
    else:
        print(f"\n最终公共牌: (未发牌 - 游戏在 {info['round']} 结束)")
    
    # 显示所有玩家的手牌（游戏结束时）
    if info['player_hands']:
        print(f"\n所有玩家手牌:")
        for player, hand in info['player_hands'].items():
            if player == human_player:
                print(f"  你: {' '.join(hand)}")
            else:
                print(f"  玩家 {player}: {' '.join(hand)}")
    
    # 显示完整动作历史
    if action_history:
        print(f"\n完整动作历史:")
        for i, (player, action, action_str) in enumerate(action_history, 1):
            if player == human_player:
                print(f"  {i}. 你: {action_str}")
            else:
                print(f"  {i}. 玩家 {player}: {action_str}")
    
    # 显示结果
    print(f"\n最终结果:")
    print(f"  你的收益: {returns[human_player]:.2f}")
    
    # 显示所有玩家的收益
    for p in range(len(returns)):
        if p != human_player:
            print(f"  玩家 {p} 收益: {returns[p]:.2f}")
    
    # 判断胜负 (显示人类玩家结果)
    human_return = returns[human_player]
    if human_return > 0:
        print(f"\n🎉 你赢了！ (收益: +{human_return:.2f})")
    elif human_return < 0:
        print(f"\n😢 你输了 (收益: {human_return:.2f})")
    else:
        print(f"\n🤝 平局 (收益: 0.00)")
    
    # 如果游戏在Preflop就结束，说明其他玩家都弃牌了
    if info['round'] == "Preflop" and len(action_history) > 0:
        print(f"\n💡 提示: 游戏在 {info['round']} 就结束了，说明其他玩家都弃牌了")
    
    return returns


def main():
    parser = argparse.ArgumentParser(description="与训练好的模型进行交互式对局")
    parser.add_argument("--model_dir", type=str, 
                       default="models/deepcfr_texas_20251121_113543",
                       help="模型目录路径")
    parser.add_argument("--num_players", type=int, default=None,
                       help="玩家数量（如果不指定，从配置文件读取）")
    parser.add_argument("--human_player", type=int, default=0,
                       help="人类玩家编号（0 或 1）")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="使用 GPU")
    
    args = parser.parse_args()
    
    # 检查模型目录
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        print("\n可用的模型目录:")
        import glob
        model_dirs = glob.glob("models/deepcfr_texas_*/")
        for d in sorted(model_dirs):
            print(f"  - {d}")
        sys.exit(1)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载模型（num_players 会在 load_model 中从配置文件读取）
    game, model = load_model(args.model_dir, args.num_players, device)
    if game is None or model is None:
        print("模型加载失败")
        sys.exit(1)
    
    model_player = 1 - args.human_player
    
    print("\n" + "=" * 70)
    print("交互式对局")
    print("=" * 70)
    print(f"人类玩家: {args.human_player}")
    print(f"模型玩家: {model_player}")
    print("\n提示: 输入 Ctrl+C 可以退出游戏")
    
    # 游戏循环
    while True:
        try:
            returns = play_interactive_game(game, model, device, 
                                          args.human_player, model_player)
            
            # 询问是否继续
            print("\n" + "-" * 70)
            choice = input("是否继续下一局? (y/n): ").strip().lower()
            if choice != 'y':
                break
        except KeyboardInterrupt:
            print("\n\n游戏被中断，退出")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 70)
    print("感谢游戏！")
    print("=" * 70)


if __name__ == "__main__":
    main()

