
import gradio as gr
import pyspiel
import torch
import numpy as np
import json
import os
import re
import sys
import glob
from collections import Counter

# 添加当前目录到 path 以导入本地模块
sys.path.append(os.getcwd())

# 尝试导入模型类
from deep_cfr_simple_feature import DeepCFRSimpleFeature, SimpleFeatureMLP
try:
    from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
except ImportError:
    pass
from open_spiel.python.pytorch.deep_cfr import MLP

# ==========================================
# 0. 牌型评估工具 (简化版)
# ==========================================

CARD_RANKS = "23456789TJQKA"
RANK_VALUES = {r: i for i, r in enumerate(CARD_RANKS)}

def evaluate_hand(hole_cards, board_cards):
    """
    评估 7 张牌的最大牌型
    返回: (rank_value, rank_name, best_5_cards_str)
    """
    if not hole_cards:
        return 0, "未知", ""
        
    # 合并牌
    all_cards = hole_cards + board_cards
    if len(all_cards) < 5:
        return 0, "牌数不足", ""
        
    # 解析牌
    # card format: "Ah", "Tc"
    parsed_cards = []
    for c in all_cards:
        if len(c) < 2: continue
        r = c[0]
        s = c[1]
        parsed_cards.append((RANK_VALUES.get(r, -1), s, c))
        
    parsed_cards.sort(key=lambda x: x[0], reverse=True)
    
    # 辅助函数：检查同花
    def check_flush(cards):
        suits = [c[1] for c in cards]
        counts = Counter(suits)
        flush_suit = None
        for s, count in counts.items():
            if count >= 5:
                flush_suit = s
                break
        if flush_suit:
            flush_cards = [c for c in cards if c[1] == flush_suit]
            return flush_cards[:5]
        return None

    # 辅助函数：检查顺子
    def check_straight(cards):
        # 去重点数
        unique_ranks = sorted(list(set([c[0] for c in cards])), reverse=True)
        # 处理 A 2 3 4 5 的情况 (A=12, 2=0)
        if 12 in unique_ranks:
            unique_ranks.append(-1) # Add A as low
            
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                # 找到顺子，重构由哪些牌组成
                straight_ranks = unique_ranks[i:i+5]
                # 针对 A 2 3 4 5 特殊处理
                if straight_ranks[-1] == -1:
                    straight_ranks = [12 if r==-1 else r for r in straight_ranks]
                    
                best_straight = []
                for r in straight_ranks:
                    for c in cards:
                        if c[0] == r:
                            best_straight.append(c)
                            break
                return best_straight
        return None

    # 1. 同花顺 (Straight Flush)
    flush_cards = check_flush(parsed_cards)
    if flush_cards:
        straight_flush = check_straight(flush_cards)
        if straight_flush:
            return 9000 + straight_flush[0][0], "同花顺", "".join([c[2] for c in straight_flush])

    # 2. 四条 (Four of a Kind)
    ranks = [c[0] for c in parsed_cards]
    counts = Counter(ranks)
    fours = [r for r, c in counts.items() if c == 4]
    if fours:
        quad_rank = fours[0]
        kicker = [r for r in ranks if r != quad_rank][0]
        return 8000 + quad_rank, "四条", "" # 略去具体牌组合显示

    # 3. 葫芦 (Full House)
    threes = [r for r, c in counts.items() if c >= 3]
    twos = [r for r, c in counts.items() if c >= 2]
    if threes:
        best_three = max(threes)
        # 找一对（排除掉组成三条的那个）
        remaining_pairs = [r for r in twos if r != best_three]
        if remaining_pairs:
            best_pair = max(remaining_pairs)
            return 7000 + best_three, "葫芦", ""

    # 4. 同花 (Flush)
    if flush_cards:
        return 6000 + flush_cards[0][0], "同花", "".join([c[2] for c in flush_cards])

    # 5. 顺子 (Straight)
    straight_cards = check_straight(parsed_cards)
    if straight_cards:
        return 5000 + straight_cards[0][0], "顺子", "".join([c[2] for c in straight_cards])

    # 6. 三条 (Three of a Kind)
    if threes:
        return 4000 + max(threes), "三条", ""

    # 7. 两对 (Two Pair)
    if len(twos) >= 2:
        twos.sort(reverse=True)
        return 3000 + twos[0], "两对", ""

    # 8. 一对 (One Pair)
    if twos:
        return 2000 + max(twos), "一对", ""

    # 9. 高牌 (High Card)
    return 1000 + parsed_cards[0][0], "高牌", ""


def strip_ansi(text):
    """去除 ANSI 颜色代码"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# ==========================================
# 1. 配置与模型加载
# ==========================================

MODEL_DIR = "models/deepcfr_stable_run/checkpoints/iter_12500"
DEVICE = "cpu"

def load_model(model_dir, num_players=None, device='cpu'):
    """加载训练好的模型（支持 checkpoint 格式）- 移植自 play_interactive.py"""
    print(f"加载模型: {model_dir}")
    
    # 读取配置文件
    config_path = os.path.join(model_dir, 'config.json')
    config = {}
    
    if not os.path.exists(config_path):
        if "checkpoints" in model_dir:
            parent_dir = os.path.dirname(model_dir)
            if "checkpoints" in parent_dir:
                main_dir = os.path.dirname(parent_dir)
            else:
                main_dir = parent_dir
            config_path = os.path.join(main_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if num_players is None:
            num_players = config.get('num_players', 6)
        
        use_simple_feature = config.get('use_simple_feature', False)
        use_feature_transform = config.get('use_feature_transform', False)
        policy_layers = tuple(config.get('policy_layers', [64, 64]))
        save_prefix = config.get('save_prefix', 'deepcfr_texas')
        betting_abstraction = config.get('betting_abstraction', 'fcpa')
        game_string = config.get('game_string', None)
    else:
        # 默认值
        if num_players is None:
            num_players = 6
        use_simple_feature = False
        use_feature_transform = False
        policy_layers = (64, 64)
        save_prefix = 'deepcfr_texas'
        betting_abstraction = 'fcpa'
        game_string = None
    
    # 创建游戏
    game = None
    if game_string:
        try:
            game = pyspiel.load_game(game_string)
        except Exception as e:
            print(f"使用 game_string 创建游戏失败: {e}")
            game = None
            
    if game is None:
        # Fallback config
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
            'bettingAbstraction': betting_abstraction,
        }
        if num_players == 6:
            game_config['blind'] = "50 100 0 0 0 0"
            game_config['firstPlayer'] = "3 1 1 1"
        elif num_players == 2:
            game_config['blind'] = "100 50"
            game_config['firstPlayer'] = "2 1 1 1"
            
        game = pyspiel.load_game('universal_poker', game_config)

    # 查找模型文件
    policy_filename = f"{save_prefix}_policy_network.pt"
    policy_path = os.path.join(model_dir, policy_filename)
    
    if not os.path.exists(policy_path):
        pt_files = glob.glob(os.path.join(model_dir, "*_policy_network*.pt"))
        if pt_files:
            checkpoint_files = [f for f in pt_files if "_iter" in os.path.basename(f)]
            if checkpoint_files:
                # 找最新的
                max_iter = 0
                latest_file = None
                for f in checkpoint_files:
                    match = re.search(r'_iter(\d+)\.pt$', f)
                    if match:
                        iter_num = int(match.group(1))
                        if iter_num > max_iter:
                            max_iter = iter_num
                            latest_file = f
                if latest_file:
                    policy_path = latest_file
            else:
                policy_path = pt_files[0]

    if not policy_path or not os.path.exists(policy_path):
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    print(f"Loading weights from {policy_path}")
    
    # 加载模型
    if use_simple_feature:
        print("Using Simple Feature Model")
        solver = DeepCFRSimpleFeature(
            game,
            policy_network_layers=policy_layers,
            advantage_network_layers=(32, 32),
            num_iterations=1,
            num_traversals=1,
            learning_rate=1e-4,
            device=device
        )
        # 处理 DataParallel
        state_dict = torch.load(policy_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        solver._policy_network.load_state_dict(new_state_dict)
        solver._policy_network.eval()
        return game, solver, config
        
    elif use_feature_transform:
        print("Using Feature Transform Model")
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
            state_dict = torch.load(policy_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            solver._policy_network.load_state_dict(new_state_dict)
            solver._policy_network.eval()
            return game, solver, config
        except ImportError:
            print("Import Error for DeepCFRWithFeatureTransform")
            pass

    # Standard MLP
    print("Using Standard MLP")
    state = game.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    network = MLP(embedding_size, list(policy_layers), num_actions)
    network = network.to(device)
    network.load_state_dict(torch.load(policy_path, map_location=device))
    network.eval()
    return game, network, config


# 全局加载
try:
    GAME, MODEL, CONFIG = load_model(MODEL_DIR, device=DEVICE)
    print("Global model loaded.")
except Exception as e:
    print(f"Error loading global model: {e}")
    GAME, MODEL, CONFIG = None, None, None


# ==========================================
# 2. 游戏逻辑
# ==========================================

def get_ai_action(state, model):
    """获取 AI 动作"""
    player = state.current_player()
    legal_actions = state.legal_actions()
    
    if not legal_actions:
        return None
    
    # Check if model is a solver with action_probabilities
    if hasattr(model, 'action_probabilities'):
        probs_dict = model.action_probabilities(state, player)
        actions = list(probs_dict.keys())
        probs = list(probs_dict.values())
        if sum(probs) > 0:
            probs = np.array(probs) / sum(probs)
            # Sample or greedy? Let's do weighted sample for variety, or greedy for strength
            # Using argmax for "best" move
            # best_idx = np.argmax(probs)
            # action = actions[best_idx]
            
            # Using random sample based on probs
            action = np.random.choice(actions, p=probs)
        else:
            action = np.random.choice(actions)
        return action
    
    # Standard Network
    info_state = torch.FloatTensor(state.information_state_tensor(player)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(info_state)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    legal_probs = np.zeros_like(probs)
    legal_probs[legal_actions] = probs[legal_actions]
    
    if legal_probs.sum() > 0:
        legal_probs /= legal_probs.sum()
        action = np.random.choice(len(legal_probs), p=legal_probs)
    else:
        action = np.random.choice(legal_actions)
        
    return action

def get_cards_from_state(state_str, player_idx=None):
    """从状态字符串中提取公共牌和手牌"""
    # 去除 ANSI 颜色代码，防止正则匹配失败
    state_str = strip_ansi(state_str)
    
    board_cards = []
    # 改进正则：只匹配连续的牌字符，遇到非牌字符（如 'Deck'）停止
    # 匹配模式：BoardCards 后面紧跟一系列 (RankSuit) 组合
    board_match = re.search(r'BoardCards:?\s*((?:[2-9TJQKA][shdc]\s*)*)', state_str)
    if board_match:
        # 提取所有匹配卡牌格式的子串
        raw_board = board_match.group(1)
        board_cards = re.findall(r'[2-9TJQKA][shdc]', raw_board)
    
    player_hand = []
    if player_idx is not None:
        p_cards_match = re.search(f'P{player_idx} Cards:?\s*([2-9TJQKA][shdc].*)', state_str)
        if p_cards_match:
            raw_hand = p_cards_match.group(1)
            player_hand = re.findall(r'[2-9TJQKA][shdc]', raw_hand)
            
    return board_cards, player_hand

def format_card_log(card_str):
    """格式化日志中的卡牌显示，例如将 Th 转换为 10h"""
    if not card_str or len(card_str) < 2:
        return card_str
    
    rank, suit = card_str[0], card_str[1:]
    if rank == 'T':
        rank = '10'
    return f"{rank}{suit}"

def run_game_step(history, user_action=None, user_seat=0):
    """
    运行游戏直到需要用户输入或游戏结束
    Args:
        history: 动作 ID 列表
        user_action: 用户刚刚选择的动作 ID
    
    Returns:
        new_history, state, logs, is_user_turn, folded_players
    """
    state = GAME.new_initial_state()
    # 本次调用的新日志
    logs = []
    folded_players = set()
    
    # 1. 重演历史
    try:
        # 重演历史不记录日志
        # 但我们需要跟踪已发出的公共牌数量，以便在"重演结束后"知道当前处于什么阶段
        # 不过，更简单的做法是：重演完后，直接看当前状态的公共牌数量
        
        for action in history:
            # 记录弃牌者
            if not state.is_chance_node():
                action_str = state.action_to_string(state.current_player(), action)
                if "Fold" in action_str:
                    folded_players.add(state.current_player())
            
            state.apply_action(action)
            
    except Exception as e:
        logs.append(f"Error replaying: {e}")
        return history, state, logs, False, folded_players

    # 获取重演后的当前公共牌数量
    curr_board, _ = get_cards_from_state(str(state))
    prev_board_count = len(curr_board)

    # 2. 应用用户动作
    if user_action is not None:
        if state.current_player() == user_seat:
            act_str = state.action_to_string(user_seat, user_action)
            logs.append(f"👤 您: {act_str}")
            
            if "Fold" in act_str:
                folded_players.add(user_seat)
                
            state.apply_action(user_action)
            history.append(user_action)
        else:
            logs.append("⚠️ 错误: 不是您的回合")

    # 3. 自动运行直到用户回合或结束
    # 我们使用暂存列表来收集同一个阶段发出的牌（例如 Flop 的 3 张）
    pending_deal_cards = []
    
    while not state.is_terminal():
        current_player = state.current_player()
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            
            # 检查这个 chance action 是否是发公共牌
            # OpenSpiel universal_poker 的 chance action string 通常是 "Deal 2h" 格式
            action_str = state.action_to_string(current_player, action)
            if "Deal" in action_str:
                # 提取牌
                # 格式通常是 "Deal 2h"
                card_match = re.search(r'Deal\s+([2-9TJQKA][shdc])', action_str)
                if card_match:
                    card = card_match.group(1)
                    pending_deal_cards.append(card)
            
            state.apply_action(action)
            history.append(action)
            
            # 检查发牌是否结束（进入下一阶段或下一个动作不是 Deal）
            # 最简单的方法是检查 pending_deal_cards 的数量是否达到了阶段要求
            # 或者看 state 是否不再是 chance node (这可能不准确，因为可能连续 deal)
            # 我们在下一次循环开头检查
            
        elif current_player == user_seat:
             # 在返回给用户前，把积攒的 pending cards 输出日志
            if pending_deal_cards:
                formatted_cards = [format_card_log(c) for c in pending_deal_cards]
                cards_str = " ".join(formatted_cards)
                
                # 确定阶段
                # 此时 state 已经是发完牌后的状态
                # 总牌数
                total_board_count = prev_board_count + len(pending_deal_cards)
                # 这只是近似，因为 pending 可能跨越多个阶段？通常不会。
                # Flop=3, Turn=4, River=5
                stage = "Flop"
                if total_board_count == 4: stage = "Turn"
                elif total_board_count == 5: stage = "River"
                
                logs.append(f"🎴 发牌 ({stage}): {cards_str}")
                
                # 更新 prev_board_count 并清空 pending
                prev_board_count = total_board_count
                pending_deal_cards = []

            return history, state, logs, True, folded_players
            
        else:
            # AI 回合
            # 先处理积攒的发牌日志
            if pending_deal_cards:
                formatted_cards = [format_card_log(c) for c in pending_deal_cards]
                cards_str = " ".join(formatted_cards)
                
                total_board_count = prev_board_count + len(pending_deal_cards)
                stage = "Flop"
                if total_board_count == 4: stage = "Turn"
                elif total_board_count == 5: stage = "River"
                
                logs.append(f"🎴 发牌 ({stage}): {cards_str}")
                prev_board_count = total_board_count
                pending_deal_cards = []
            
            action = get_ai_action(state, MODEL)
            act_str = state.action_to_string(current_player, action)
            logs.append(f"🤖 AI {current_player}: {act_str}")
            
            if "Fold" in act_str:
                folded_players.add(current_player)
                
            state.apply_action(action)
            history.append(action)

    # 游戏结束
    # 处理剩余的 pending cards
    if pending_deal_cards:
        formatted_cards = [format_card_log(c) for c in pending_deal_cards]
        cards_str = " ".join(formatted_cards)
        
        total_board_count = prev_board_count + len(pending_deal_cards)
        stage = "Flop"
        if total_board_count == 4: stage = "Turn"
        elif total_board_count == 5: stage = "River"
        
        logs.append(f"🎴 发牌 ({stage}): {cards_str}")

    return history, state, logs, False, folded_players

# ==========================================
# 3. 界面渲染
# ==========================================

def get_ordered_board_cards(state):
    """
    通过重演历史获取按发牌顺序排列的公共牌。
    使用集合差异法确定新牌，确保准确性。
    """
    ordered_cards = []
    known_cards = set()
    
    try:
        temp_state = GAME.new_initial_state()
        history = state.history()
        
        for action in history:
            if temp_state.is_chance_node():
                # 获取动作前的板牌
                # 实际上不需要动作前的，只需要动作后的，然后对比 known_cards
                
                temp_state.apply_action(action)
                
                # 获取当前的板牌
                curr_board_str = strip_ansi(str(temp_state))
                
                # 改进正则：只匹配连续的牌，避免匹配到 Deck
                curr_board_match = re.search(r'BoardCards:?\s*((?:[2-9TJQKA][shdc]\s*)*)', curr_board_str)
                
                board_text = ""
                if curr_board_match:
                    board_text = curr_board_match.group(1)
                
                current_all_cards = re.findall(r'[2-9TJQKA][shdc]', board_text)
                    
                # 找出新出现的牌
                new_cards = []
                for card in current_all_cards:
                    if card not in known_cards:
                        new_cards.append(card)
                        known_cards.add(card)
                    
                # 如果有新牌，按顺序加入 ordered_cards
                if new_cards:
                    ordered_cards.extend(new_cards)
            else:
                temp_state.apply_action(action)
                
    except Exception as e:
        print(f"Error getting ordered board: {e}")
        # Fallback
        ordered_cards, _ = get_cards_from_state(str(state))
        
    return ordered_cards

def format_card_html(card_str):
    """将牌字符串（如 'Ah', 'Tc'）转换为 HTML"""
    if len(card_str) < 2:
        return card_str
    
    rank, suit_char = card_str[0], card_str[1]
    
    # 将 T 替换为 10
    display_rank = rank
    if rank == 'T':
        display_rank = '10'
        
    suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    suit = suit_map.get(suit_char, suit_char)
    color = "red" if suit_char in ['h', 'd'] else "black"
    
    return f"<span style='color:{color}; font-size: 1.5em; background: white; padding: 2px 5px; border: 1px solid #ccc; border-radius: 4px; margin: 2px;'>{display_rank}{suit}</span>"

def get_player_positions(state, num_players):
    """推断玩家位置 (BTN, SB, BB, etc.)"""
    # 简单的基于 Round 0 的盲注动作推断
    # 找到第一个 SmallBlind 动作的玩家作为 SB
    # 前一个就是 BTN
    positions = [""] * num_players
    
    try:
        # 去除 ANSI
        state_str = strip_ansi(str(state))
        
        # 尝试从 state_str 的 action log 中找 (如果有的话)
        # 否则，默认配置: 
        # 6人局: P0=SB? 不一定。
        # 让我们尝试找一下
        
        # 如果找不到，就按默认顺序给个大概位置，或者不显示
        # 假设 Dealer 是随机的。
        pass
    except:
        pass
        
    # 定义标准位置名称 (6-max)
    pos_names_6 = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
    # pos_names_2 = ["SB", "BB"] # Heads up: SB is BTN
    
    # 我们尝试解析 Dealer (d=...)
    # universal_poker 的 state string 包含 "Dealer: 0" 吗？
    # 如果有：
    dealer_match = re.search(r'Dealer:?\s*(\d+)', state_str)
    if dealer_match:
        dealer_idx = int(dealer_match.group(1))
        # Assign positions relative to dealer
        # In 6-max: Dealer=BTN. Next is SB.
        for i in range(num_players):
            offset = (i - dealer_idx) % num_players
            # offset 0 = BTN
            # offset 1 = SB
            # offset 2 = BB
            # ...
            if num_players == 6:
                name_map = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}
                positions[i] = name_map.get(offset, "")
            elif num_players == 2:
                # HU: Dealer is SB, Other is BB
                name_map = {0: "SB/BTN", 1: "BB"}
                positions[i] = name_map.get(offset, "")
            else:
                if offset == 0: positions[i] = "BTN"
                elif offset == 1: positions[i] = "SB"
                elif offset == 2: positions[i] = "BB"
                
    return positions

def format_state_html(state, user_seat=0, logs=[], folded_players=set()):
    if state is None:
        return "<h3>点击 '开始新游戏'</h3>", ""
    
    # 使用正则表达式解析状态字符串
    state_str = strip_ansi(str(state))
    print(f"DEBUG State String:\n{state_str}\n-------------------")
    info_str = ""
    try:
        info_str = state.information_state_string(user_seat) 
    except:
        pass
        
    full_info = state_str + "\n" + info_str

    # 1. 解析底池 (Pot)
    pot = 0
    pot_match = re.search(r'Pot: (\d+)', full_info)
    if pot_match:
        pot = pot_match.group(1)
        
    # 2. 解析公共牌
    board_html = ""
    # 使用有序获取
    board_cards_list = get_ordered_board_cards(state)
    # 如果有序获取失败或者为空但 state string 有牌（比如 pre-existing state），fallback
    if not board_cards_list:
        fallback_list, _ = get_cards_from_state(state_str)
        if fallback_list:
            board_cards_list = fallback_list
            
    board_list = board_cards_list # 用于评估牌型
    if board_list:
        for c in board_list:
            try:
                board_html += format_card_html(c)
            except Exception as e:
                print(f"Error formatting card {c}: {e}")
                board_html += f"[{c}?]"
    else:
        board_html = "<span style='color: gray'>(Pre-flop)</span>"

    html = f"""
    <div style='font-family: Arial; padding: 20px; background-color: #f0f2f5; border-radius: 10px;'>
        <div style='background: #e6f7ff; padding: 10px; border-radius: 8px; margin-bottom: 20px; text-align: center; border: 1px solid #91d5ff;'>
            <h2 style='margin:0; color: #0050b3;'>💰 底池: {pot}</h2>
        </div>
        
        <div style='text-align: center; margin-bottom: 30px;'>
            <div style='font-weight: bold; margin-bottom: 10px;'>公共牌</div>
            <div style='min-height: 50px;'>{board_html}</div>
        </div>
        
        <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 15px;'>
    """
    
    num_players = CONFIG["num_players"] if CONFIG else 6
    current_player = state.current_player() if not state.is_terminal() else -1
    
    # 解析筹码
    stacks = []
    money_match = re.search(r'Money:\s*([\d\s]+)', full_info)
    if money_match:
        stacks = money_match.group(1).strip().split()
    
    final_hands = [] # 用于结算显示 (p, hand_str, hand_rank_name)
    
    # 解析位置
    positions = get_player_positions(state, num_players)

    for p in range(num_players):
        is_user = (p == user_seat)
        is_active = (p == current_player)
        is_folded = (p in folded_players)
        
        pos_name = positions[p]
        if pos_name:
            pos_label = f"<span style='background:#ccc; color:white; border-radius:3px; padding:0 2px; font-size:0.7em; margin-left:5px;'>{pos_name}</span>"
        else:
            pos_label = ""
        
        bg_color = "#fff7e6" if is_active else "#ffffff"
        border_color = "#faad14" if is_active else "#d9d9d9"
        border_width = "3px" if is_active else "1px"
        
        opacity = "1.0"
        if is_folded:
            bg_color = "#f5f5f5"
            border_color = "#d9d9d9"
            opacity = "0.6"
        
        if is_user:
            bg_color = "#f6ffed" if not is_active else "#d9f7be"
            border_color = "#52c41a"
            border_width = "3px" if is_active else "2px"
            if is_folded:
                bg_color = "#e6f7ff" # 玩家弃牌后颜色
            
        name = f"👤 您{pos_label}" if is_user else f"🤖 AI {p}{pos_label}"
        if is_folded:
            name += " (弃牌)"
        
        # 3. 解析玩家手牌
        hand_html = ""
        _, p_cards_list = get_cards_from_state(state_str, p)
        
        # 保存手牌用于结算 (无论是否弃牌，只要有牌都显示)
        if p_cards_list:
            # 评估牌型
            hole_cards = p_cards_list
            _, rank_name, _ = evaluate_hand(hole_cards, board_list)
            # 保存 p_cards_list 供后面使用，转为字符串仅用于显示
            p_cards_str = "".join(p_cards_list)
            
            display_rank = rank_name
            if is_folded:
                display_rank = f"{rank_name} (弃牌)"
                
            final_hands.append((p, p_cards_str, display_rank))
        else:
            final_hands.append((p, "", "弃牌"))
        
        show_cards = False
        if is_user:
            show_cards = True
        elif state.is_terminal():
             # 游戏结束时，所有玩家都显示牌（包括弃牌的，为了复盘）
            show_cards = True
        else:
            show_cards = False 
            
        if p_cards_list and show_cards:
            cards = p_cards_list
            for c in cards:
                hand_html += format_card_html(c)
        elif not show_cards and p_cards_list: # 有牌但不显示
             if is_folded:
                 hand_html = "<span style='color:gray; font-size: 0.8em;'>(已弃牌)</span>"
             else:
                 hand_html = "<span style='font-size: 1.5em;'>🂠 🂠</span>"
        else:
             hand_html = "<span style='color:gray; font-size: 0.8em;'>(等待发牌)</span>"

        stack_val = stacks[p] if p < len(stacks) else "?"
        
        html += f"""
        <div style='background: {bg_color}; border: {border_width} solid {border_color}; padding: 10px; border-radius: 8px; width: 140px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); opacity: {opacity};'>
            <div style='font-weight: bold; margin-bottom: 5px; font-size: 0.9em;'>{name}</div>
            <div style='margin: 5px 0; min-height: 35px; display: flex; justify-content: center; align-items: center;'>{hand_html}</div>
            <div style='font-size: 0.8em; color: #595959;'>Stack: {stack_val}</div>
        </div>
        """

    html += "</div>" # Flex container end

    if state.is_terminal():
        returns = state.returns()
        user_ret = returns[user_seat]
        result_color = "#f6ffed" if user_ret > 0 else "#fff1f0"
        result_border = "#b7eb8f" if user_ret > 0 else "#ffa39e"
        msg = "🎉 胜利!" if user_ret > 0 else ("😢 失败" if user_ret < 0 else "🤝 平局")
        
        # 结算详情表
        result_table = "<table style='width:100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;'>"
        result_table += "<tr style='background:#fafafa; border-bottom: 1px solid #eee;'><th>玩家</th><th>手牌</th><th>牌型</th><th>收益</th></tr>"
        
        for p in range(num_players):
            p_name = "Player 0 (你)" if p == user_seat else f"Player {p}"
            p_hand_str = final_hands[p][1] if p < len(final_hands) else ""
            p_rank_name = final_hands[p][2] if p < len(final_hands) else ""
            
            # 格式化手牌
            p_hand_html = ""
            if p_hand_str:
                cards = [p_hand_str[i:i+2] for i in range(0, len(p_hand_str), 2)]
                for c in cards:
                    p_hand_html += format_card_html(c)
            else:
                p_hand_html = "<span style='color:gray'>-</span>"
                
            p_ret = returns[p]
            ret_color = "green" if p_ret > 0 else "red"
            
            result_table += f"<tr><td style='padding:5px;'>{p_name}</td><td style='padding:5px;'>{p_hand_html}</td><td style='padding:5px;'>{p_rank_name}</td><td style='padding:5px; color:{ret_color}; font-weight:bold;'>{p_ret}</td></tr>"
            
        result_table += "</table>"
        
        html += f"""
        <div style='margin-top: 30px; padding: 15px; background: {result_color}; border: 1px solid {result_border}; border-radius: 8px; text-align: center;'>
            <h3 style='margin:0 0 10px 0;'>{msg}</h3>
            <div>您的收益: <span style='font-weight:bold; font-size: 1.2em;'>{user_ret}</span></div>
            <div style='margin-top: 15px; text-align: left;'>
                <div style='font-weight: bold; margin-bottom: 5px;'>📊 结算详情:</div>
                {result_table}
            </div>
        </div>
        """
        
    html += "</div>" # Main container end
    return html, "\n".join(logs)

# ==========================================
# 4. Gradio Callbacks
# ==========================================

def start_new_game():
    if GAME is None:
        return [], None, "<h1>❌ 模型加载失败</h1>", "Check console logs", gr.update(choices=[], value=None, interactive=False), gr.update(interactive=False)
        
    history = []
    # Reset game
    new_history, state, logs, is_user_turn, folded_players = run_game_step(history, user_action=None, user_seat=0)
    
    # 初始日志
    logs.insert(0, "🏁 新游戏开始")
    log_text = "\n".join(logs)
    
    html, _ = format_state_html(state, user_seat=0, logs=logs, folded_players=folded_players)
    
    choices_display = []
    if is_user_turn:
        legal_actions = state.legal_actions()
        choices_display = [f"{state.action_to_string(0, a)} (ID: {a})" for a in legal_actions]
        
    return (
        new_history, 
        html,
        log_text,
        gr.update(choices=choices_display, value=None, interactive=is_user_turn),
        gr.update(interactive=is_user_turn)
    )

def on_submit_action(history, action_str, current_logs):
    if not action_str:
        return history, None, current_logs, gr.update(), gr.update()
        
    # Extract ID
    try:
        action_id = int(re.search(r'ID: (\d+)', action_str).group(1))
    except:
        return history, None, current_logs + "\n❌ 动作解析错误", gr.update(), gr.update()
        
    new_history, state, new_logs, is_user_turn, folded_players = run_game_step(history, user_action=action_id, user_seat=0)
    
    if current_logs:
        full_log_text = current_logs + "\n" + "\n".join(new_logs)
    else:
        full_log_text = "\n".join(new_logs)
        
    html, _ = format_state_html(state, user_seat=0, folded_players=folded_players)
    
    choices_display = []
    if is_user_turn:
        legal_actions = state.legal_actions()
        choices_display = [f"{state.action_to_string(0, a)} (ID: {a})" for a in legal_actions]
        
    return (
        new_history,
        html,
        full_log_text,
        gr.update(choices=choices_display, value=None, interactive=is_user_turn),
        gr.update(interactive=is_user_turn)
    )

# 构建界面
with gr.Blocks(title="Texas Hold'em vs AI") as demo:
    gr.Markdown("# 🃏 德州扑克人机对战 (6人局)")
    
    history_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=2):
            board_display = gr.HTML(label="游戏桌面", value="<h3>请点击'开始新游戏'</h3>")
            game_log = gr.Textbox(label="游戏日志", lines=15, max_lines=20)
            
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 操作区")
            action_radio = gr.Radio(label="选择动作", choices=[], interactive=False)
            submit_btn = gr.Button("✅ 确认动作", variant="primary", interactive=False)
            new_game_btn = gr.Button("🔄 开始新游戏", variant="secondary")
            
            gr.Markdown("""
            ### ℹ️ 说明
            - 您是 **Player 0**
            - 5 个 AI 对手 (DeepCFR)
            """)

    new_game_btn.click(
        fn=start_new_game,
        inputs=[],
        outputs=[history_state, board_display, game_log, action_radio, submit_btn]
    )
    
    submit_btn.click(
        fn=on_submit_action,
        inputs=[history_state, action_radio, game_log],
        outputs=[history_state, board_display, game_log, action_radio, submit_btn]
    )

if __name__ == "__main__":
    print(f"Starting Gradio...")
    try:
        demo.launch(server_name="0.0.0.0", server_port=8827)
    except OSError:
        print("Port 8827 in use, trying random port...")
        demo.launch(server_name="0.0.0.0")
