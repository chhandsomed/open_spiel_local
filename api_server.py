#!/usr/bin/env python3
"""API服务器 - 为后端提供推荐动作接口

基于 play_gradio.py 改造，支持任意位置的推理。
后端传：当前玩家手牌 + 公共牌 + 历史动作 + 盲注 + 筹码
其他玩家手牌由系统随机分配（不影响推理结果）
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import json
import argparse
import random
import numpy as np
import torch
import pyspiel
import re
import glob
import sys
from flask import Flask, request, jsonify
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

# 添加当前目录到 path 以导入本地模块
sys.path.append(os.getcwd())

# 尝试导入自定义特征类（基于 play_gradio.py）
try:
    from deep_cfr_simple_feature import DeepCFRSimpleFeature, SimpleFeatureMLP
    try:
        from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
    except ImportError:
        pass
    HAVE_CUSTOM_FEATURES = True
except ImportError:
    HAVE_CUSTOM_FEATURES = False
    from open_spiel.python.pytorch.deep_cfr import MLP

app = Flask(__name__)

# 全局变量：模型和游戏
GAME = None
MODEL = None
CONFIG = None
DEVICE = 'cpu'
MODEL_DIR = None

# 游戏实例缓存：根据配置缓存游戏实例，避免重复创建
GAME_CACHE = {}  # key: (tuple(blinds), tuple(stacks), dealer_pos, betting_abstraction, num_players)


# ==========================================
# 1. 牌面转换工具
# ==========================================

def convert_user_card_to_openspiel(card_input) -> int:
    """将用户输入的牌面格式转换为OpenSpiel的card index
    
    用户输入格式：
    - 数字格式（0-51）：数字已经包含花色信息
      * 花色顺序：方块(Diamond)[0-12] -> 梅花(Clubs)[13-25] -> 红桃(Hearts)[26-38] -> 黑桃(Spade)[39-51]
      * 每个花色内：2~JQKA 对应 0~12（rank）
    - 字符串格式：如 "As", "Kh", "2d", "Tc", "Xh"（传统格式，兼容）
    - 大小王：JL(小王), JB(大王) - 不支持
    
    OpenSpiel格式（suit * 13 + rank）：
    - Diamonds(0-12): suit=0
    - Spades(13-25): suit=1
    - Hearts(26-38): suit=2
    - Clubs(39-51): suit=3
    
    Args:
        card_input: 用户输入的牌面，可以是：
                   - int: 0-51的数字
                   - str: 传统格式字符串如 "As", "Kh"
    
    Returns:
        OpenSpiel的card index (0-51)
    """
    # 如果是整数，直接转换
    if isinstance(card_input, int):
        user_index = card_input
        
        if user_index < 0 or user_index > 51:
            raise ValueError(f"Invalid card index: {user_index}, must be 0-51")
        
        # 用户输入的花色顺序：方块[0-12] -> 梅花[13-25] -> 红桃[26-38] -> 黑桃[39-51]
        # OpenSpiel顺序：方块[0-12] -> 黑桃[13-25] -> 红桃[26-38] -> 梅花[39-51]
        
        if 0 <= user_index <= 12:
            # 方块：不变
            return user_index  # 0-12
        elif 13 <= user_index <= 25:
            # 用户：梅花[13-25] -> OpenSpiel：梅花[39-51]
            rank = user_index - 13
            return 39 + rank  # 39-51
        elif 26 <= user_index <= 38:
            # 红桃：不变
            return user_index  # 26-38
        elif 39 <= user_index <= 51:
            # 用户：黑桃[39-51] -> OpenSpiel：黑桃[13-25]
            rank = user_index - 39
            return 13 + rank  # 13-25
        else:
            raise ValueError(f"Invalid card index: {user_index}")
    
    # 如果是字符串，处理传统格式或大小王
    elif isinstance(card_input, str):
        card_str = card_input
        card_upper = card_str.upper()
        
        # 处理大小王
        if card_upper == "JL" or card_upper == "JB":
            raise ValueError(f"Joker cards ({card_str}) are not supported in standard poker")
        
        # 检查是否是纯数字字符串（如 "0", "13", "26", "39"）
        if card_str.isdigit():
            return convert_user_card_to_openspiel(int(card_str))
        
        # 传统格式（如 "As", "Kh", "2d", "Tc", "Xh"）
        return card_string_to_index(card_str)
    
    else:
        raise ValueError(f"Invalid card input type: {type(card_input)}, expected int or str")


def card_string_to_index(card_str: str) -> int:
    """将传统牌面字符串转换为OpenSpiel的card index (0-51)
    
    OpenSpiel格式：suit * 13 + rank
    - Diamonds(0-12): suit=0
    - Spades(13-25): suit=1
    - Hearts(26-38): suit=2
    - Clubs(39-51): suit=3
    
    Args:
        card_str: 牌面字符串，如 "As", "Kh", "2d", "Tc", "Xh"
                 格式：Rank + Suit
                 Rank: 2-9, T(10), X(10), J, Q, K, A
                 Suit: s(spades), h(hearts), d(diamonds), c(clubs)
    
    Returns:
        card index (0-51)
    """
    if len(card_str) < 2:
        raise ValueError(f"Invalid card string: {card_str}, expected at least 2 characters")
    
    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()
    
    # 转换rank: 2~JQKA 对应 0~12，其中10可能用X代替
    rank_names = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, 
        '8': 6, '9': 7, 'T': 8, 'X': 8,  # T和X都表示10
        'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    
    if rank_char not in rank_names:
        raise ValueError(f"Invalid rank: {rank_char}")
    
    rank = rank_names[rank_char]
    
    # 转换suit: OpenSpiel顺序 Diamonds(0-12), Spades(13-25), Hearts(26-38), Clubs(39-51)
    suit_map = {
        'd': 0,  # Diamonds
        's': 1,  # Spades
        'h': 2,  # Hearts
        'c': 3   # Clubs
    }
    
    if suit_char not in suit_map:
        raise ValueError(f"Invalid suit: {suit_char}")
    
    suit = suit_map[suit_char]
    
    # OpenSpiel格式：suit * 13 + rank
    return suit * 13 + rank


def card_index_to_string(card_idx: int) -> str:
    """将card index转换为牌面字符串
    
    Args:
        card_idx: card index (0-51)
    
    Returns:
        牌面字符串，如 "As", "Kh"
    """
    suit_names = ['s', 'h', 'd', 'c']
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    suit = card_idx % 4
    rank = card_idx // 4
    
    return rank_names[rank] + suit_names[suit]


# ==========================================
# 2. 状态构建函数
# ==========================================

def get_player_contributions(state):
    """从 state.to_struct() 获取玩家投入 (player_contributions)"""
    try:
        state_struct = state.to_struct()
        contributions = getattr(state_struct, 'player_contributions', [])
        if contributions:
            return list(contributions)
    except:
        pass
    return []

def create_game_with_config(
    num_players: int,
    blinds: list,  # 盲注列表，如 [50, 100, 0, 0, 0, 0]
    stacks: list,  # 筹码列表，如 [2000, 2000, 2000, 2000, 2000, 2000]
    betting_abstraction: str = "fchpa",
    dealer_pos: int = None  # Dealer位置（必需，0-5）
) -> pyspiel.Game:
    """根据配置创建游戏实例
    
    Args:
        num_players: 玩家数量
        blinds: 盲注列表
        stacks: 筹码列表
        betting_abstraction: 下注抽象
        dealer_pos: Dealer位置（必需，0-5）
    """
    if len(blinds) != num_players:
        raise ValueError(f"Blinds length ({len(blinds)}) != num_players ({num_players})")
    if len(stacks) != num_players:
        raise ValueError(f"Stacks length ({len(stacks)}) != num_players ({num_players})")
    if dealer_pos is None:
        raise ValueError("dealer_pos is required")
    if dealer_pos < 0 or dealer_pos >= num_players:
        raise ValueError(f"Invalid dealer_pos: {dealer_pos}, must be 0-{num_players-1}")
    
    blinds_str = " ".join(map(str, blinds))
    stacks_str = " ".join(map(str, stacks))
    
    # 根据Dealer位置计算firstPlayer
    if num_players == 2:
        # Heads Up: Dealer is SB
        sb_pos = dealer_pos
        bb_pos = (dealer_pos + 1) % num_players
        # Preflop: SB(D) starts. Postflop: BB starts.
        first_player_str = f"{sb_pos + 1} {bb_pos + 1} {bb_pos + 1} {bb_pos + 1}"
    else:
        # Ring Game (3+ players)
        # Dealer -> SB -> BB -> UTG
        sb_pos = (dealer_pos + 1) % num_players
        bb_pos = (dealer_pos + 2) % num_players
        utg_pos = (dealer_pos + 3) % num_players
        # Preflop: UTG starts. Postflop: SB starts.
        # 注意：universal_poker使用1-based indexing
        first_player_str = f"{utg_pos + 1} {sb_pos + 1} {sb_pos + 1} {sb_pos + 1}"
    
    game_string = (
        f"universal_poker("
        f"betting=nolimit,"
        f"numPlayers={num_players},"
        f"numRounds=4,"
        f"blind={blinds_str},"
        f"stack={stacks_str},"
        f"numHoleCards=2,"
        f"numBoardCards=0 3 1 1,"
        f"firstPlayer={first_player_str},"
        f"numSuits=4,"
        f"numRanks=13,"
        f"bettingAbstraction={betting_abstraction}"
        f")"
    )
    
    return pyspiel.load_game(game_string)


def build_state_from_cards(
    game,
    current_player_id: int,
    hole_cards: list,  # 当前玩家的手牌，如 ["As", "Kh"]
    board_cards: list,  # 公共牌，如 ["2d", "3c", "4h"]
    action_history: list,  # 历史动作（只包含玩家动作，不包含发牌动作）
    action_sizings: list = None,  # 每次动作的下注金额，与action_history一一对应
    seed: int = None
) -> pyspiel.State:
    """从指定的手牌和公共牌构建游戏状态
    
    Args:
        game: OpenSpiel游戏实例
        current_player_id: 当前玩家ID (0-5)
        hole_cards: 当前玩家的手牌列表，如 ["As", "Kh"]
        board_cards: 公共牌列表，如 ["2d", "3c", "4h"] 或 []
        action_history: 历史动作列表（玩家动作，不包含发牌动作）
        seed: 随机种子（用于分配其他玩家的手牌）
    
    Returns:
        构建好的游戏状态
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    state = game.new_initial_state()
    num_players = game.num_players()
    num_hole_cards = 2  # 德州扑克每人2张手牌
    
    # 转换牌面字符串为card index（支持用户输入格式）
    current_player_hole_indices = [convert_user_card_to_openspiel(c) for c in hole_cards]
    board_indices = [convert_user_card_to_openspiel(c) for c in board_cards]
    
    # 检查牌面冲突
    all_specified_cards = set(current_player_hole_indices + board_indices)
    if len(all_specified_cards) != len(current_player_hole_indices) + len(board_indices):
        raise ValueError("Duplicate cards detected in hole_cards or board_cards")
    
    # 构建完整的手牌分配
    # 发牌顺序：P0手牌1, P0手牌2, P1手牌1, P1手牌2, ..., P5手牌1, P5手牌2
    all_hole_cards = [None] * (num_players * num_hole_cards)
    
    # 设置当前玩家的手牌
    current_player_start_idx = current_player_id * num_hole_cards
    all_hole_cards[current_player_start_idx] = current_player_hole_indices[0]
    all_hole_cards[current_player_start_idx + 1] = current_player_hole_indices[1]
    
    # 从剩余牌中随机分配其他玩家的手牌
    all_cards = set(range(52))
    used_cards = set(current_player_hole_indices + board_indices)
    available_cards = list(all_cards - used_cards)
    random.shuffle(available_cards)
    
    card_idx = 0
    for i in range(num_players * num_hole_cards):
        if all_hole_cards[i] is None:
            all_hole_cards[i] = available_cards[card_idx]
            card_idx += 1
    
    # 处理chance节点：发所有玩家的手牌
    # 发牌顺序：P0手牌1, P0手牌2, P1手牌1, P1手牌2, ..., P5手牌1, P5手牌2
    hole_card_idx = 0
    while state.is_chance_node() and hole_card_idx < len(all_hole_cards):
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        
        target_card = all_hole_cards[hole_card_idx]
        
        # 找到对应的action（card index）
        if target_card in legal_actions:
            state.apply_action(target_card)
            hole_card_idx += 1
        else:
            # 如果指定的牌不在legal_actions中（不应该发生），随机选择
            action = random.choice(legal_actions)
            state.apply_action(action)
            hole_card_idx += 1
    
    # 处理chance节点：发公共牌
    # 根据当前轮次决定发多少张公共牌
    board_card_idx = 0
    while state.is_chance_node() and board_card_idx < len(board_indices):
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        
        target_card = board_indices[board_card_idx]
        
        if target_card in legal_actions:
            state.apply_action(target_card)
            board_card_idx += 1
        else:
            # 如果指定的牌不在legal_actions中，随机选择
            action = random.choice(legal_actions)
            state.apply_action(action)
            board_card_idx += 1
    
    # 如果还有chance节点（说明公共牌还没发完），随机发完
    # 这通常发生在需要发Turn或River牌时
    while state.is_chance_node():
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        action = random.choice(legal_actions)
        state.apply_action(action)
    
    # 应用历史动作（只包含玩家动作，不包含发牌动作）
    # 注意：如果历史动作中包含chance节点，说明公共牌还没发完，需要先发完公共牌
    for action in action_history:
        if state.is_terminal():
            break
        
        # 如果遇到chance节点，说明需要发公共牌（Turn或River）
        # 这种情况不应该出现在action_history中，因为后端只传玩家动作
        # 但为了健壮性，我们处理一下
        while state.is_chance_node():
            legal_actions = state.legal_actions()
            if not legal_actions:
                break
            # 随机发牌（这些牌不影响当前玩家的信息状态）
            state.apply_action(random.choice(legal_actions))
        
        if state.is_terminal():
            break
        
        # 应用玩家动作
        legal_actions = state.legal_actions()
        if action not in legal_actions:
            raise ValueError(f"Illegal action {action} at current state. Legal actions: {legal_actions}")
        state.apply_action(action)
    
    # 如果还有chance节点（说明需要发Turn或River），随机发完
    while state.is_chance_node():
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        state.apply_action(random.choice(legal_actions))
    
    # 验证action_sizings（如果提供）
    # 注意：前端可能传入增量格式的action_sizings，而OpenSpiel存储的是"bet to"格式
    # 我们尝试兼容两种格式：如果直接比较不匹配，尝试将增量格式转换为"bet to"格式再比较
    if action_sizings is not None:
        try:
            # 从信息状态tensor中提取OpenSpiel计算的action_sizings
            info_state = state.information_state_tensor(current_player_id)
            num_players = game.num_players()
            max_game_length = game.max_game_length()
            
            # 计算action_sizings在tensor中的位置
            # 格式：玩家位置(6) + 手牌(52) + 公共牌(52) + 动作序列(2*max_game_length) + action_sizings(max_game_length)
            header_size = num_players + 52 + 52
            action_seq_size = max_game_length * 2
            action_sizings_start = header_size + action_seq_size
            
            # 提取OpenSpiel计算的所有action_sizings
            openspiel_all_sizings = info_state[action_sizings_start:action_sizings_start + max_game_length]
            
            # 从state.history()中找出玩家动作的位置（排除chance节点）
            # 重建一个临时状态来识别哪些是玩家动作，并计算每个动作前的贡献
            temp_state = game.new_initial_state()
            player_action_indices = []  # 记录玩家动作在完整历史中的索引
            player_contributions_before = []  # 记录每个动作前的玩家贡献（用于转换增量格式）
            
            for action in state.history():
                if temp_state.is_chance_node():
                    # 跳过chance节点（发牌动作）
                    temp_state.apply_action(action)
                else:
                    # 这是玩家动作，记录索引和动作前的贡献
                    current_player = temp_state.current_player()
                    contributions = get_player_contributions(temp_state)
                    if not contributions:
                        contributions = [0] * num_players
                    prev_contribution = contributions[current_player] if current_player < len(contributions) else 0
                    
                    player_action_indices.append(len(temp_state.history()))
                    player_contributions_before.append(prev_contribution)
                    temp_state.apply_action(action)
            
            # 提取玩家动作对应的OpenSpiel action_sizings（"bet to"格式）
            openspiel_player_sizings = [openspiel_all_sizings[i] for i in player_action_indices[:len(action_history)]]
            
            # 首先尝试直接比较（如果前端传的是"bet to"格式）
            direct_mismatches = []
            for i, (provided, calculated) in enumerate(zip(action_sizings, openspiel_player_sizings)):
                if abs(provided - calculated) > 1.0:  # 允许1的误差
                    direct_mismatches.append(i)
            
            # 如果直接比较不匹配，尝试将增量格式转换为"bet to"格式
            if direct_mismatches and len(player_contributions_before) == len(action_sizings):
                # 假设传入的是增量格式，转换为"bet to"格式
                converted_sizings = []
                for i, (increment, prev_contrib) in enumerate(zip(action_sizings, player_contributions_before)):
                    if i < len(action_history):
                        action_id = action_history[i]
                        if action_id == 0:  # Fold
                            converted_sizings.append(0.0)
                        elif action_id == 1:  # Call/Check
                            converted_sizings.append(0.0)
                        else:
                            # Raise/Bet/All-in: "bet to" = 之前的贡献 + 增量
                            bet_to = prev_contrib + increment
                            converted_sizings.append(bet_to)
                    else:
                        converted_sizings.append(increment)
                
                # 使用转换后的格式比较
                converted_mismatches = []
                for i, (converted, calculated) in enumerate(zip(converted_sizings, openspiel_player_sizings)):
                    if abs(converted - calculated) > 1.0:
                        converted_mismatches.append({
                            'index': i,
                            'provided_increment': action_sizings[i],
                            'converted_bet_to': converted,
                            'calculated': float(calculated),
                            'diff': abs(converted - calculated)
                        })
                
                if converted_mismatches:
                    # 转换后仍不匹配，记录警告
                    print(f"⚠️ 警告: action_sizings 不匹配（已尝试增量格式转换）！")
                    print(f"  传入的action_sizings（增量格式）: {action_sizings[:min(10, len(action_sizings))]}...")
                    print(f"  转换后的bet_to格式: {[float(x) for x in converted_sizings[:min(10, len(converted_sizings))]]}...")
                    print(f"  OpenSpiel计算的: {[float(x) for x in openspiel_player_sizings[:min(10, len(openspiel_player_sizings))]]}...")
                    print(f"  不匹配的位置: {[m['index'] for m in converted_mismatches[:5]]}")
                    for m in converted_mismatches[:3]:
                        print(f"    位置 {m['index']}: 增量={m['provided_increment']}, 转换后={m['converted_bet_to']:.1f}, 计算={m['calculated']:.1f}, 差异={m['diff']:.1f}")
                else:
                    # 转换后匹配，说明前端传的是增量格式，这是正常的
                    print(f"✅ action_sizings验证通过（增量格式已转换为bet_to格式）")
            elif direct_mismatches:
                # 直接比较不匹配，且无法转换（缺少贡献信息），记录警告
                print(f"⚠️ 警告: action_sizings 不匹配！")
                print(f"  传入的action_sizings: {action_sizings[:min(10, len(action_sizings))]}...")
                print(f"  OpenSpiel计算的: {[float(x) for x in openspiel_player_sizings[:min(10, len(openspiel_player_sizings))]]}...")
                print(f"  不匹配的位置: {direct_mismatches[:5]}")
            else:
                # 直接比较匹配，说明前端传的是"bet to"格式
                pass
        except Exception as e:
            # 如果验证失败，记录错误但不阻止推理
            print(f"⚠️ 警告: 无法验证action_sizings: {e}")
            import traceback
            traceback.print_exc()
    
    return state


# ==========================================
# 3. 模型加载和推理
# ==========================================

def load_model(model_dir, device='cpu'):
    """加载训练好的模型"""
    global GAME, MODEL, CONFIG, MODEL_DIR
    
    MODEL_DIR = model_dir
    print(f"Loading model from: {model_dir}")
    
    # 读取配置文件（基于 play_gradio.py 的逻辑）
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
            CONFIG = json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    num_players = CONFIG.get('num_players', 6)
    betting_abstraction = CONFIG.get('betting_abstraction', 'fchpa')
    game_string = CONFIG.get('game_string', None)
    
    # 创建游戏
    if game_string:
        try:
            GAME = pyspiel.load_game(game_string)
        except Exception as e:
            print(f"Failed to load game from game_string: {e}")
            GAME = None
    
    if GAME is None:
        # Fallback: 手动创建游戏
        if num_players == 6:
            blinds_str = "50 100 0 0 0 0"
            first_player_str = "3 1 1 1"
        elif num_players == 2:
            blinds_str = "100 50"
            first_player_str = "2 1 1 1"
        else:
            blinds_list = ["50", "100"] + ["0"] * (num_players - 2)
            blinds_str = " ".join(blinds_list)
            first_player_str = " ".join(["3"] + ["1"] * 3)
        
        stacks_str = " ".join(["2000"] * num_players)
        game_string = (
            f"universal_poker("
            f"betting=nolimit,"
            f"numPlayers={num_players},"
            f"numRounds=4,"
            f"blind={blinds_str},"
            f"stack={stacks_str},"
            f"numHoleCards=2,"
            f"numBoardCards=0 3 1 1,"
            f"firstPlayer={first_player_str},"
            f"numSuits=4,"
            f"numRanks=13,"
            f"bettingAbstraction={betting_abstraction}"
            f")"
        )
        GAME = pyspiel.load_game(game_string)
    
    # 加载模型
    save_prefix = CONFIG.get('save_prefix', 'deepcfr_texas')
    policy_filename = f"{save_prefix}_policy_network.pt"
    policy_path = os.path.join(model_dir, policy_filename)
    
    if not os.path.exists(policy_path):
        # 尝试checkpoint格式
        import glob
        pt_files = glob.glob(os.path.join(model_dir, "*_policy_network*.pt"))
        if pt_files:
            # 选择最新的checkpoint
            checkpoint_files = [f for f in pt_files if "_iter" in os.path.basename(f)]
            if checkpoint_files:
                import re
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
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Model file not found: {policy_path}")
    
    # 获取网络结构
    use_simple_feature = CONFIG.get('use_simple_feature', False)
    use_feature_transform = CONFIG.get('use_feature_transform', False)
    policy_layers = tuple(CONFIG.get('policy_layers', [64, 64]))
    
    # 创建测试状态获取embedding size
    test_state = GAME.new_initial_state()
    while test_state.is_chance_node():
        legal_actions = test_state.legal_actions()
        if legal_actions:
            test_state.apply_action(random.choice(legal_actions))
        else:
            break
    
    embedding_size = len(test_state.information_state_tensor(0))
    num_actions = GAME.num_distinct_actions()
    
    # 创建网络（基于 play_gradio.py 的逻辑）
    if use_simple_feature and HAVE_CUSTOM_FEATURES:
        print("Using Simple Feature Model")
        solver = DeepCFRSimpleFeature(
            GAME,
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
        MODEL = solver
        return GAME, MODEL, CONFIG
        
    elif use_feature_transform and HAVE_CUSTOM_FEATURES:
        print("Using Feature Transform Model")
        try:
            from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
            transformed_size = CONFIG.get('transformed_size', 150)
            use_hybrid_transform = CONFIG.get('use_hybrid_transform', True)
            
            solver = DeepCFRWithFeatureTransform(
                GAME,
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
            MODEL = solver
            return GAME, MODEL, CONFIG
        except ImportError:
            print("Import Error for DeepCFRWithFeatureTransform")
            pass

    # Standard MLP（基于 play_gradio.py）
    print("Using Standard MLP")
    state = GAME.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = GAME.num_distinct_actions()
    network = MLP(embedding_size, list(policy_layers), num_actions)
    network = network.to(device)
    network.load_state_dict(torch.load(policy_path, map_location=device))
    network.eval()
    MODEL = network
    
    print(f"Model loaded successfully")
    print(f"  Players: {num_players}")
    print(f"  Betting abstraction: {betting_abstraction}")
    print(f"  Embedding size: {embedding_size}")
    print(f"  Num actions: {num_actions}")
    
    return GAME, MODEL, CONFIG


def map_position_encoding(info_state_tensor, actual_player_id, actual_dealer_pos, training_dealer_pos=5, num_players=6):
    """映射位置编码，使位置角色与训练时一致
    
    训练时：dealer_pos=5, P0=SB, P1=BB, P2=UTG, P3=MP, P4=CO, P5=BTN
    推理时：根据actual_dealer_pos，将位置编码映射到训练时的player_id
    
    Args:
        info_state_tensor: 信息状态tensor（numpy array或torch tensor）
        actual_player_id: 实际的player_id
        actual_dealer_pos: 实际的dealer位置
        training_dealer_pos: 训练时的dealer位置（默认5）
        num_players: 玩家数量
    
    Returns:
        修改后的信息状态tensor
    """
    import numpy as np
    print(f"\n🔧 map_position_encoding被调用: actual_player_id={actual_player_id}, actual_dealer_pos={actual_dealer_pos}, training_dealer_pos={training_dealer_pos}, num_players={num_players}", flush=True)
    
    # 转换为numpy array（如果是torch tensor）
    if hasattr(info_state_tensor, 'cpu'):
        is_torch = True
        device = info_state_tensor.device
        info_state = info_state_tensor.cpu().numpy().copy()
    else:
        is_torch = False
        info_state = np.array(info_state_tensor).copy()
    
    # 计算位置映射
    # 实际位置相对于dealer的偏移
    actual_offset = (actual_player_id - actual_dealer_pos) % num_players
    
    # 训练时相同偏移对应的player_id
    mapped_player_id = (training_dealer_pos + actual_offset) % num_players
    
    # 打印映射前后的位置编码
    position_before = info_state[:num_players].copy()
    actual_position_idx = np.argmax(position_before)
    
    # 位置角色名称映射
    position_names = {
        0: 'SB', 1: 'BB', 2: 'UTG', 3: 'MP', 4: 'CO', 5: 'BTN'
    }
    
    # 计算实际位置角色
    def get_position_role(player_id, dealer_pos, num_players):
        offset = (player_id - dealer_pos) % num_players
        if offset == 0:
            return 'BTN'
        elif offset == 1:
            return 'SB'
        elif offset == 2:
            return 'BB'
        elif offset == 3:
            return 'UTG'
        elif offset == 4:
            return 'MP'
        elif offset == 5:
            return 'CO'
        return f'P{offset}'
    
    actual_role = get_position_role(actual_player_id, actual_dealer_pos, num_players)
    mapped_role = get_position_role(mapped_player_id, training_dealer_pos, num_players)
    
    print(f"\n📍 位置编码映射:", flush=True)
    print(f"  实际: Player {actual_player_id} ({actual_role}), dealer_pos={actual_dealer_pos}", flush=True)
    print(f"  映射: Player {mapped_player_id} ({mapped_role}), dealer_pos={training_dealer_pos}", flush=True)
    print(f"  映射前位置编码: {position_before.tolist()}", flush=True)
    
    # 修改位置编码部分（前num_players个元素）
    # 将实际player_id的位置设为0，映射后的player_id位置设为1
    info_state[actual_player_id] = 0.0
    info_state[mapped_player_id] = 1.0
    
    position_after = info_state[:num_players].copy()
    print(f"  映射后位置编码: {position_after.tolist()}", flush=True)
    print(f"  偏移量: {actual_offset} (相对于dealer)", flush=True)
    
    # 转换回原始格式
    if is_torch:
        import torch
        return torch.FloatTensor(info_state).unsqueeze(0).to(device)
    else:
        return info_state


def get_recommended_action(state, model, device='cpu', dealer_pos=None):
    """获取推荐动作（基于 play_gradio.py 的 get_ai_action）
    
    Args:
        state: 游戏状态
        model: 策略网络或solver
        device: 设备
        dealer_pos: Dealer位置（用于位置编码映射，可选）
    
    Returns:
        (recommended_action, action_probabilities, legal_actions)
    """
    if state.is_terminal():
        return None, {}, []
    
    if state.is_chance_node():
        return None, {}, []
    
    player = state.current_player()
    legal_actions = state.legal_actions()
    
    if not legal_actions:
        return None, {}, []
    
    # Check if model is a solver with action_probabilities（基于 play_gradio.py）
    if hasattr(model, 'action_probabilities'):
        # 对于solver，如果提供了dealer_pos，需要手动进行位置编码映射
        # 因为action_probabilities内部会调用information_state_tensor，我们无法直接修改
        # 所以直接使用策略网络，而不是action_probabilities方法
        if dealer_pos is not None and hasattr(model, '_policy_network'):
            # 使用策略网络，并进行位置编码映射
            info_state_raw = state.information_state_tensor(player)
            info_state = torch.FloatTensor(info_state_raw).unsqueeze(0).to(device)
            num_players = state.get_game().num_players()
            
            # 打印手牌和公共牌信息（用于调试）
            hole_cards_start = num_players
            hole_cards_end = hole_cards_start + 52
            board_cards_start = hole_cards_end
            board_cards_end = board_cards_start + 52
            
            hole_cards_bits = info_state_raw[hole_cards_start:hole_cards_end]
            board_cards_bits = info_state_raw[board_cards_start:board_cards_end]
            hole_cards = [i for i, bit in enumerate(hole_cards_bits) if bit > 0.5]
            board_cards = [i for i, bit in enumerate(board_cards_bits) if bit > 0.5]
            
            def card_index_to_string(card_idx):
                """将OpenSpiel的card index转换为字符串"""
                suits = ['d', 'c', 'h', 's']  # OpenSpiel的顺序：Diamonds, Clubs, Hearts, Spades
                ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                suit_idx = card_idx // 13
                rank_idx = card_idx % 13
                return ranks[rank_idx] + suits[suit_idx]
            
            hole_cards_str = [card_index_to_string(c) for c in hole_cards]
            board_cards_str = [card_index_to_string(c) for c in board_cards] if board_cards else []
            
            # 打印动作序列和下注金额（用于调试）
            max_game_length = state.get_game().max_game_length()
            action_seq_start = board_cards_end
            action_seq_end = action_seq_start + max_game_length * 2
            action_sizings_start = action_seq_end
            action_sizings_end = action_sizings_start + max_game_length
            
            action_seq_bits = info_state_raw[action_seq_start:action_seq_end]
            action_sizings_bits = info_state_raw[action_sizings_start:action_sizings_end]
            
            # 解析动作序列
            action_seq = []
            for i in range(max_game_length):
                bit0 = action_seq_bits[2*i]
                bit1 = action_seq_bits[2*i+1]
                if bit0 > 0.5 and bit1 < 0.5:
                    action_seq.append('c')  # call
                elif bit0 < 0.5 and bit1 > 0.5:
                    action_seq.append('p')  # raise
                elif bit0 > 0.5 and bit1 > 0.5:
                    action_seq.append('a')  # all-in
                elif bit0 < 0.5 and bit1 < 0.5:
                    action_seq.append('f')  # fold/deal
                else:
                    action_seq.append('?')
            
            # 提取非零的action_sizings
            nonzero_sizings = [(i, float(s)) for i, s in enumerate(action_sizings_bits) if s > 0.5]
            
            # 找出动作序列中非'f'的位置（实际玩家动作）
            actual_actions = []
            for i, act in enumerate(action_seq):
                if act != 'f' or (i < len(action_sizings_bits) and action_sizings_bits[i] > 0.5):
                    actual_actions.append((i, act, float(action_sizings_bits[i]) if i < len(action_sizings_bits) else 0.0))
            
            print(f"\n🃏 Solver模型信息状态: player={player}, 手牌={hole_cards_str}, 公共牌={board_cards_str}", flush=True)
            print(f"   动作序列(前20个): {action_seq[:20]}", flush=True)
            print(f"   action_sizings(非零): {nonzero_sizings[:10]}", flush=True)
            print(f"   实际动作(前10个): {actual_actions[:10]}", flush=True)
            
            print(f"\n🔍 Solver模型，准备进行位置编码映射: player={player}, dealer_pos={dealer_pos}, num_players={num_players}", flush=True)
            
            # ⚠️ 关键修复：禁用位置编码映射！
            # 
            # 问题分析：
            # OpenSpiel的information_state_tensor(player)返回的是：
            # - 位置编码：values[player] = 1 - "我是player"
            # - 手牌：HoleCards(player) - 该player的手牌
            #
            # 如果我们映射位置编码（比如从Player 0映射到Player 2），但手牌编码保持不变：
            # - 位置编码：[0,0,1,0,0,0] - "我是Player 2"
            # - 手牌编码：Player 0的手牌（Th, Ts）
            # 这是不一致的！模型看到的是"我是Player 2，但我的手牌是Player 0的手牌"
            #
            # 正确的做法：
            # 不应该进行位置编码映射，因为：
            # 1. OpenSpiel的information_state_tensor已经正确地返回了该player的手牌
            # 2. 位置编码只是表示"我是player"，不应该改变
            # 3. 如果模型训练时使用了特定的dealer_pos，那可能是因为训练数据中dealer_pos固定
            # 4. 但推理时，我们应该相信OpenSpiel的信息状态是正确的
            # 5. 模型应该能够处理不同dealer位置的情况，因为位置编码表示的是player ID，而不是位置角色
            #
            # 如果模型确实需要位置角色信息，应该在训练时使用相对位置特征，而不是绝对位置编码
            
            print(f"⚠️ 警告: 已禁用位置编码映射，直接使用OpenSpiel的信息状态", flush=True)
            print(f"   原因: 位置编码映射会导致位置和手牌不一致，影响模型推理", flush=True)
            print(f"   位置编码表示'我是player'，不应该改变", flush=True)
            print(f"   手牌编码是相对于实际player的，不应该映射", flush=True)
            
            with torch.no_grad():
                logits = model._policy_network(info_state)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # 构建概率字典
            legal_probs = {}
            for action in legal_actions:
                legal_probs[action] = float(probs[action])
            
            # 归一化
            total_prob = sum(legal_probs.values())
            if total_prob > 0:
                for action in legal_probs:
                    legal_probs[action] /= total_prob
            else:
                uniform_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    legal_probs[action] = uniform_prob
            
            # 选择推荐动作（概率最大的）
            if legal_probs:
                recommended_action = max(legal_probs.items(), key=lambda x: x[1])[0]
            else:
                recommended_action = legal_actions[0]
            
            return recommended_action, legal_probs, legal_actions
        else:
            # 如果没有dealer_pos或没有_policy_network，使用原始的action_probabilities
            if dealer_pos is None:
                print(f"\n⚠️ Solver模型，但未提供dealer_pos，使用原始action_probabilities（位置编码可能不正确）", flush=True)
            probs_dict = model.action_probabilities(state, player)
            actions = list(probs_dict.keys())
            probs = list(probs_dict.values())
            
            # 构建概率字典
            legal_probs = {}
            total_prob = sum(probs)
            if total_prob > 0:
                for a, p in zip(actions, probs):
                    if a in legal_actions:
                        legal_probs[a] = float(p / total_prob)
            else:
                uniform_prob = 1.0 / len(legal_actions)
                for a in legal_actions:
                    legal_probs[a] = uniform_prob
            
            # 选择推荐动作（概率最大的）
            if legal_probs:
                recommended_action = max(legal_probs.items(), key=lambda x: x[1])[0]
            else:
                recommended_action = legal_actions[0]
            
            return recommended_action, legal_probs, legal_actions
    
    # Standard Network（基于 play_gradio.py）
    info_state_raw = state.information_state_tensor(player)
    info_state = torch.FloatTensor(info_state_raw).unsqueeze(0).to(device)
    
    # 打印手牌和公共牌信息（用于调试）
    num_players = state.get_game().num_players()
    hole_cards_start = num_players
    hole_cards_end = hole_cards_start + 52
    board_cards_start = hole_cards_end
    board_cards_end = board_cards_start + 52
    
    hole_cards_bits = info_state_raw[hole_cards_start:hole_cards_end]
    board_cards_bits = info_state_raw[board_cards_start:board_cards_end]
    hole_cards = [i for i, bit in enumerate(hole_cards_bits) if bit > 0.5]
    board_cards = [i for i, bit in enumerate(board_cards_bits) if bit > 0.5]
    
    def card_index_to_string(card_idx):
        """将OpenSpiel的card index转换为字符串"""
        suits = ['d', 'c', 'h', 's']  # OpenSpiel的顺序：Diamonds, Clubs, Hearts, Spades
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suit_idx = card_idx // 13
        rank_idx = card_idx % 13
        return ranks[rank_idx] + suits[suit_idx]
    
    hole_cards_str = [card_index_to_string(c) for c in hole_cards]
    board_cards_str = [card_index_to_string(c) for c in board_cards] if board_cards else []
    print(f"\n🃏 信息状态中的牌: player={player}, 手牌={hole_cards_str}, 公共牌={board_cards_str}", flush=True)
    
    # 位置编码映射：如果提供了dealer_pos，将位置编码映射到训练时的位置
    if dealer_pos is not None:
        print(f"\n🔍 准备进行位置编码映射: player={player}, dealer_pos={dealer_pos}, num_players={num_players}", flush=True)
        info_state = map_position_encoding(
            info_state.squeeze(0),  # 去掉batch维度
            player,
            dealer_pos,
            training_dealer_pos=5,  # 训练时dealer_pos=5
            num_players=num_players
        )
    else:
        print(f"\n⚠️ 未提供dealer_pos，跳过位置编码映射", flush=True)
    with torch.no_grad():
        logits = model(info_state)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # 只保留合法动作的概率
    legal_probs = {}
    for action in legal_actions:
        legal_probs[action] = float(probs[action])
    
    # 归一化
    total_prob = sum(legal_probs.values())
    if total_prob > 0:
        for action in legal_probs:
            legal_probs[action] /= total_prob
    else:
        # 如果所有概率都是0，均匀分布
        uniform_prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            legal_probs[action] = uniform_prob
    
    # 选择推荐动作（概率最大的）
    recommended_action = max(legal_probs.items(), key=lambda x: x[1])[0]
    
    return recommended_action, legal_probs, legal_actions


# ==========================================
# 4. API接口
# ==========================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'success': True,
        'message': 'API server is running',
        'model_loaded': MODEL is not None,
        'game_loaded': GAME is not None
    })


@app.route('/api/v1/recommend_action', methods=['POST'])
def recommend_action():
    """获取推荐动作
    
    请求格式:
    {
        "player_id": 0,
        "hole_cards": ["As", "Kh"],
        "board_cards": ["2d", "3c", "4h"],
        "action_history": [0, 1, 2, ...],  // 只包含玩家动作，不包含发牌动作
        "action_sizings": [0, 0, 100, ...],  // 每次动作的下注金额，与action_history一一对应
        "blinds": [50, 100, 0, 0, 0, 0],  // 可选，如果不传则使用模型默认配置
        "stacks": [2000, 2000, 2000, 2000, 2000, 2000],  // 可选，如果不传则使用模型默认配置
        "seed": 12345  // 可选，用于随机分配其他玩家的手牌
    }
    
    响应格式:
    {
        "success": true,
        "data": {
            "recommended_action": 1,
            "action_probabilities": {"0": 0.05, "1": 0.75, "2": 0.15, "3": 0.05},
            "legal_actions": [0, 1, 2, 3],
            "current_player": 0
        },
        "error": null
    }
    """
    if MODEL is None or GAME is None:
        return jsonify({
            'success': False,
            'data': None,
            'error': 'Model or game not loaded'
        }), 500
    
    try:
        data = request.get_json()
        
        # 验证输入
        if 'player_id' not in data:
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Missing required field: player_id'
            }), 400
        
        if 'hole_cards' not in data:
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Missing required field: hole_cards'
            }), 400
        
        if 'board_cards' not in data:
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Missing required field: board_cards'
            }), 400
        
        if 'action_history' not in data:
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Missing required field: action_history'
            }), 400
        
        player_id = data['player_id']
        hole_cards = data['hole_cards']
        board_cards = data['board_cards']
        action_history = data['action_history']
        action_sizings = data.get('action_sizings', None)  # 每次动作的下注金额
        blinds = data.get('blinds', None)
        stacks = data.get('stacks', None)
        seed = data.get('seed', None)
        
        # 验证action_sizings长度（如果提供）
        if action_sizings is not None and len(action_sizings) != len(action_history):
            return jsonify({
                'success': False,
                'data': None,
                'error': f'action_sizings length ({len(action_sizings)}) != action_history length ({len(action_history)})'
            }), 400
        
        # 确定玩家数量
        if blinds is not None:
            num_players = len(blinds)
        elif stacks is not None:
            num_players = len(stacks)
        else:
            # 使用模型默认配置
            num_players = CONFIG.get('num_players', 6) if CONFIG else 6
        
        # 验证玩家ID
        if player_id < 0 or player_id >= num_players:
            return jsonify({
                'success': False,
                'data': None,
                'error': f'Invalid player_id: {player_id}, must be 0-{num_players-1}'
            }), 400
        
        # 创建游戏实例（如果提供了blinds和stacks，使用它们；否则使用模型默认配置）
        dealer_pos = data.get('dealer_pos', None)  # 获取dealer_pos（用于位置编码映射）
        print(f"\n📥 API请求接收: player_id={player_id}, dealer_pos={dealer_pos}, blinds={blinds is not None}, stacks={stacks is not None}", flush=True)
        if blinds is not None and stacks is not None:
            betting_abstraction = CONFIG.get('betting_abstraction', 'fchpa') if CONFIG else 'fchpa'
            game = create_game_with_config(num_players, blinds, stacks, betting_abstraction, dealer_pos)
        else:
            # 使用全局游戏实例（从模型配置加载）
            if GAME is None:
                return jsonify({
                    'success': False,
                    'data': None,
                    'error': 'Game not loaded and no blinds/stacks provided'
                }), 500
            game = GAME
        
        # 验证手牌数量
        if len(hole_cards) != 2:
            return jsonify({
                'success': False,
                'data': None,
                'error': f'Invalid hole_cards length: {len(hole_cards)}, must be 2'
            }), 400
        
        # 构建状态
        state = build_state_from_cards(
            game=game,
            current_player_id=player_id,
            hole_cards=hole_cards,
            board_cards=board_cards,
            action_history=action_history,
            action_sizings=action_sizings,  # 传递action_sizings用于验证
            seed=seed
        )
        
        # 验证状态
        if state.is_terminal():
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Game is already terminal'
            }), 400
        
        if state.is_chance_node():
            return jsonify({
                'success': False,
                'data': None,
                'error': 'State is at chance node (cards not fully dealt)'
            }), 400
        
        # 验证当前玩家
        current_player = state.current_player()
        if current_player != player_id:
            return jsonify({
                'success': False,
                'data': None,
                'error': f'Current player mismatch: expected {player_id}, got {current_player}'
            }), 400
        
        # 获取推荐动作（传入dealer_pos用于位置编码映射）
        print(f"\n🎯 调用get_recommended_action: player_id={player_id}, dealer_pos={dealer_pos}", flush=True)
        recommended_action, action_probs, legal_actions = get_recommended_action(
            state, MODEL, DEVICE, dealer_pos=dealer_pos
        )
        print(f"✅ get_recommended_action返回: recommended_action={recommended_action}", flush=True)
        
        if recommended_action is None:
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Failed to get recommended action'
            }), 500
        
        # 转换action_probs的key为字符串（JSON要求）
        action_probs_str = {str(k): v for k, v in action_probs.items()}
        
        return jsonify({
            'success': True,
            'data': {
                'recommended_action': int(recommended_action),
                'action_probabilities': action_probs_str,
                'legal_actions': [int(a) for a in legal_actions],
                'current_player': int(current_player)
            },
            'error': None
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'data': None,
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'data': None,
            'error': f'Internal error: {str(e)}'
        }), 500


@app.route('/api/v1/reload_model', methods=['POST'])
def reload_model():
    """重新加载模型（支持动态切换模型）
    
    请求格式:
    {
        "model_dir": "models/deepcfr_stable_run",
        "device": "cpu"  // 可选，默认使用当前设备
    }
    """
    global GAME, MODEL, CONFIG
    
    try:
        data = request.get_json() or {}
        model_dir = data.get('model_dir', MODEL_DIR)
        device = data.get('device', DEVICE)
        
        if model_dir is None:
            return jsonify({
                'success': False,
                'error': 'model_dir not provided and no default model loaded'
            }), 400
        
        # 加载新模型
        load_model(model_dir, device=device)
        
        return jsonify({
            'success': True,
            'message': f'Model reloaded from {model_dir}',
            'model_dir': model_dir,
            'device': device
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to reload model: {str(e)}'
        }), 500


@app.route('/api/v1/action_mapping', methods=['GET'])
def action_mapping():
    """获取动作映射表"""
    betting_abstraction = CONFIG.get('betting_abstraction', 'fchpa') if CONFIG else 'fchpa'
    
    if betting_abstraction == 'fchpa':
        mapping = {
            '0': 'Fold',
            '1': 'Call/Check',
            '2': 'Pot (Raise to Pot)',
            '3': 'All-in',
            '4': 'Half-Pot (Raise to Half Pot)'
        }
    elif betting_abstraction == 'fcpa':
        mapping = {
            '0': 'Fold',
            '1': 'Call/Check',
            '2': 'Pot (Raise to Pot)',
            '3': 'All-in'
        }
    else:
        mapping = {
            '0': 'Fold',
            '1': 'Call/Check'
        }
    
    return jsonify({
        'success': True,
        'betting_abstraction': betting_abstraction,
        'action_mapping': mapping
    })


def main():
    parser = argparse.ArgumentParser(description='API Server for Poker Recommendation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory (containing config.json and model files)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind to (default: 5000)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    global DEVICE
    DEVICE = args.device
    
    # 加载模型
    try:
        load_model(args.model_dir, device=DEVICE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 启动服务器
    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"Device: {DEVICE}")
    print(f"\nAPI endpoints:")
    print(f"  GET  /api/v1/health - Health check")
    print(f"  POST /api/v1/recommend_action - Get recommended action")
    print(f"  POST /api/v1/reload_model - Reload model (dynamic model switching)")
    print(f"  GET  /api/v1/action_mapping - Get action mapping")
    print()
    
    # 确保print输出不被缓冲
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
