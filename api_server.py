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
import torch.nn as nn
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

# 全局变量：模型和游戏（支持多模型）
MODELS = {}  # {num_players: model} 例如 {5: model_5p, 6: model_6p}
CONFIGS = {}  # {num_players: config} 例如 {5: config_5p, 6: config_6p}
GAMES = {}  # {num_players: game} 例如 {5: game_5p, 6: game_6p}（可选，主要用于默认配置）
DEVICE = 'cpu'
MODEL_DIRS = {}  # {num_players: model_dir} 例如 {5: dir_5p, 6: dir_6p}

# 向后兼容的全局变量（指向默认模型）
GAME = None  # 向后兼容，指向GAMES中的第一个或默认模型
MODEL = None  # 向后兼容，指向MODELS中的第一个或默认模型
CONFIG = None  # 向后兼容，指向CONFIGS中的第一个或默认配置
MODEL_DIR = None  # 向后兼容

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


def normalize_info_state_action_sizings(info_state, game, max_stack=None):
    """归一化information_state_tensor中的action_sizings部分
    
    训练时模型使用了归一化的action_sizings（除以max_stack），
    推理时也需要归一化以保持一致。
    
    Args:
        info_state: information_state_tensor（numpy array或torch tensor）
        game: OpenSpiel游戏实例
        max_stack: 最大筹码值（如果为None，从游戏配置解析）
    
    Returns:
        归一化后的info_state（numpy array）
    """
    import numpy as np
    
    # 转换为numpy array
    if isinstance(info_state, torch.Tensor):
        info_state_np = info_state.cpu().numpy()
        is_torch = True
    else:
        info_state_np = np.array(info_state)
        is_torch = False
    
    # 如果已经是2D，取第一个样本
    if len(info_state_np.shape) == 2:
        info_state_np = info_state_np[0]
    
    num_players = game.num_players()
    max_game_length = game.max_game_length()
    
    # 计算action_sizings的起始位置
    # 格式：玩家位置(N) + 手牌(52) + 公共牌(52) + 动作序列(2*max_game_length) + action_sizings(max_game_length)
    header_size = num_players + 52 + 52
    action_seq_size = max_game_length * 2
    action_sizings_start = header_size + action_seq_size
    
    # 获取max_stack
    if max_stack is None:
        # 从游戏配置解析
        import re
        game_string = str(game)
        match = re.search(r'stack=([\d\s]+)', game_string)
        if match:
            stack_str = match.group(1).strip()
            stack_values = stack_str.split()
            if stack_values:
                try:
                    max_stack = int(stack_values[0])
                except ValueError:
                    max_stack = 2000  # 默认值
        else:
            max_stack = 2000  # 默认值
    
    # 归一化action_sizings部分
    # 使用log归一化：log(1 + amount) / log(1 + max_stack)
    # 与训练时保持一致（deep_cfr_simple_feature.py）
    if action_sizings_start < len(info_state_np):
        action_sizings_end = action_sizings_start + max_game_length
        if action_sizings_end <= len(info_state_np):
            # 使用log归一化，避免小注值太小被其他特征稀释
            log_max_stack = np.log1p(max_stack)
            info_state_np[action_sizings_start:action_sizings_end] = np.log1p(
                np.maximum(info_state_np[action_sizings_start:action_sizings_end], 0)
            ) / log_max_stack
    
    return info_state_np

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
    debug_info = []  # 记录调试信息
    while state.is_chance_node() and hole_card_idx < len(all_hole_cards):
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        
        target_card = all_hole_cards[hole_card_idx]
        expected_player = hole_card_idx // num_hole_cards
        expected_card_idx = hole_card_idx % num_hole_cards
        
        # 找到对应的action（card index）
        if target_card in legal_actions:
            state.apply_action(target_card)
            debug_info.append((hole_card_idx, expected_player, expected_card_idx, target_card, True))
            hole_card_idx += 1
        else:
            # 如果指定的牌不在legal_actions中（不应该发生），随机选择
            action = random.choice(legal_actions)
            state.apply_action(action)
            debug_info.append((hole_card_idx, expected_player, expected_card_idx, target_card, False, action))
            hole_card_idx += 1
        
        # 注意：不在发牌过程中验证，因为此时 to_struct() 可能返回中间状态
        # 验证将在所有chance节点处理完后进行
    
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
    
    # 在所有chance节点处理完后，验证当前玩家的手牌
    # 使用 information_state_tensor 验证（更准确），忽略手牌顺序
    try:
        info_state = state.information_state_tensor(current_player_id)
        num_players = game.num_players()
        hole_cards_start = num_players
        hole_cards_end = hole_cards_start + 52
        hole_cards_bits = info_state[hole_cards_start:hole_cards_end]
        hole_cards_indices = [i for i, bit in enumerate(hole_cards_bits) if bit > 0.5]
        
        # 转换为字符串格式
        suits = ['d', 's', 'h', 'c']  # OpenSpiel顺序
        ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        actual_hand_set = set([ranks[c%13] + suits[c//13] for c in hole_cards_indices])
        expected_hand_set = set([f"{ranks[c%13]}{suits[c//13]}" for c in current_player_hole_indices])
        
        # 忽略顺序，只比较牌的集合
        if actual_hand_set != expected_hand_set:
            actual_hand_str = "".join(sorted(actual_hand_set))
            expected_hand_str = "".join(sorted(expected_hand_set))
            print(f"⚠️ 警告: Player {current_player_id}手牌不匹配！期望: {expected_hand_str}, 实际: {actual_hand_str}", flush=True)
            print(f"  调试信息: {debug_info[-num_hole_cards:] if len(debug_info) >= num_hole_cards else debug_info}", flush=True)
    except Exception as e:
        # 验证失败不影响功能
        pass
    
    # 应用历史动作（只包含玩家动作，不包含发牌动作）
    # 注意：如果历史动作中包含chance节点，说明公共牌还没发完，需要先发完公共牌
    action_history_debug = []  # 记录调试信息
    for i, action in enumerate(action_history):
        if state.is_terminal():
            break
        
        # 如果遇到chance节点，说明需要发公共牌（Turn或River）
        # 这种情况不应该出现在action_history中，因为后端只传玩家动作
        # 但为了健壮性，我们处理一下
        chance_actions_applied = 0
        while state.is_chance_node():
            legal_actions = state.legal_actions()
            if not legal_actions:
                break
            # 随机发牌（这些牌不影响当前玩家的信息状态）
            chance_action = random.choice(legal_actions)
            state.apply_action(chance_action)
            chance_actions_applied += 1
        
        if state.is_terminal():
            break
        
        # 应用玩家动作
        current_player_before = state.current_player()
        legal_actions = state.legal_actions()
        
        # 记录调试信息
        action_str = {0: 'Fold', 1: 'Call/Check', 2: 'Pot', 3: 'All-in', 4: 'Half-Pot'}.get(action, f'Unknown({action})')
        action_history_debug.append({
            'step': i,
            'action': action,
            'action_str': action_str,
            'current_player': current_player_before,
            'legal_actions': legal_actions,
            'chance_actions_applied': chance_actions_applied
        })
        
        if action not in legal_actions:
            error_msg = f"Illegal action {action} ({action_str}) at step {i}, current player {current_player_before}. Legal actions: {legal_actions}"
            print(f"❌ {error_msg}", flush=True)
            print(f"   动作历史调试信息: {action_history_debug}", flush=True)
            raise ValueError(error_msg)
        
        state.apply_action(action)
        
        # 记录应用后的状态
        action_history_debug[-1]['current_player_after'] = state.current_player()
        action_history_debug[-1]['is_terminal'] = state.is_terminal()
    
    # 如果还有chance节点（说明需要发Turn或River），随机发完
    while state.is_chance_node():
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        state.apply_action(random.choice(legal_actions))
    
    # 验证状态重建：检查信息状态中的动作序列
    if len(action_history) > 0:
        try:
            info_state = state.information_state_tensor(current_player_id)
            num_players = game.num_players()
            max_game_length = game.max_game_length()
            
            # 解析信息状态中的动作序列
            header_size = num_players + 52 + 52
            action_seq_start = header_size
            action_seq_end = action_seq_start + max_game_length * 2
            action_seq_bits = info_state[action_seq_start:action_seq_end]
            
            # 解析动作序列
            action_seq_parsed = []
            for i in range(max_game_length):
                bit0 = action_seq_bits[2*i]
                bit1 = action_seq_bits[2*i+1]
                if bit0 > 0.5 and bit1 < 0.5:
                    action_seq_parsed.append('c')  # call
                elif bit0 < 0.5 and bit1 > 0.5:
                    action_seq_parsed.append('p')  # raise
                elif bit0 > 0.5 and bit1 > 0.5:
                    action_seq_parsed.append('a')  # all-in
                elif bit0 < 0.5 and bit1 < 0.5:
                    action_seq_parsed.append('f')  # fold/deal
                else:
                    action_seq_parsed.append('?')
            
            # 找出非'f'的动作（实际玩家动作）
            actual_player_actions = []
            for i, act in enumerate(action_seq_parsed):
                if act != 'f':
                    actual_player_actions.append((i, act))
            
            # 将输入的action_history转换为动作字符
            action_map = {0: 'f', 1: 'c', 2: 'p', 3: 'a', 4: 'h'}  # h for half-pot
            input_action_chars = [action_map.get(a, '?') for a in action_history]
            
            # 打印验证信息
            print(f"\n🔍 状态重建验证 (Player {current_player_id}):", flush=True)
            print(f"   输入的action_history: {action_history} -> {input_action_chars}", flush=True)
            print(f"   信息状态中的动作序列(前20个): {action_seq_parsed[:20]}", flush=True)
            print(f"   实际玩家动作位置: {actual_player_actions[:10]}", flush=True)
            
            # 验证是否有加注动作
            has_raise_in_input = 2 in action_history  # action 2 = Pot
            has_raise_in_state = any(act == 'p' for _, act in actual_player_actions)
            
            if has_raise_in_input and not has_raise_in_state:
                print(f"⚠️ 警告: 输入包含加注动作(action=2)，但信息状态中未找到加注动作！", flush=True)
            elif has_raise_in_input and has_raise_in_state:
                print(f"✅ 验证通过: 输入包含加注动作，信息状态中也包含加注动作", flush=True)
            
            # 打印动作应用详情
            if action_history_debug:
                print(f"   动作应用详情:", flush=True)
                for debug_info in action_history_debug:
                    print(f"     步骤{debug_info['step']}: Player {debug_info['current_player']} -> {debug_info['action_str']} ({debug_info['action']})", flush=True)
                    if debug_info['chance_actions_applied'] > 0:
                        print(f"        (应用了{debug_info['chance_actions_applied']}个chance动作)", flush=True)
            
        except Exception as e:
            print(f"⚠️ 状态重建验证失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
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

def load_model(model_dir, device='cpu', num_players=None):
    """加载训练好的模型
    
    Args:
        model_dir: 模型目录路径
        device: 设备（cpu/cuda）
        num_players: 玩家数量（如果为None，从config.json读取）
    """
    global GAME, MODEL, CONFIG, MODEL_DIR, MODELS, CONFIGS, GAMES, MODEL_DIRS
    
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
            config = json.load(f)
    else:
        # 兼容老模型：如果没有 config.json，使用默认配置（标准MLP）
        print(f"⚠️  Config file not found: {config_path}, using default config for legacy model")
        config = {
            'use_simple_feature': False,
            'use_feature_transform': False,
            'policy_layers': [64, 64],  # 默认层数，老模型常用
            'betting_abstraction': 'fchpa'
        }
    
    # 如果num_players未指定，从config读取
    if num_players is None:
        num_players = config.get('num_players', 6)
    
    # 存储到对应的字典中
    CONFIGS[num_players] = config
    MODEL_DIRS[num_players] = model_dir
    
    # 向后兼容：如果是第一个模型，设置为默认
    if CONFIG is None:
        CONFIG = config
    
    betting_abstraction = config.get('betting_abstraction', 'fchpa')
    game_string = config.get('game_string', None)
    
    # 创建游戏
    game = None
    if game_string:
        try:
            game = pyspiel.load_game(game_string)
        except Exception as e:
            print(f"Failed to load game from game_string: {e}")
            game = None
    
    if game is None:
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
        game = pyspiel.load_game(game_string)
    
    # 存储游戏实例
    GAMES[num_players] = game
    
    # 向后兼容：如果是第一个模型，设置为默认
    if GAME is None:
        GAME = game
    
    # 加载模型
    save_prefix = config.get('save_prefix', 'deepcfr_texas')
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
    use_simple_feature = config.get('use_simple_feature', False)
    use_feature_transform = config.get('use_feature_transform', False)
    policy_layers = tuple(config.get('policy_layers', [64, 64]))
    
    # 创建测试状态获取embedding size
    test_state = game.new_initial_state()
    while test_state.is_chance_node():
        legal_actions = test_state.legal_actions()
        if legal_actions:
            test_state.apply_action(random.choice(legal_actions))
        else:
            break
    
    embedding_size = len(test_state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    # 创建网络（基于 play_gradio.py 的逻辑）
    if use_simple_feature and HAVE_CUSTOM_FEATURES:
        print(f"Using Simple Feature Model (num_players={num_players})")
        
        # 先加载权重，检测特征维度
        state_dict = torch.load(policy_path, map_location=device)
        print(f"state_dict: {state_dict.keys()}")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 自动检测手动特征维度（兼容老模型7维和新模型1维）
        from deep_cfr_simple_feature import detect_manual_feature_size_from_state_dict
        detected_feature_size = detect_manual_feature_size_from_state_dict(
            new_state_dict, embedding_size
        )
        
        if detected_feature_size is not None:
            print(f"  ✓ 自动检测到特征维度: {detected_feature_size}维 ({'老版本' if detected_feature_size == 7 else '新版本'})")
            manual_feature_size = detected_feature_size
        else:
            # 如果无法检测，默认使用新版本（1维）
            print(f"  ⚠️  无法自动检测特征维度，使用默认值: 1维（新版本）")
            manual_feature_size = 23
        
        # 创建 solver（指定特征维度）
        solver = DeepCFRSimpleFeature(
            game,
            policy_network_layers=policy_layers,
            advantage_network_layers=(32, 32),
            num_iterations=1,
            num_traversals=1,
            learning_rate=1e-4,
            device=device,
            manual_feature_size=manual_feature_size  # 传递特征维度
        )
        
        solver._policy_network.load_state_dict(new_state_dict)
        solver._policy_network.eval()
        
        # 存储模型到字典中
        MODELS[num_players] = solver
        
        # 向后兼容：如果是第一个模型，设置为默认
        if MODEL is None:
            MODEL = solver
            GAME = game
            CONFIG = config
        
        print(f"Model loaded successfully (num_players={num_players})")
        return game, solver, config
        
    elif use_feature_transform and HAVE_CUSTOM_FEATURES:
        print(f"Using Feature Transform Model (num_players={num_players})")
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
            
            # 存储模型到字典中
            MODELS[num_players] = solver
            
            # 向后兼容：如果是第一个模型，设置为默认
            if MODEL is None:
                MODEL = solver
                GAME = game
                CONFIG = config
            
            print(f"Model loaded successfully (num_players={num_players})")
            return game, solver, config
        except ImportError:
            print("Import Error for DeepCFRWithFeatureTransform")
            pass

    # Standard MLP（老模型或默认模型）
    print(f"Using Standard MLP (num_players={num_players})")
    state = game.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    network = MLP(embedding_size, list(policy_layers), num_actions)
    network = network.to(device)
    
    # 处理 DataParallel（老模型可能也有）
    state_dict = torch.load(policy_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    network.load_state_dict(new_state_dict)
    network.eval()
    
    # 存储模型到字典中
    MODELS[num_players] = network
    
    # 向后兼容：如果是第一个模型，设置为默认
    if MODEL is None:
        MODEL = network
        GAME = game
        CONFIG = config
    
    print(f"Model loaded successfully")
    print(f"  Players: {num_players}")
    print(f"  Betting abstraction: {betting_abstraction}")
    print(f"  Embedding size: {embedding_size}")
    print(f"  Num actions: {num_actions}")
    
    return game, network, config


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
            num_players = state.get_game().num_players()
            
            # 归一化action_sizings部分（与训练时保持一致）
            # 获取max_stack（从游戏配置或stacks中）
            max_stack = None
            if hasattr(model, '_policy_network'):
                # 尝试从模型获取max_stack
                policy_net = model._policy_network
                if isinstance(policy_net, nn.DataParallel):
                    policy_net = policy_net.module
                if hasattr(policy_net, 'max_stack'):
                    max_stack = policy_net.max_stack
            
            # 如果模型没有max_stack，从游戏配置解析
            if max_stack is None:
                import re
                game_string = str(state.get_game())
                match = re.search(r'stack=([\d\s]+)', game_string)
                if match:
                    stack_str = match.group(1).strip()
                    stack_values = stack_str.split()
                    if stack_values:
                        try:
                            max_stack = int(stack_values[0])
                        except ValueError:
                            max_stack = 2000  # 默认值
                else:
                    max_stack = 2000  # 默认值
            
            # 归一化action_sizings
            # 注意：在归一化前打印原始值，归一化后打印归一化后的值
            max_game_length = state.get_game().max_game_length()
            header_size = num_players + 52 + 52
            action_seq_size = max_game_length * 2
            action_sizings_start = header_size + action_seq_size
            action_sizings_end = action_sizings_start + max_game_length
            
            if action_sizings_start < len(info_state_raw):
                original_sizings = info_state_raw[action_sizings_start:action_sizings_end].copy()
                nonzero_original = [(i, float(s)) for i, s in enumerate(original_sizings) if abs(s) > 1e-6]
                if nonzero_original:
                    print(f"💰 归一化前action_sizings(非零): {nonzero_original[:10]}", flush=True)
                    print(f"💰 max_stack用于归一化: {max_stack}", flush=True)
            
            info_state_raw = normalize_info_state_action_sizings(info_state_raw, state.get_game(), max_stack)
            
            # 打印归一化后的值
            if action_sizings_start < len(info_state_raw):
                normalized_sizings = info_state_raw[action_sizings_start:action_sizings_end]
                nonzero_normalized = [(i, float(s)) for i, s in enumerate(normalized_sizings) if abs(s) > 1e-6]
                if nonzero_normalized:
                    print(f"💰 归一化后action_sizings(非零): {nonzero_normalized[:10]}", flush=True)
            
            info_state = torch.FloatTensor(info_state_raw).unsqueeze(0).to(device)
            
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
                suits = ['d', 's', 'h', 'c']  # OpenSpiel的顺序：Diamonds(0-12), Spades(13-25), Hearts(26-38), Clubs(39-51)
                ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                suit_idx = card_idx // 13
                rank_idx = card_idx % 13
                return ranks[rank_idx] + suits[suit_idx]
            
            hole_cards_str = [card_index_to_string(c) for c in hole_cards]
            board_cards_str = [card_index_to_string(c) for c in board_cards] if board_cards else []
            
            # 验证位置和手牌一致性
            position_encoding = info_state_raw[:num_players]
            actual_position_idx = np.argmax(position_encoding)
            print(f"\n🃏 Solver模式信息状态验证: player={player}, 位置编码索引={actual_position_idx}, 手牌={hole_cards_str}, 公共牌={board_cards_str}", flush=True)
            if actual_position_idx != player:
                print(f"⚠️ 警告: 位置编码索引({actual_position_idx})与player({player})不一致！", flush=True)
            else:
                print(f"✅ 位置编码和player一致", flush=True)
            
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
            
            # 提取非零的action_sizings（注意：action_sizings是连续值，不是二进制位）
            nonzero_sizings = [(i, float(s)) for i, s in enumerate(action_sizings_bits) if abs(s) > 1e-6]
            
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
            
            # 打印原始概率分布（用于调试）
            print(f"📊 模型原始概率分布（前5个动作）: {dict(zip(range(5), probs[:5]))}", flush=True)
            print(f"📊 模型输出维度: {len(probs)}, 合法动作: {legal_actions}", flush=True)
            
            # 构建概率字典（只考虑在模型输出范围内的合法动作）
            legal_probs = {}
            max_action_index = len(probs) - 1
            skipped_actions = []
            for action in legal_actions:
                if action <= max_action_index:
                    legal_probs[action] = float(probs[action])
                else:
                    skipped_actions.append(action)
            
            if skipped_actions:
                print(f"⚠️ 警告: 以下合法动作超出模型输出范围，将被忽略: {skipped_actions} (模型最大动作索引: {max_action_index})", flush=True)
            
            # 归一化
            if legal_probs:
                total_prob = sum(legal_probs.values())
                if total_prob > 0:
                    for action in legal_probs:
                        legal_probs[action] /= total_prob
                else:
                    # 如果所有概率都是0，均匀分布
                    uniform_prob = 1.0 / len(legal_probs)
                    for action in legal_probs:
                        legal_probs[action] = uniform_prob
            else:
                # 如果所有合法动作都超出模型输出范围，使用均匀分布
                print(f"⚠️ 警告: 所有合法动作都超出模型输出范围，使用均匀分布", flush=True)
                uniform_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    legal_probs[action] = uniform_prob
            
            # 打印归一化后的概率分布（用于调试）
            print(f"📊 归一化后的合法动作概率: {legal_probs}", flush=True)
            
            # 选择推荐动作（概率最大的）
            if legal_probs:
                recommended_action = max(legal_probs.items(), key=lambda x: x[1])[0]
            else:
                recommended_action = legal_actions[0] if legal_actions else None
            
            print(f"🎯 推荐动作: {recommended_action} (概率: {legal_probs.get(recommended_action, 0.0):.4f})", flush=True)
            
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
    
    # Standard Network（老模型：直接是 Network 对象）
    print(f"\n📦 使用标准 Network 模型（老模型格式）", flush=True)
    info_state_raw = state.information_state_tensor(player)
    
    # 归一化action_sizings部分（与训练时保持一致）
    # 获取max_stack（从模型或游戏配置）
    max_stack = None
    # 处理 DataParallel（老模型可能也有）
    actual_model = model
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    
    if hasattr(actual_model, 'max_stack'):
        max_stack = actual_model.max_stack
    
    # 如果模型没有max_stack，从游戏配置解析
    if max_stack is None:
        import re
        game_string = str(state.get_game())
        match = re.search(r'stack=([\d\s]+)', game_string)
        if match:
            stack_str = match.group(1).strip()
            stack_values = stack_str.split()
            if stack_values:
                try:
                    max_stack = int(stack_values[0])
                except ValueError:
                    max_stack = 2000  # 默认值
        else:
            max_stack = 2000  # 默认值
    
    # 归一化action_sizings
    info_state_raw = normalize_info_state_action_sizings(info_state_raw, state.get_game(), max_stack)
    
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
        suits = ['d', 's', 'h', 'c']  # OpenSpiel的顺序：Diamonds(0-12), Spades(13-25), Hearts(26-38), Clubs(39-51)
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suit_idx = card_idx // 13
        rank_idx = card_idx % 13
        return ranks[rank_idx] + suits[suit_idx]
    
    hole_cards_str = [card_index_to_string(c) for c in hole_cards]
    board_cards_str = [card_index_to_string(c) for c in board_cards] if board_cards else []
    
    # 验证位置和手牌一致性
    position_encoding = info_state_raw[:num_players]
    actual_position_idx = np.argmax(position_encoding)
    print(f"\n🃏 信息状态验证: player={player}, 位置编码索引={actual_position_idx}, 手牌={hole_cards_str}, 公共牌={board_cards_str}", flush=True)
    if actual_position_idx != player:
        print(f"⚠️ 警告: 位置编码索引({actual_position_idx})与player({player})不一致！", flush=True)
    else:
        print(f"✅ 位置编码和player一致", flush=True)
    
    # ⚠️ 关键修复：禁用位置编码映射！
    # 
    # 问题分析：
    # OpenSpiel的information_state_tensor(player)返回的是：
    # - 位置编码：values[player] = 1 - "我是player"
    # - 手牌：HoleCards(player) - 该player的手牌
    #
    # 如果我们映射位置编码（比如从Player 0映射到Player 2），但手牌编码保持不变：
    # - 位置编码：[0,0,1,0,0,0] - "我是Player 2"
    # - 手牌编码：Player 0的手牌（As, Ah）
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
    
    print(f"\n⚠️ 警告: 已禁用位置编码映射，直接使用OpenSpiel的信息状态", flush=True)
    print(f"   原因: 位置编码映射会导致位置和手牌不一致，影响模型推理", flush=True)
    print(f"   位置编码表示'我是player'，不应该改变", flush=True)
    print(f"   手牌编码是相对于实际player的，不应该映射", flush=True)
    
    # 不再进行位置编码映射，直接使用原始信息状态
    # 处理 DataParallel（老模型可能也有）
    with torch.no_grad():
        logits = model(info_state)  # DataParallel 会自动处理
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # 只保留合法动作的概率（只考虑在模型输出范围内的合法动作）
    legal_probs = {}
    max_action_index = len(probs) - 1
    skipped_actions = []
    for action in legal_actions:
        if action <= max_action_index:
            legal_probs[action] = float(probs[action])
        else:
            skipped_actions.append(action)
    
    if skipped_actions:
        print(f"⚠️ 警告: 以下合法动作超出模型输出范围，将被忽略: {skipped_actions} (模型最大动作索引: {max_action_index})", flush=True)
    
    # 归一化
    if legal_probs:
        total_prob = sum(legal_probs.values())
        if total_prob > 0:
            for action in legal_probs:
                legal_probs[action] /= total_prob
        else:
            # 如果所有概率都是0，均匀分布
            uniform_prob = 1.0 / len(legal_probs)
            for action in legal_probs:
                legal_probs[action] = uniform_prob
    else:
        # 如果所有合法动作都超出模型输出范围，使用均匀分布
        print(f"⚠️ 警告: 所有合法动作都超出模型输出范围，使用均匀分布", flush=True)
        uniform_prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            legal_probs[action] = uniform_prob
    
    # 选择推荐动作（概率最大的）
    if legal_probs:
        recommended_action = max(legal_probs.items(), key=lambda x: x[1])[0]
    else:
        recommended_action = legal_actions[0] if legal_actions else None
    
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
    # 检查是否有模型加载（向后兼容：检查全局MODEL或MODELS字典）
    if not MODELS and MODEL is None:
        return jsonify({
            'success': False,
            'data': None,
            'error': 'No model loaded. Please load model using /api/v1/reload_model or start server with --model_dir/--model_5p/--model_6p'
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
        
        # 调试：打印接收到的action_history和action_sizings
        print(f"📋 接收到的请求数据: player_id={player_id}, action_history={action_history}, action_sizings={action_sizings}", flush=True)
        
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
        
        # 根据玩家数量选择对应的模型
        model = MODELS.get(num_players, None)
        if model is None:
            # 如果没有对应玩家数量的模型，尝试使用全局MODEL（向后兼容）
            if MODEL is None:
                return jsonify({
                    'success': False,
                    'data': None,
                    'error': f'No model loaded for {num_players} players. Please load model using /api/v1/reload_model or start server with --model_dir'
                }), 500
            model = MODEL
            print(f"⚠️ 警告: 没有找到{num_players}人场的模型，使用默认模型", flush=True)
        else:
            print(f"✅ 使用{num_players}人场模型", flush=True)
        
        # 获取推荐动作（传入dealer_pos用于位置编码映射）
        print(f"\n🎯 调用get_recommended_action: player_id={player_id}, dealer_pos={dealer_pos}, num_players={num_players}", flush=True)
        recommended_action, action_probs, legal_actions = get_recommended_action(
            state, model, DEVICE, dealer_pos=dealer_pos
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
    """重新加载模型（支持动态切换模型，支持替换特定场次的模型）
    
    请求格式:
    {
        "model_dir": "models/deepcfr_stable_run",
        "device": "cpu",  // 可选，默认使用当前设备
        "num_players": 5  // 可选，明确指定场次（5或6）。如果不指定，从config.json读取
    }
    
    示例：
    - 替换5人场模型: {"model_dir": "models/5p_model", "num_players": 5}
    - 替换6人场模型: {"model_dir": "models/6p_model", "num_players": 6}
    - 自动检测场次: {"model_dir": "models/some_model"}  // 从config.json读取num_players
    """
    global GAME, MODEL, CONFIG, MODELS, CONFIGS, GAMES, MODEL_DIRS
    
    try:
        data = request.get_json() or {}
        model_dir = data.get('model_dir', MODEL_DIR)
        device = data.get('device', DEVICE)
        num_players = data.get('num_players', None)  # 可选：明确指定场次
        
        if model_dir is None:
            return jsonify({
                'success': False,
                'error': 'model_dir not provided and no default model loaded'
            }), 400
        
        # 加载新模型（如果指定了num_players，会明确替换对应场次的模型）
        # 先保存旧的MODEL_DIRS，用于判断是否新增了模型
        old_model_dirs = dict(MODEL_DIRS)
        old_num_players = set(MODEL_DIRS.keys())
        
        # 如果指定了num_players，记录它
        specified_num_players = num_players
        
        load_model(model_dir, device=device, num_players=num_players)
        
        # 获取实际加载的num_players
        actual_num_players = None
        
        # 方法1: 如果指定了num_players，直接使用它（因为load_model会按此存储）
        if specified_num_players is not None and specified_num_players in MODEL_DIRS:
            actual_num_players = specified_num_players
        else:
            # 方法2: 从MODEL_DIRS中查找匹配的路径（考虑相对路径和绝对路径）
            import os
            abs_model_dir = os.path.abspath(model_dir)
            for np, dir_path in MODEL_DIRS.items():
                abs_dir_path = os.path.abspath(dir_path)
                if dir_path == model_dir or abs_dir_path == abs_model_dir or dir_path == abs_model_dir or abs_dir_path == model_dir:
                    actual_num_players = np
                    break
            
            # 方法3: 如果没找到，查找新增的模型（刚加载的）
            if actual_num_players is None:
                new_num_players = set(MODEL_DIRS.keys()) - old_num_players
                if new_num_players:
                    actual_num_players = list(new_num_players)[0]
            
            # 方法4: 如果还是没找到，从CONFIGS中查找最新的（刚加载的模型）
            if actual_num_players is None and CONFIGS:
                # 找到最近加载的模型对应的num_players（取最大的key，通常是最后加载的）
                actual_num_players = max(CONFIGS.keys()) if CONFIGS else None
        
        return jsonify({
            'success': True,
            'message': f'Model reloaded from {model_dir}',
            'model_dir': model_dir,
            'device': device,
            'num_players': actual_num_players,
            'loaded_models': {str(np): MODEL_DIRS.get(np, 'N/A') for np in sorted(MODELS.keys())}
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to reload model: {str(e)}'
        }), 500


@app.route('/api/v1/model_info', methods=['GET'])
def model_info():
    """查看当前使用的模型信息
    
    返回所有已加载模型的详细信息
    """
    global MODELS, CONFIGS, MODEL_DIRS, GAMES
    
    try:
        num_players = request.args.get('num_players', type=int)
        
        if num_players is not None:
            # 返回指定场次的模型信息
            if num_players not in MODELS:
                return jsonify({
                    'success': False,
                    'error': f'No model loaded for {num_players} players'
                }), 404
            
            model = MODELS[num_players]
            config = CONFIGS.get(num_players, {})
            model_dir = MODEL_DIRS.get(num_players, 'N/A')
            game = GAMES.get(num_players)
            
            # 获取模型类型信息
            model_type = 'unknown'
            feature_info = {}
            
            # 检查是否是 DeepCFRSolver（新模型）或直接是 Network（老模型）
            if hasattr(model, '_policy_network'):
                # 新模型：DeepCFRSolver 包装
                policy_net = model._policy_network
                # 处理 DataParallel
                if isinstance(policy_net, nn.DataParallel):
                    policy_net = policy_net.module
                
                if isinstance(policy_net, SimpleFeatureMLP):
                    model_type = 'SimpleFeatureMLP'
                    feature_info = {
                        'manual_feature_size': policy_net.manual_feature_size,
                        'raw_input_size': policy_net.raw_input_size,
                        'input_size': policy_net.raw_input_size + policy_net.manual_feature_size,
                        'description': '简单特征模型：原始信息状态 + 1维手牌强度特征'
                    }
                elif hasattr(policy_net, 'transformed_size'):
                    model_type = 'FeatureTransformMLP'
                    feature_info = {
                        'transformed_size': getattr(policy_net, 'transformed_size', 'N/A'),
                        'description': '特征转换模型：使用特征转换层'
                    }
                else:
                    model_type = 'StandardMLP'
                    feature_info = {
                        'description': '标准MLP模型：无自定义特征（DeepCFRSolver包装）'
                    }
            elif isinstance(model, nn.Module):
                # 老模型：直接是 Network 对象
                policy_net = model
                # 处理 DataParallel
                if isinstance(policy_net, nn.DataParallel):
                    policy_net = policy_net.module
                
                # 检查是否是 SimpleFeatureMLP（虽然老模型通常不是，但兼容性检查）
                if isinstance(policy_net, SimpleFeatureMLP):
                    model_type = 'SimpleFeatureMLP'
                    feature_info = {
                        'manual_feature_size': policy_net.manual_feature_size,
                        'raw_input_size': policy_net.raw_input_size,
                        'input_size': policy_net.raw_input_size + policy_net.manual_feature_size,
                        'description': '简单特征模型：原始信息状态 + 1维手牌强度特征'
                    }
                elif hasattr(policy_net, 'transformed_size'):
                    model_type = 'FeatureTransformMLP'
                    feature_info = {
                        'transformed_size': getattr(policy_net, 'transformed_size', 'N/A'),
                        'description': '特征转换模型：使用特征转换层'
                    }
                else:
                    # 标准 MLP（老模型）
                    model_type = 'StandardMLP'
                    feature_info = {
                        'description': '标准MLP模型：无自定义特征（老模型格式）'
                    }
            
            return jsonify({
                'success': True,
                'num_players': num_players,
                'model_dir': model_dir,
                'model_type': model_type,
                'feature_info': feature_info,
                'config': {
                    'policy_layers': config.get('policy_layers', []),
                    'advantage_layers': config.get('advantage_layers', []),
                    'betting_abstraction': config.get('betting_abstraction', 'N/A'),
                    'use_simple_feature': config.get('use_simple_feature', False),
                    'use_feature_transform': config.get('use_feature_transform', False),
                    'save_prefix': config.get('save_prefix', 'N/A'),
                    'blinds': config.get('blinds', 'N/A'),
                    'stack_size': config.get('stack_size', 'N/A'),
                },
                'device': str(model._device) if hasattr(model, '_device') else 'N/A'
            })
        else:
            # 返回所有已加载模型的信息
            all_models = {}
            for np in sorted(MODELS.keys()):
                model = MODELS[np]
                config = CONFIGS.get(np, {})
                model_dir = MODEL_DIRS.get(np, 'N/A')
                
                # 获取模型类型
                model_type = 'unknown'
                feature_info = {}
                
                # 检查是否是 DeepCFRSolver（新模型）或直接是 Network（老模型）
                if hasattr(model, '_policy_network'):
                    # 新模型：DeepCFRSolver 包装
                    policy_net = model._policy_network
                    if isinstance(policy_net, nn.DataParallel):
                        policy_net = policy_net.module
                    
                    if isinstance(policy_net, SimpleFeatureMLP):
                        model_type = 'SimpleFeatureMLP'
                        feature_info = {
                            'manual_feature_size': policy_net.manual_feature_size,
                            'raw_input_size': policy_net.raw_input_size,
                            'input_size': policy_net.raw_input_size + policy_net.manual_feature_size,
                            'description': '简单特征模型：原始信息状态 + 1维手牌强度特征'
                        }
                    elif hasattr(policy_net, 'transformed_size'):
                        model_type = 'FeatureTransformMLP'
                        feature_info = {
                            'transformed_size': getattr(policy_net, 'transformed_size', 'N/A'),
                            'description': '特征转换模型：使用特征转换层'
                        }
                    else:
                        model_type = 'StandardMLP'
                        feature_info = {
                            'description': '标准MLP模型：无自定义特征（DeepCFRSolver包装）'
                        }
                elif isinstance(model, nn.Module):
                    # 老模型：直接是 Network 对象
                    policy_net = model
                    if isinstance(policy_net, nn.DataParallel):
                        policy_net = policy_net.module
                    
                    # 检查是否是 SimpleFeatureMLP（虽然老模型通常不是，但兼容性检查）
                    if isinstance(policy_net, SimpleFeatureMLP):
                        model_type = 'SimpleFeatureMLP'
                        feature_info = {
                            'manual_feature_size': policy_net.manual_feature_size,
                            'raw_input_size': policy_net.raw_input_size,
                            'input_size': policy_net.raw_input_size + policy_net.manual_feature_size,
                            'description': '简单特征模型：原始信息状态 + 1维手牌强度特征'
                        }
                    elif hasattr(policy_net, 'transformed_size'):
                        model_type = 'FeatureTransformMLP'
                        feature_info = {
                            'transformed_size': getattr(policy_net, 'transformed_size', 'N/A'),
                            'description': '特征转换模型：使用特征转换层'
                        }
                    else:
                        # 标准 MLP（老模型）
                        model_type = 'StandardMLP'
                        feature_info = {
                            'description': '标准MLP模型：无自定义特征（老模型格式）'
                        }
                
                all_models[str(np)] = {
                    'model_dir': model_dir,
                    'model_type': model_type,
                    'feature_info': feature_info,
                    'config': {
                        'policy_layers': config.get('policy_layers', []),
                        'advantage_layers': config.get('advantage_layers', []),
                        'betting_abstraction': config.get('betting_abstraction', 'N/A'),
                        'use_simple_feature': config.get('use_simple_feature', False),
                        'use_feature_transform': config.get('use_feature_transform', False),
                        'save_prefix': config.get('save_prefix', 'N/A'),
                    }
                }
            
            return jsonify({
                'success': True,
                'loaded_models': all_models,
                'total_models': len(all_models)
            })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get model info: {str(e)}'
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
    parser.add_argument('--model_dir', type=str, required=False,
                        help='Path to model directory (containing config.json and model files). Can specify multiple times for different player counts.')
    parser.add_argument('--model_5p', type=str, default=None,
                        help='Path to 5-player model directory')
    parser.add_argument('--model_6p', type=str, default=None,
                        help='Path to 6-player model directory')
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
    
    # 加载模型（支持多模型）
    models_loaded = False
    
    # 加载5人场模型
    if args.model_5p:
        try:
            print(f"\n📦 加载5人场模型: {args.model_5p}")
            load_model(args.model_5p, device=DEVICE, num_players=5)
            models_loaded = True
            print(f"✅ 5人场模型加载成功")
        except Exception as e:
            print(f"❌ 加载5人场模型失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 加载6人场模型
    if args.model_6p:
        try:
            print(f"\n📦 加载6人场模型: {args.model_6p}")
            load_model(args.model_6p, device=DEVICE, num_players=6)
            models_loaded = True
            print(f"✅ 6人场模型加载成功")
        except Exception as e:
            print(f"❌ 加载6人场模型失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 向后兼容：如果指定了--model_dir，加载它（自动检测玩家数量）
    if args.model_dir:
        try:
            print(f"\n📦 加载模型（自动检测玩家数量）: {args.model_dir}")
            load_model(args.model_dir, device=DEVICE)
            models_loaded = True
            print(f"✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
    
    if not models_loaded:
        print(f"\n⚠️ 警告: 没有加载任何模型！")
        print(f"   请使用 --model_dir, --model_5p, 或 --model_6p 指定模型目录")
        print(f"   或者启动后使用 /api/v1/reload_model 接口加载模型")
    
    # 打印已加载的模型
    if MODELS:
        print(f"\n📊 已加载的模型:")
        for np, model_dir in MODEL_DIRS.items():
            print(f"   {np}人场: {model_dir}")
    
    # 启动服务器
    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"Device: {DEVICE}")
    print(f"\nAPI endpoints:")
    print(f"  GET  /api/v1/health - Health check")
    print(f"  POST /api/v1/recommend_action - Get recommended action")
    print(f"  POST /api/v1/reload_model - Reload model (dynamic model switching)")
    print(f"  GET  /api/v1/model_info - Get current model information (supports ?num_players=X)")
    print(f"  GET  /api/v1/action_mapping - Get action mapping")
    print()
    
    # 确保print输出不被缓冲
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
