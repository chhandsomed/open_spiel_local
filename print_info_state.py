#!/usr/bin/env python3
"""打印信息状态 (Information State) 的详细内容

展示德州扑克中信息状态的各个组成部分
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import numpy as np
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401


def card_index_to_string(card_idx, num_suits=4, num_ranks=13):
    """将牌索引转换为字符串表示"""
    suit_names = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    suit = card_idx % num_suits
    rank = card_idx // num_suits
    
    return rank_names[rank] + suit_names[suit]


def parse_info_state(info_state, num_players, max_game_length, num_suits=4, num_ranks=13):
    """解析信息状态向量
    
    信息状态格式（从 universal_poker.cc）：
    1. 玩家位置：num_players bits
    2. 手牌：52 bits (num_suits * num_ranks)
    3. 公共牌：52 bits
    4. 动作序列：max_game_length * 2 bits (fold/raise/call/all-in)
    5. 动作序列大小：max_game_length integers
    """
    offset = 0
    result = {}
    
    # 1. 玩家位置
    player_position = np.argmax(info_state[offset:offset+num_players])
    result['player_position'] = player_position
    result['player_one_hot'] = info_state[offset:offset+num_players].tolist()
    offset += num_players
    
    # 2. 手牌
    num_cards = num_suits * num_ranks
    hole_cards_bits = info_state[offset:offset+num_cards]
    hole_cards = [i for i, bit in enumerate(hole_cards_bits) if bit > 0.5]
    result['hole_cards'] = [card_index_to_string(c, num_suits, num_ranks) for c in hole_cards]
    result['hole_cards_bits'] = hole_cards_bits.tolist()
    offset += num_cards
    
    # 3. 公共牌
    board_cards_bits = info_state[offset:offset+num_cards]
    board_cards = [i for i, bit in enumerate(board_cards_bits) if bit > 0.5]
    result['board_cards'] = [card_index_to_string(c, num_suits, num_ranks) for c in board_cards]
    result['board_cards_bits'] = board_cards_bits.tolist()
    offset += num_cards
    
    # 4. 动作序列（每2个bit表示一个动作）
    # 编码：'f'(fold)=00, 'c'(call)=10, 'p'(raise)=01, 'a'(all-in)=11, 'd'(deal)=00
    action_sequence = []
    action_sequence_bits = info_state[offset:offset+max_game_length*2]
    
    for i in range(max_game_length):
        bit0 = action_sequence_bits[2*i]
        bit1 = action_sequence_bits[2*i+1]
        
        if bit0 < 0.5 and bit1 < 0.5:
            action = 'f'  # fold or deal
        elif bit0 > 0.5 and bit1 < 0.5:
            action = 'c'  # call
        elif bit0 < 0.5 and bit1 > 0.5:
            action = 'p'  # raise
        elif bit0 > 0.5 and bit1 > 0.5:
            action = 'a'  # all-in
        else:
            action = '?'  # unknown
        
        if action != 'f' or i < 10:  # 只显示非空或前10个
            action_sequence.append({
                'index': i,
                'action': action,
                'bits': [float(bit0), float(bit1)]
            })
    
    result['action_sequence'] = action_sequence
    result['action_sequence_bits'] = action_sequence_bits.tolist()
    offset += max_game_length * 2
    
    # 5. 动作序列大小（下注金额）
    action_sizings = info_state[offset:offset+max_game_length]
    result['action_sizings'] = action_sizings.tolist()
    result['action_sizings_nonzero'] = [
        {'index': i, 'size': float(s)} 
        for i, s in enumerate(action_sizings) 
        if s > 0.5
    ]
    offset += max_game_length
    
    # 验证
    if offset != len(info_state):
        result['warning'] = f"Offset ({offset}) != info_state length ({len(info_state)})"
    
    return result


def print_info_state_detailed(state, player, num_players=6, max_game_length=None):
    """打印信息状态的详细信息"""
    print("\n" + "=" * 70)
    print(f"信息状态详情 (玩家 {player})")
    print("=" * 70)
    
    # 获取信息状态向量
    info_state = np.array(state.information_state_tensor(player))
    print(f"\n信息状态向量大小: {len(info_state)}")
    print(f"非零元素数量: {np.count_nonzero(info_state)}")
    
    # 获取最大游戏长度
    if max_game_length is None:
        try:
            # 尝试从状态获取游戏对象
            game_obj = state.get_game() if hasattr(state, 'get_game') else None
            if game_obj:
                max_game_length = game_obj.max_game_length()
            else:
                # 估算：总长度 - 前面已知部分
                max_game_length = (len(info_state) - num_players - 52 - 52) // 3
        except:
            # 估算：总长度 - 前面已知部分
            max_game_length = (len(info_state) - num_players - 52 - 52) // 3
    
    # 解析信息状态
    parsed = parse_info_state(info_state, num_players, max_game_length)
    
    # 打印各部分
    print(f"\n[1] 玩家位置 ({num_players} bits):")
    print(f"  当前玩家: {parsed['player_position']}")
    print(f"  One-hot 编码: {parsed['player_one_hot']}")
    
    print(f"\n[2] 手牌 (52 bits):")
    if parsed['hole_cards']:
        print(f"  手牌: {', '.join(parsed['hole_cards'])}")
        print(f"  手牌数量: {len(parsed['hole_cards'])}")
    else:
        print(f"  手牌: (未发牌)")
    print(f"  手牌 bits 中1的数量: {int(sum(parsed['hole_cards_bits']))}")
    
    print(f"\n[3] 公共牌 (52 bits):")
    if parsed['board_cards']:
        print(f"  公共牌: {', '.join(parsed['board_cards'])}")
        print(f"  公共牌数量: {len(parsed['board_cards'])}")
    else:
        print(f"  公共牌: (未发牌)")
    print(f"  公共牌 bits 中1的数量: {int(sum(parsed['board_cards_bits']))}")
    
    print(f"\n[4] 动作序列 ({max_game_length} * 2 bits):")
    if parsed['action_sequence']:
        print(f"  已记录的动作数: {len([a for a in parsed['action_sequence'] if a['action'] != 'f'])}")
        print(f"  动作序列:")
        for act in parsed['action_sequence'][:20]:  # 只显示前20个
            action_name = {
                'f': 'Fold/Deal',
                'c': 'Call',
                'p': 'Raise',
                'a': 'All-in',
                '?': 'Unknown'
            }.get(act['action'], act['action'])
            print(f"    位置 {act['index']}: {action_name} (bits: {act['bits']})")
    else:
        print(f"  动作序列: (空)")
    
    print(f"\n[5] 动作大小 ({max_game_length} integers):")
    if parsed['action_sizings_nonzero']:
        print(f"  非零下注金额:")
        for sizing in parsed['action_sizings_nonzero'][:20]:  # 只显示前20个
            print(f"    位置 {sizing['index']}: {sizing['size']:.0f}")
    else:
        print(f"  动作大小: (全为0，表示未下注或已过牌)")
    
    # 获取游戏状态信息
    try:
        state_struct = state.to_struct()
        print(f"\n[游戏状态信息]")
        print(f"  底池: {getattr(state_struct, 'pot_size', 'N/A')}")
        print(f"  玩家投入: {getattr(state_struct, 'player_contributions', 'N/A')}")
        print(f"  下注历史: {getattr(state_struct, 'betting_history', 'N/A')}")
        if hasattr(state_struct, 'player_hands'):
            print(f"  所有玩家手牌: {state_struct.player_hands}")
    except:
        pass
    
    if 'warning' in parsed:
        print(f"\n⚠️ 警告: {parsed['warning']}")
    
    print("=" * 70)
    
    return parsed


def simulate_game_and_print_states(num_players=6, num_steps=10):
    """模拟游戏并打印不同阶段的信息状态"""
    print("=" * 70)
    print("德州扑克信息状态打印工具")
    print("=" * 70)
    
    # 创建游戏
    blinds_str = " ".join(["100"] * num_players)
    stacks_str = " ".join(["2000"] * num_players)
    
    if num_players == 2:
        first_player_str = " ".join(["2"] + ["1"] * 3)
    else:
        first_player_str = " ".join(["3"] + ["1"] * 3)
    
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
        f"numRanks=13"
        f")"
    )
    
    game = pyspiel.load_game(game_string)
    print(f"\n游戏: {game.get_type().short_name}")
    print(f"玩家数量: {num_players}")
    print(f"信息状态大小: {game.information_state_tensor_shape()[0]}")
    
    # 模拟游戏
    state = game.new_initial_state()
    step = 0
    
    print("\n" + "=" * 70)
    print("开始模拟游戏...")
    print("=" * 70)
    
    while not state.is_terminal() and step < num_steps:
        if state.is_chance_node():
            # 机会节点：发牌
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
            print(f"\n[步骤 {step}] 机会节点：发牌 (动作 {action})")
        else:
            # 玩家节点
            player = state.current_player()
            legal_actions = state.legal_actions()
            
            print(f"\n[步骤 {step}] 玩家 {player} 行动")
            print(f"  合法动作: {legal_actions}")
            
            # 打印当前玩家的信息状态
            if player < num_players:
                max_game_length = game.max_game_length()
                parsed = print_info_state_detailed(state, player, num_players, max_game_length)
            
            # 随机选择一个动作
            action = np.random.choice(legal_actions)
            print(f"  选择动作: {action}")
            state = state.child(action)
        
        step += 1
    
    # 最终状态
    if state.is_terminal():
        print("\n" + "=" * 70)
        print("游戏结束")
        print("=" * 70)
        returns = state.returns()
        print(f"最终收益: {returns}")
        
        # 打印所有玩家的最终信息状态
        max_game_length = game.max_game_length()
        for p in range(num_players):
            print(f"\n玩家 {p} 的最终信息状态:")
            print_info_state_detailed(state, p, num_players, max_game_length)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="打印信息状态详情")
    parser.add_argument("--num_players", type=int, default=6, help="玩家数量")
    parser.add_argument("--num_steps", type=int, default=15, help="模拟步数")
    
    args = parser.parse_args()
    
    simulate_game_and_print_states(args.num_players, args.num_steps)
    
    print("\n" + "=" * 70)
    print("关于信息状态自定义的说明")
    print("=" * 70)
    print("""
信息状态是在 C++ 层面（universal_poker.cc）生成的，格式是固定的：
1. 玩家位置：num_players bits (one-hot)
2. 手牌：52 bits (每张牌一个bit，1表示有这张牌)
3. 公共牌：52 bits (每张牌一个bit，1表示是公共牌)
4. 动作序列：max_game_length * 2 bits (编码动作类型)
5. 动作大小：max_game_length integers (下注金额)

⚠️ 注意：信息状态的格式不能直接在 Python 中自定义。
如果需要自定义，需要：
1. 修改 C++ 代码 (universal_poker.cc 中的 InformationStateTensor 方法)
2. 重新编译 OpenSpiel

或者，可以使用 Observer 模式来创建自定义的观察表示。
    """)


if __name__ == "__main__":
    main()

