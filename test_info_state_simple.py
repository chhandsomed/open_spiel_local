#!/usr/bin/env python3
"""简单测试：打印信息状态"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import numpy as np
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401


def card_index_to_string(card_idx, num_suits=4, num_ranks=13):
    """将牌索引转换为字符串表示"""
    suit_names = ['s', 'h', 'd', 'c']
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit = card_idx % num_suits
    rank = card_idx // num_suits
    return rank_names[rank] + suit_names[suit]


def main():
    num_players = 6
    blinds_str = " ".join(["100"] * num_players)
    stacks_str = " ".join(["2000"] * num_players)
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
    state = game.new_initial_state()
    
    print("=" * 70)
    print("信息状态结构分析")
    print("=" * 70)
    print(f"\n游戏: {game.get_type().short_name}")
    print(f"玩家数量: {num_players}")
    print(f"最大游戏长度: {game.max_game_length()}")
    print(f"信息状态大小: {game.information_state_tensor_shape()[0]}")
    
    # 模拟几步
    steps = 0
    while not state.is_terminal() and steps < 15:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
        else:
            player = state.current_player()
            info_state = np.array(state.information_state_tensor(player))
            
            print(f"\n{'='*70}")
            print(f"步骤 {steps}: 玩家 {player} 行动")
            print(f"{'='*70}")
            print(f"信息状态大小: {len(info_state)}")
            print(f"非零元素: {np.count_nonzero(info_state)}")
            
            # 解析各部分
            offset = 0
            
            # 1. 玩家位置
            player_pos = np.argmax(info_state[offset:offset+num_players])
            print(f"\n[1] 玩家位置 (bits {offset}-{offset+num_players-1}):")
            print(f"  玩家: {player_pos}")
            offset += num_players
            
            # 2. 手牌
            num_cards = 52
            hole_cards_bits = info_state[offset:offset+num_cards]
            hole_cards = [i for i, bit in enumerate(hole_cards_bits) if bit > 0.5]
            print(f"\n[2] 手牌 (bits {offset}-{offset+num_cards-1}):")
            if hole_cards:
                print(f"  手牌: {[card_index_to_string(c) for c in hole_cards]}")
            else:
                print(f"  手牌: (未发牌)")
            offset += num_cards
            
            # 3. 公共牌
            board_cards_bits = info_state[offset:offset+num_cards]
            board_cards = [i for i, bit in enumerate(board_cards_bits) if bit > 0.5]
            print(f"\n[3] 公共牌 (bits {offset}-{offset+num_cards-1}):")
            if board_cards:
                print(f"  公共牌: {[card_index_to_string(c) for c in board_cards]}")
            else:
                print(f"  公共牌: (未发牌)")
            offset += num_cards
            
            # 4. 动作序列
            max_len = game.max_game_length()
            action_seq_bits = info_state[offset:offset+max_len*2]
            print(f"\n[4] 动作序列 (bits {offset}-{offset+max_len*2-1}, {max_len}个动作):")
            action_count = 0
            for i in range(min(20, max_len)):  # 只显示前20个
                bit0 = action_seq_bits[2*i]
                bit1 = action_seq_bits[2*i+1]
                if bit0 < 0.5 and bit1 < 0.5:
                    act = 'f'
                elif bit0 > 0.5 and bit1 < 0.5:
                    act = 'c'
                elif bit0 < 0.5 and bit1 > 0.5:
                    act = 'p'
                elif bit0 > 0.5 and bit1 > 0.5:
                    act = 'a'
                else:
                    act = '?'
                
                if act != 'f' or i < 5:
                    action_count += 1
                    act_name = {'f': 'Fold', 'c': 'Call', 'p': 'Raise', 'a': 'All-in'}.get(act, act)
                    print(f"  位置 {i}: {act_name} (bits: [{bit0:.0f}, {bit1:.0f}])")
            print(f"  已记录动作数: {action_count}")
            offset += max_len * 2
            
            # 5. 动作大小
            action_sizings = info_state[offset:offset+max_len]
            print(f"\n[5] 动作大小 (bits {offset}-{offset+max_len-1}):")
            nonzero_sizings = [(i, s) for i, s in enumerate(action_sizings) if s > 0.5]
            if nonzero_sizings:
                for idx, size in nonzero_sizings[:10]:
                    print(f"  位置 {idx}: {size:.0f}")
            else:
                print(f"  (全为0)")
            
            # 游戏状态
            try:
                state_struct = state.to_struct()
                print(f"\n[游戏状态]")
                print(f"  底池: {getattr(state_struct, 'pot_size', 'N/A')}")
                print(f"  下注历史: {getattr(state_struct, 'betting_history', 'N/A')}")
            except:
                pass
            
            # 选择动作
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)
            state = state.child(action)
            print(f"\n选择动作: {action} (合法动作: {legal_actions})")
        
        steps += 1
    
    print(f"\n{'='*70}")
    print("关于信息状态自定义")
    print(f"{'='*70}")
    print("""
信息状态格式（固定，在 C++ 中定义）：
1. 玩家位置：num_players bits (one-hot)
2. 手牌：52 bits (每张牌一个bit)
3. 公共牌：52 bits (每张牌一个bit)
4. 动作序列：max_game_length * 2 bits
   - 'f'(fold) = 00
   - 'c'(call) = 10
   - 'p'(raise) = 01
   - 'a'(all-in) = 11
5. 动作大小：max_game_length integers (下注金额)

⚠️ 不能直接在 Python 中自定义信息状态格式。
如需自定义，需要修改 C++ 代码并重新编译。
    """)


if __name__ == "__main__":
    main()


