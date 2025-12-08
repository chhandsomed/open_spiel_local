import re
from collections import defaultdict

log_file = '/home/ch/work/texas_hold/open_spiel_local/play_interactive_7300_100games.log'

def analyze_log(filepath):
    total_games = 0
    player_profits = defaultdict(float)
    player_wins = defaultdict(int)
    action_counts = defaultdict(lambda: defaultdict(int))
    game_end_rounds = defaultdict(int)
    current_round = "Preflop"
    
    # Pre-compile regex for performance
    profit_pattern = re.compile(r'\s*(你|玩家 \d+)(?: \(模型\))? (?:收益|Win): ([-\d\.]+)')
    # Adjusted regex to match the log format: "玩家 2 (模型) 选择了: 弃牌 (Fold)" or "你 (自动模式) 选择了: ..."
    action_pattern = re.compile(r'(玩家 \d+|你) (?:.*?选择了|Win): (.*?)(?: \(|$)') 
    round_pattern = re.compile(r'当前轮次: (\w+)')
    game_start_pattern = re.compile(r'开始第 (\d+)/100 局')

    with open(filepath, 'r') as f:
        lines = f.readlines()

    game_active = False
    
    for line in lines:
        line = line.strip()
        
        if "开始第" in line and "/100 局" in line:
            game_active = True
            total_games += 1
            current_round = "Preflop" # Reset round
            
        if not game_active:
            continue

        # Track rounds
        match_round = round_pattern.search(line)
        if match_round:
            current_round = match_round.group(1)

        # Track game end rounds (heuristic based on "最终结果" appearing)
        if "最终结果:" in line:
            game_end_rounds[current_round] += 1

        # Track profits
        # Format example: "  你的收益: -50.00" or "  玩家 1 收益: 150.00"
        if "收益:" in line:
            parts = line.split("收益:")
            if len(parts) == 2:
                player_part = parts[0].strip()
                # Clean the string before converting to float
                profit_str = parts[1].strip().replace(')', '').replace('(', '')
                try:
                    profit = float(profit_str)
                except ValueError:
                     # Fallback regex extraction if simple strip fails
                    match_val = re.search(r'([-\d\.]+)', parts[1])
                    if match_val:
                        profit = float(match_val.group(1))
                    else:
                        print(f"Warning: Could not parse profit from line: {line}")
                        continue
                
                # Normalize player names
                if "你" in player_part:
                    player_name = "Player 0 (You)"
                else:
                    # Extract number
                    match_p = re.search(r'玩家 (\d+)', player_part)
                    if match_p:
                        player_name = f"Player {match_p.group(1)}"
                    else:
                        continue # Should not happen based on log format
                
                player_profits[player_name] += profit
                if profit > 0:
                    player_wins[player_name] += 1

        # Track actions
        # Format example: "玩家 2 (模型) 选择了: 弃牌 (Fold)"
        # Format example: "你 (自动模式) 选择了: 半池加注 (Half-pot)"
        if "选择了:" in line:
            parts = line.split("选择了:")
            if len(parts) == 2:
                player_info = parts[0].strip()
                action_full = parts[1].strip()
                
                # Extract action name (e.g., "弃牌", "跟注/过牌")
                # Usually it's "ActionName (EnglishName)"
                action_name = action_full.split('(')[0].strip()
                
                if "弃牌" in action_name: action_type = "Fold"
                elif "跟注" in action_name or "过牌" in action_name: action_type = "Check/Call"
                elif "加注" in action_name or "全押" in action_name: action_type = "Raise/All-in"
                else: action_type = "Other"

                if "你" in player_info:
                    p_name = "Player 0 (You)"
                else:
                    match_p = re.search(r'玩家 (\d+)', player_info)
                    if match_p:
                        p_name = f"Player {match_p.group(1)}"
                    else:
                        continue
                
                action_counts[p_name][action_type] += 1

    print(f"Analysis of {total_games} games:")
    print("-" * 30)
    print(f"{'Player':<15} {'Total Profit':<15} {'Games Won (>0)':<15}")
    print("-" * 30)
    
    sorted_players = sorted(player_profits.keys())
    for p in sorted_players:
        print(f"{p:<15} {player_profits[p]:<15.2f} {player_wins[p]:<15}")

    print("\nAction Distribution (Count):")
    print("-" * 50)
    print(f"{'Player':<15} {'Fold':<10} {'Check/Call':<12} {'Raise/All-in':<12}")
    print("-" * 50)
    for p in sorted_players:
        counts = action_counts[p]
        print(f"{p:<15} {counts['Fold']:<10} {counts['Check/Call']:<12} {counts['Raise/All-in']:<12}")

    print("\nGame End Rounds Distribution:")
    for r, c in game_end_rounds.items():
        print(f"{r}: {c}")

if __name__ == "__main__":
    analyze_log(log_file)
