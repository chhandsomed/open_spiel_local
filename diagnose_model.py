import torch
import pyspiel
import numpy as np
import argparse
from play_interactive import load_model, get_model_action

def test_hand_sensitivity():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/deepcfr_2p_v1/deepcfr_2p_norm_fix", help="模型目录")
    parser.add_argument("--num_players", type=int, default=2, help="玩家数量")
    args = parser.parse_args()

    print("="*60)
    print("🔍 模型手牌敏感度测试 (Sanity Check)")
    print("="*60)
    
    # 1. 加载模型
    model_dir = args.model_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"正在加载模型: {model_dir} ...")
    try:
        # 注意：load_model 内部会读取 config.json，如果有的话
        game, model = load_model(model_dir, device=device)
        if model is None:
            print("❌ 模型加载失败")
            return
    except Exception as e:
        print(f"❌ 加载出错: {e}")
        return

    if game.num_players() != args.num_players:
        print(f"⚠️ 警告: 游戏配置玩家数 ({game.num_players()}) 与 参数 ({args.num_players}) 不一致")

    # 2. 构造测试环境
    print(f"\n正在进行统计测试 (随机生成 20 个 Preflop 状态)...")
    
    # 用于存储不同手牌类别的平均动作概率
    # 简化分类: 强牌(AA/KK/AK), 中牌, 弱牌(72o)
    hand_stats = {
        "Strong": {"count": 0, "probs": np.zeros(4)}, # 假设4个动作
        "Weak":   {"count": 0, "probs": np.zeros(4)},
        "Other":  {"count": 0, "probs": np.zeros(4)}
    }
    
    # 简单的手牌评估
    def get_hand_category(cards):
        # cards: ['Ah', 'Kd']
        ranks = [c[0] for c in cards]
        suits = [c[1] for c in cards]
        
        high_cards = {'A', 'K', 'Q', 'J'}
        # Strong: Pair of high cards, or AK/AQ/KQ/AJ...
        if ranks[0] == ranks[1] and ranks[0] in high_cards: return "Strong" # AA, KK, QQ, JJ
        if ranks[0] in high_cards and ranks[1] in high_cards: return "Strong" # AK, AQ...
        
        # Weak: Low unsuited, e.g. 72o, 83o
        low_cards = {'2', '3', '4', '5', '6', '7'}
        if ranks[0] in low_cards and ranks[1] in low_cards and ranks[0] != ranks[1] and suits[0] != suits[1]:
            return "Weak"
            
        return "Other"

    samples = 0
    target_samples = 20
    
    while samples < target_samples:
        state = game.new_initial_state()
        
        # 走到发牌结束
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], p=[p for _, p in outcomes])
            state.apply_action(action)
        
        player = state.current_player()
        
        # 获取手牌 (通过字符串解析，虽然丑但有效)
        state_str = str(state)
        # 示例: P0 Cards: 7s8h ...
        # 我们需要解析当前玩家的手牌
        import re
        match = re.search(f"P{player} Cards: ([2-9TJQKA][shdc][2-9TJQKA][shdc])", state_str)
        hand_str = "Unknown"
        cards = []
        if match:
            hand_raw = match.group(1) # e.g. 7s8h
            hand_str = f"{hand_raw[:2]} {hand_raw[2:]}"
            cards = [hand_raw[:2], hand_raw[2:]]
        
        category = "Other"
        if cards:
            category = get_hand_category(cards)

        # 获取模型动作概率
        _, probs = get_model_action(state, model, device, player)
        
        # 记录概率
        # 假设动作空间大小为 4 (FCPA)
        prob_vec = np.zeros(4)
        for a, p in probs.items():
            if a < 4: prob_vec[a] = p
            
        hand_stats[category]["count"] += 1
        hand_stats[category]["probs"] += prob_vec
        
        # 打印个例
        prob_str = ", ".join([f"{a}:{p:.2f}" for a, p in probs.items()])
        print(f"[{category:6}] 手牌: {hand_str} -> {prob_str}")
        
        samples += 1

    print("\n" + "="*60)
    print("📊 统计结果 (平均概率)")
    print("="*60)
    actions = ["Fold", "Call", "Bet/Raise", "All-in"] # 假设
    
    for cat in ["Strong", "Weak", "Other"]:
        count = hand_stats[cat]["count"]
        if count > 0:
            avg_probs = hand_stats[cat]["probs"] / count
            prob_fmt = ", ".join([f"{actions[i]}:{avg_probs[i]:.1%}" for i in range(4)])
            print(f"{cat:6} (N={count}): {prob_fmt}")
        else:
            print(f"{cat:6} (N=0): 无样本")

    print("\n💡 预期: Strong 牌的 Bet/Raise 概率应显著高于 Weak 牌。")
    print("       如果所有类别概率分布相似 (例如都接近 25% 或全部 Call)，说明模型无效。")

if __name__ == "__main__":
    test_hand_sensitivity()

