import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np

if __package__:
    from . import run_week2_ensemble as detectors
    from . import week3_vlm_system as visual
    from .ruleshift_env import ACTION_DELTAS, GREEN_TILE as GREEN, RED_TILE as RED, REMAP_ACTION, WALL
else:
    import run_week2_ensemble as detectors
    import week3_vlm_system as visual
    from ruleshift_env import ACTION_DELTAS, GREEN_TILE as GREEN, RED_TILE as RED, REMAP_ACTION, WALL

NAME_TO_ACTION = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "STAY": 4}


# ==============================================================================
# 1. ADAPTIVE A* PLANNER (Hỗ trợ tổ hợp movement_rule x tile_semantics)
# ==============================================================================
class AdaptiveAStarPlanner:
    def __init__(self, movement_rule: str = "NORMAL", tile_semantics: str = "NORMAL"):
        self.movement_rule = movement_rule
        self.tile_semantics = tile_semantics

    def update_rule(self, movement_rule: str, tile_semantics: str):
        self.movement_rule = movement_rule
        self.tile_semantics = tile_semantics

    def _simulate_move(self, grid: np.ndarray, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        H, W = grid.shape
        r, c = pos
        eff_act = REMAP_ACTION[action] if self.movement_rule == "REMAP_90" else action
        dr, dc = ACTION_DELTAS[eff_act]
        steps = 2 if (self.movement_rule == "SLIPPERY" and eff_act != 4) else 1

        nr, nc = r, c
        for _ in range(steps):
            tr, tc = nr + dr, nc + dc
            if 0 <= tr < H and 0 <= tc < W and grid[tr, tc] != WALL:
                nr, nc = tr, tc
                if grid[nr, nc] in (GREEN, RED):
                    break
            else:
                break
        return (nr, nc)

    def plan_action(self, grid: np.ndarray, start: Tuple[int, int]) -> int:
        goal_tile = RED if self.tile_semantics == "SWAPPED_GREEN_RED" else GREEN
        hazard_tile = GREEN if self.tile_semantics == "SWAPPED_GREEN_RED" else RED

        goals = list(zip(*np.where(grid == goal_tile)))
        if not goals:
            return 4

        def h(p):
            return min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for g in goals)

        pq = [(h(start), 0, start, [])]
        visited = {start: 0}

        while pq:
            _, cost, curr, path = heapq.heappop(pq)
            if curr in goals:
                return path[0] if path else 4

            for act in range(4):
                nxt = self._simulate_move(grid, curr, act)
                step_cost = 1 + (15 if grid[nxt[0], nxt[1]] == hazard_tile else 0)
                new_cost = cost + step_cost
                if nxt not in visited or new_cost < visited[nxt]:
                    visited[nxt] = new_cost
                    heapq.heappush(pq, (new_cost + h(nxt), new_cost, nxt, path + [act]))

        return 0  # Fallback nếu không tìm thấy đường


# ==============================================================================
# 2. BASELINE 1: CLOSED-BANK BAYES FILTER (Chỉ có 4 luật cố định R0..R3)
# ==============================================================================
CLOSED_BANK = {
    "R0_BASE":      ("NORMAL",   "NORMAL"),
    "R1_REMAP":     ("REMAP_90", "NORMAL"),
    "R2_SLIPPERY":  ("SLIPPERY", "NORMAL"),
    "R3_SOFT_SWAP": ("NORMAL",   "SWAPPED_GREEN_RED"),
}


class ClosedBankBayesAgent:
    def __init__(self, aleatoric_eps: float = 0.10):
        self.eps = aleatoric_eps
        self.belief = {r: 0.25 for r in CLOSED_BANK}
        self.planner = AdaptiveAStarPlanner()

    def reset(self):
        self.belief = {r: (0.70 if r == "R0_BASE" else 0.10) for r in CLOSED_BANK}
        self.planner.update_rule("NORMAL", "NORMAL")

    def _likelihood(self, rule: str, grid: np.ndarray, prev_pos, action: int, next_pos, delta_hp: int) -> float:
        m_rule, t_sem = CLOSED_BANK[rule]
        tmp_planner = AdaptiveAStarPlanner(m_rule, t_sem)
        pred_pos = tmp_planner._simulate_move(grid, prev_pos, action)

        # Xác suất vị trí (tính cả nhiễu trượt đứng yên aleatoric_eps)
        if next_pos == pred_pos:
            p_pos = 1.0 - self.eps
        elif next_pos == prev_pos:
            p_pos = self.eps
        else:
            p_pos = 0.01

        # Xác suất thay đổi máu
        hazard = GREEN if t_sem == "SWAPPED_GREEN_RED" else RED
        exp_dhp = -25 if grid[next_pos[0], next_pos[1]] == hazard else 0
        p_hp = 0.98 if delta_hp == exp_dhp else 0.02
        return p_pos * p_hp

    def update(self, grid, prev_pos, action, next_pos, delta_hp):
        total = 0.0
        for r in CLOSED_BANK:
            self.belief[r] *= self._likelihood(r, grid, prev_pos, action, next_pos, delta_hp)
            total += self.belief[r]
        for r in CLOSED_BANK:
            self.belief[r] = (self.belief[r] / total) if total > 0 else 0.25

        best_rule = max(self.belief, key=self.belief.get)
        self.planner.update_rule(*CLOSED_BANK[best_rule])


# ==============================================================================
# 3. CHẠY 1 EPISODE CHO TỪNG PHƯƠNG PHÁP
# ==============================================================================
def run_episode(
    method: str,
    rule: str,
    seed: int,
    ensemble: detectors.TransitionEnsemble,
    k_std: float,
    tau_std: float,
    reasoner: visual.FrozenVLMReasoner,
    shift_step: int = 4,
) -> Dict[str, float]:
    env = visual.RuleShiftEnvV2(grid_size=7, max_steps=35, aleatoric_eps=0.10, obs_mode="rgb_som")
    obs, _ = env.reset(seed=seed, options={"target_rule": rule, "shift_step": shift_step})

    planner = AdaptiveAStarPlanner("NORMAL", "NORMAL")
    bayes_agent = ClosedBankBayesAgent()
    bayes_agent.reset()

    # Lưu lịch sử khung hình & transitions để tạo Keyframe Strip khi cần gọi VLM
    frame_buf = deque(maxlen=3)
    act_buf = deque(maxlen=2)
    trans_buf = deque(maxlen=4)

    init_frame = visual.render_som_frame(obs["grid"], tuple(obs["agent_pos"]), int(obs["hp"][0]), 0, None)
    frame_buf.append(init_frame)

    cusum_val = 0.0
    vlm_calls = 0
    rule_committed = False
    probe_queue = deque()
    done = False
    steps = 0
    success = 0.0

    while not done:
        prev_pos = tuple(obs["agent_pos"])
        prev_hp = int(obs["hp"][0])

        # Chọn hành động
        if probe_queue:
            action = probe_queue.popleft()
        elif method == "closed_bayes":
            action = bayes_agent.planner.plan_action(obs["grid"], prev_pos)
        else:
            action = planner.plan_action(obs["grid"], prev_pos)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        next_pos = tuple(next_obs["agent_pos"])
        next_hp = int(next_obs["hp"][0])
        dhp = next_hp - prev_hp

        # Kiểm tra thành công (đến đích còn sống)
        if terminated and next_hp > 0:
            success = 1.0

        # Cập nhật buffer thị giác & transition
        frame_buf.append(info["rgb_som"])
        act_buf.append(action)
        trans_buf.append({
            "step": steps,
            "action": int(action),
            "delta_pos": [int(next_pos[0] - prev_pos[0]), int(next_pos[1] - prev_pos[1])],
            "delta_hp": int(dhp),
            "tile_landed": int(next_obs["grid"][next_pos[0], next_pos[1]]),
        })

        # --- Xử lý thích nghi luật theo từng phương pháp ---
        if method == "closed_bayes":
            bayes_agent.update(obs["grid"], prev_pos, action, next_pos, dhp)

        elif method == "per_step_vlm" and not done:
            strip = visual.make_keyframe_strip(list(frame_buf), list(act_buf))
            out = reasoner.reason(strip, list(trans_buf))
            vlm_calls += 1
            if out["tool"] == "commit_rule":
                planner.update_rule(out["args"]["movement_rule"], out["args"]["tile_semantics"])
            elif out["tool"] == "probe_action":
                probe_queue.append(NAME_TO_ACTION.get(out["args"]["action"], 0))

        elif method == "ours_event_vlm" and not done and not rule_committed:
            # Fast Path: Tính STD Residual từ MLP Ensemble + CUSUM
            x = detectors.extract_features(obs["grid"], prev_pos, action)
            y = np.array([
                float(next_pos[0] - prev_pos[0]),
                float(next_pos[1] - prev_pos[1]),
                float(dhp) / 25.0,
            ], dtype=np.float32)
            _, std_res, _ = ensemble.compute_signals(x, y)
            cusum_val = max(0.0, cusum_val + (float(std_res[0]) - k_std))

            # Slow Path: Chỉ kích hoạt Frozen VLM khi CUSUM vượt ngưỡng tau
            if cusum_val > tau_std and not probe_queue:
                strip = visual.make_keyframe_strip(list(frame_buf), list(act_buf))
                out = reasoner.reason(strip, list(trans_buf))
                vlm_calls += 1

                if out["tool"] == "probe_action" and vlm_calls < 3:
                    probe_queue.append(NAME_TO_ACTION.get(out["args"]["action"], 0))
                else:
                    args = out.get("args", {})
                    planner.update_rule(
                        args.get("movement_rule", "NORMAL"),
                        args.get("tile_semantics", "NORMAL"),
                    )
                    cusum_val = 0.0
                    rule_committed = True

        obs = next_obs

    return {"success": success, "steps": float(steps), "vlm_calls": float(vlm_calls)}


# ==============================================================================
# 4. CHẠY BENCHMARK TỔNG HỢP & IN BẢNG KẾT QUẢ PAPER
# ==============================================================================
def main():
    print("1. Train nhanh Probabilistic MLP Ensemble trên R0_BASE (Tuần 2)...")
    X_train, Y_train = detectors.collect_r0_dataset(num_transitions=12000, seed=10)
    ensemble = detectors.TransitionEnsemble(num_models=5, device="cpu")
    ensemble.fit(X_train, Y_train, epochs=20, batch_size=256)

    print("2. Hiệu chuẩn ngưỡng k và tau (FAR = 5%) cho STD + CUSUM...")
    val_eps = detectors.generate_eval_episodes(ensemble, rule="R0_BASE", num_eps=50, shift_step=4, seed_offset=5000)
    all_std = np.concatenate([ep["std"] for ep in val_eps])
    k_std = float(np.median(all_std) + 1.0 * np.std(all_std))
    tau_std = float(np.percentile([np.max(detectors.apply_cusum(ep["std"], k_std)) for ep in val_eps], 95))
    print(f"   -> k_std = {k_std:.3f}, tau_std = {tau_std:.3f}")

    reasoner = visual.FrozenVLMReasoner(backend="mock", cache_path="vlm_cache_week4.db")
    methods = [
        ("closed_bayes",   "1. Closed-Bank Bayes (W1-2)"),
        ("per_step_vlm",   "2. Per-step mock"),
        ("ours_event_vlm", "3. STD+CUSUM + mock"),
    ]
    num_eps_per_rule = 30

    print(f"\n3. SYNTHETIC mock run ({num_eps_per_rule} episodes/luật x 5 luật)...")
    print(f"\n{'Method':<28} | {'SR (R0..R3)':>11} | {'SR (R4_COMP)':>12} | {'Avg Steps':>9} | {'Mock calls (R0)':>8} | {'Mock calls (R4)':>8}")
    print("-" * 92)

    for m_key, m_label in methods:
        res_r0_r3, res_r4, calls_r0, calls_r4, all_steps = [], [], [], [], []
        for r_idx, rule in enumerate(visual.RULES_V2):
            for ep in range(num_eps_per_rule):
                out = run_episode(
                    method=m_key,
                    rule=rule,
                    seed=10000 + r_idx * 100 + ep,
                    ensemble=ensemble,
                    k_std=k_std,
                    tau_std=tau_std,
                    reasoner=reasoner,
                    shift_step=4,
                )
                all_steps.append(out["steps"])
                if rule == "R4_COMPOUND":
                    res_r4.append(out["success"])
                    calls_r4.append(out["vlm_calls"])
                else:
                    res_r0_r3.append(out["success"])
                if rule == "R0_BASE":
                    calls_r0.append(out["vlm_calls"])

        print(
            f"{m_label:<28} | "
            f"{100 * np.mean(res_r0_r3):>10.1f}% | "
            f"{100 * np.mean(res_r4):>11.1f}% | "
            f"{np.mean(all_steps):>9.2f} | "
            f"{np.mean(calls_r0):>8.2f} | "
            f"{np.mean(calls_r4):>8.2f}"
        )

    print(f"\n[Disk Cache Stats] API calls: {reasoner.api_calls} | Cache hits: {reasoner.cache_hits}")


if __name__ == "__main__":
    main()
