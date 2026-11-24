import heapq

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# Mã vật thể trên lưới
EMPTY = 0
WALL = 1
GREEN_TILE = 2  # Goal ở R0, R1, R2; Soft Hazard (-25 HP) ở R3
RED_TILE = 3    # Soft Hazard (-25 HP) ở R0, R1, R2; Goal ở R3

# Hành động: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: INTERACT
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}

RULES = ["R0_BASE", "R1_REMAP", "R2_SLIPPERY", "R3_SOFT_SWAP"]
REMAP_ACTION = {0: 2, 2: 0, 1: 3, 3: 1, 4: 4}


class RuleShiftEnv(gym.Env):
    """
    RuleShift-Grid v2 (Gymnasium-compatible).
    Hỗ trợ nhiễu aleatoric, Soft Hazard (-25 HP) và đổi luật giữa chừng hoặc đầu episode.
    """
    metadata = {"render_modes": ["ansi"]}
    rules = RULES

    def __init__(
        self,
        grid_size: int = 7,
        max_steps: int = 40,
        aleatoric_eps: float = 0.10,
        tool_cost_weight: float = 5.0,
        obs_mode: str = "symbolic",
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.aleatoric_eps = aleatoric_eps
        self.tool_cost_weight = tool_cost_weight
        self.obs_mode = obs_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 3, shape=(grid_size, grid_size), dtype=np.int32),
            "agent_pos": spaces.Box(0, grid_size - 1, shape=(2,), dtype=np.int32),
            "hp": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
        })

        self.grid = None
        self.agent_pos = None
        self.hp = 100
        self.step_count = 0
        self.tool_calls = 0
        self.base_rule = "R0_BASE"
        self.target_rule = "R0_BASE"
        self.shift_step = 0
        self.current_rule = "R0_BASE"

    def _generate_map(self, rng):
        """Tạo bản đồ ngẫu nhiên đảm bảo có đường đi từ agent đến cả ô Xanh và ô Đỏ."""
        while True:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            # Đặt tường bao quanh
            grid[0, :] = WALL
            grid[-1, :] = WALL
            grid[:, 0] = WALL
            grid[:, -1] = WALL

            # Đặt tường ngẫu nhiên bên trong
            inner_cells = [
                (r, c)
                for r in range(1, self.grid_size - 1)
                for c in range(1, self.grid_size - 1)
            ]
            rng.shuffle(inner_cells)

            agent_pos = inner_cells.pop()
            green_pos = inner_cells.pop()
            red_pos = inner_cells.pop()

            grid[green_pos] = GREEN_TILE
            grid[red_pos] = RED_TILE

            # Thêm 3-4 tường cản
            for _ in range(4):
                if inner_cells:
                    w = inner_cells.pop()
                    grid[w] = WALL

            if self._is_reachable(grid, agent_pos, green_pos) and self._is_reachable(
                grid, agent_pos, red_pos
            ):
                return grid, agent_pos

    def _is_reachable(self, grid, start, goal):
        q = deque([start])
        visited = {start}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and grid[nr, nc] != WALL:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        options = options or {}
        self.target_rule = options.get("target_rule", rng.choice(self.rules))
        if self.target_rule not in self.rules:
            raise ValueError(f"Unknown target rule: {self.target_rule}")
        self.shift_step = options.get("shift_step", 0)
        self.current_rule = (
            self.target_rule if self.shift_step == 0 else self.base_rule
        )

        self.grid, self.agent_pos = self._generate_map(rng)
        self.hp = 100
        self.step_count = 0
        self.tool_calls = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "hp": np.array([self.hp], dtype=np.int32),
        }

    def _get_info(self):
        return {
            "env_steps": self.step_count,
            "tool_calls": self.tool_calls,
            "total_cost": self.step_count + self.tool_cost_weight * self.tool_calls,
        }

    @staticmethod
    def get_transition_distribution(grid, pos, action, rule, eps=0.10):
        """
        Hàm chuyển xác suất giải tích P(outcome | pos, action, rule).
        Trả về dict: {(next_pos, hp_delta, reached_goal): prob}
        Dùng cho cả env.step() lẫn Bayesian Belief Update ở Tuần 1.
        """
        def apply_move(p, act, r_name):
            # 1. Xử lý ánh xạ phím
            if r_name in ("R1_REMAP", "R4_COMPOUND"):
                eff_act = REMAP_ACTION[act]
            else:
                eff_act = act

            dr, dc = ACTION_DELTAS[eff_act]
            steps = 2 if (r_name == "R2_SLIPPERY" and eff_act != 4) else 1

            curr_r, curr_c = p
            for _ in range(steps):
                nr, nc = curr_r + dr, curr_c + dc
                if grid[nr, nc] == WALL:
                    break
                curr_r, curr_c = nr, nc
                # Dừng ngay nếu chạm ô đặc biệt
                if grid[curr_r, curr_c] in (GREEN_TILE, RED_TILE):
                    break

            # 2. Xử lý hiệu ứng ô đích / soft hazard
            tile = grid[curr_r, curr_c]
            swapped = r_name in ("R3_SOFT_SWAP", "R4_COMPOUND")
            goal_tile = RED_TILE if swapped else GREEN_TILE
            hazard_tile = GREEN_TILE if swapped else RED_TILE

            reached_goal = (tile == goal_tile)
            hp_delta = -25 if (tile == hazard_tile) else 0
            return (curr_r, curr_c), hp_delta, reached_goal

        dist = {}
        # Nhánh chính (1 - eps): thực hiện đúng hành động
        main_out = apply_move(pos, action, rule)
        dist[main_out] = dist.get(main_out, 0.0) + (1.0 - eps)

        # Nhánh nhiễu aleatoric (eps): trượt đứng yên (action=4)
        noise_out = apply_move(pos, 4, rule)
        dist[noise_out] = dist.get(noise_out, 0.0) + eps
        return dist

    def step(self, action: int):
        self.step_count += 1
        if self.step_count >= self.shift_step:
            self.current_rule = self.target_rule

        # Lấy phân phối chuyển dịch thật
        dist = self.get_transition_distribution(
            self.grid, self.agent_pos, action, self.current_rule, self.aleatoric_eps
        )
        outcomes = list(dist.keys())
        probs = list(dist.values())

        idx = self.np_random.choice(len(outcomes), p=probs)
        next_pos, hp_delta, reached_goal = outcomes[idx]

        self.agent_pos = next_pos
        self.hp = max(0, self.hp + hp_delta)

        terminated = reached_goal or (self.hp <= 0)
        truncated = self.step_count >= self.max_steps

        reward = 100.0 if reached_goal else (float(hp_delta) - 1.0)
        return self._get_obs(), reward, bool(terminated), truncated, self._get_info()

    def call_tool(self, tool_name: str, target=None):
        """Cổng mở rộng cho Epistemic Tools ở Tầng B (INSPECT / READ_CLUE)."""
        self.tool_calls += 1
        if tool_name == "INSPECT" and target is not None:
            r, c = target
            return {"tile": int(self.grid[r, c]), "pos": target}
        return {"status": "ok"}


def entropy(probs):
    p = probs[probs > 1e-12]
    return -np.sum(p * np.log2(p))


def astar_first_action(grid, start, rule):
    """Tìm hành động đầu tiên tối ưu theo A* giả định luật `rule` là đúng."""
    goal_tile = RED_TILE if rule == "R3_SOFT_SWAP" else GREEN_TILE
    hazard_tile = GREEN_TILE if rule == "R3_SOFT_SWAP" else RED_TILE

    goals = list(zip(*np.where(grid == goal_tile)))
    if not goals:
        return 4
    goal = goals[0]

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    pq = [(h(start), 0, start, None)]
    visited = {start: 0}

    while pq:
        _, g, curr, first_act = heapq.heappop(pq)
        if curr == goal:
            return first_act if first_act is not None else 4

        for act in range(4):
            # Lấy trạng thái kỳ vọng chính (eps=0) để lập kế hoạch
            dist = RuleShiftEnv.get_transition_distribution(grid, curr, act, rule, eps=0.0)
            nxt, hp_delta, _ = next(iter(dist.keys()))
            if nxt == curr:
                continue
            step_cost = 15 if grid[nxt] == hazard_tile else 1
            ng = g + step_cost
            if nxt not in visited or ng < visited[nxt]:
                visited[nxt] = ng
                fa = act if first_act is None else first_act
                heapq.heappush(pq, (ng + h(nxt), ng, nxt, fa))
    return 4


class BeliefAgent:
    def __init__(self, mode: str, eps: float = 0.10, trigger_conf: float = 0.85):
        """
        mode:
          - 'no_adapt': Luôn dùng R0_BASE
          - 'action_advice': Khi bất định, gọi advisor xin 1 hành động (tốn tool_cost, không cập nhật belief)
          - 'random_probe': Khi bất định, đi ngẫu nhiên 1 bước để cập nhật belief
          - 'infogain_probe': Khi bất định, chọn hành động tối đa hóa Expected Information Gain
          - 'oracle': Biết trước luật thật
        """
        self.mode = mode
        self.eps = eps
        self.trigger_conf = trigger_conf
        self.reset()

    def reset(self):
        # Prior ban đầu nghiêng về R0_BASE nhưng vẫn cho xác suất nhỏ cho các luật khác
        self.belief = np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float64)
        self.probed_steps = 0

    def update_belief(self, grid, prev_pos, action, next_pos, hp_delta, reached_goal):
        if self.mode in ("no_adapt", "action_advice", "oracle"):
            return
        outcome = (next_pos, hp_delta, reached_goal)
        likelihoods = np.zeros(len(RULES), dtype=np.float64)
        for i, r in enumerate(RULES):
            dist = RuleShiftEnv.get_transition_distribution(grid, prev_pos, action, r, self.eps)
            likelihoods[i] = dist.get(outcome, 1e-6)

        posterior = self.belief * likelihoods
        if posterior.sum() > 0:
            self.belief = posterior / posterior.sum()
        else:
            self.belief = np.ones(len(RULES)) / len(RULES)

    def select_infogain_action(self, grid, pos):
        curr_H = entropy(self.belief)
        best_act, best_ig = 0, -1e9

        for act in range(4):
            # Gom phân phối biên P(o | pos, act)
            outcome_probs = {}
            rule_dists = []
            for i, r in enumerate(RULES):
                d = RuleShiftEnv.get_transition_distribution(grid, pos, act, r, self.eps)
                rule_dists.append(d)
                for o, p in d.items():
                    outcome_probs[o] = outcome_probs.get(o, 0.0) + self.belief[i] * p

            # Tính kỳ vọng entropy hậu nghiệm
            exp_H = 0.0
            for o, p_o in outcome_probs.items():
                if p_o < 1e-9:
                    continue
                post = np.array([self.belief[i] * rule_dists[i].get(o, 0.0) for i in range(len(RULES))])
                post /= post.sum()
                exp_H += p_o * entropy(post)

            ig = curr_H - exp_H
            if ig > best_ig:
                best_ig = ig
                best_act = act
        return best_act

    def act(self, env: RuleShiftEnv, obs, rng):
        grid = obs["grid"]
        pos = tuple(obs["agent_pos"])

        if self.mode == "oracle":
            return astar_first_action(grid, pos, env.current_rule)

        if self.mode == "no_adapt":
            return astar_first_action(grid, pos, "R0_BASE")

        max_conf = float(np.max(self.belief))
        best_rule = RULES[int(np.argmax(self.belief))]

        # Kích hoạt thăm dò / gọi advisor khi độ tin cậy tụt xuống dưới ngưỡng
        if max_conf < self.trigger_conf:
            self.probed_steps += 1
            if self.mode == "action_advice":
                env.call_tool("ACTION_ADVICE")  # Tốn chi phí gọi advisor
                return astar_first_action(grid, pos, env.current_rule)
            elif self.mode == "random_probe":
                return int(rng.integers(0, 4))
            elif self.mode == "infogain_probe":
                return self.select_infogain_action(grid, pos)

        return astar_first_action(grid, pos, best_rule)


def run_benchmark(num_episodes_per_rule=50):
    modes = ["no_adapt", "action_advice", "random_probe", "infogain_probe", "oracle"]
    env = RuleShiftEnv(grid_size=7, max_steps=40, aleatoric_eps=0.10, tool_cost_weight=5.0)

    print(f"{'Agent':<16} | {'Success %':>9} | {'Avg Steps':>9} | {'Total Cost':>10} | {'Adapt Steps':>11} | {'False Alarm (R0)':>16}")
    print("-" * 86)

    for mode in modes:
        successes = []
        steps_list = []
        costs_list = []
        adapt_steps_list = []
        r0_false_alarms = []

        rng = np.random.default_rng(42)
        ep_idx = 0

        for rule in RULES:
            for _ in range(num_episodes_per_rule):
                obs, info = env.reset(seed=1000 + ep_idx, options={"target_rule": rule, "shift_step": 0})
                ep_idx += 1

                agent = BeliefAgent(mode=mode, eps=0.10, trigger_conf=0.85)
                done = False
                adapted_at = None

                # Với action_advice, vẫn cập nhật một bộ detector riêng để biết lúc nào cần gọi advisor
                detector_belief = np.array([0.85, 0.05, 0.05, 0.05])

                while not done:
                    prev_pos = tuple(obs["agent_pos"])
                    prev_hp = int(obs["hp"][0])

                    if mode == "action_advice":
                        agent.belief = detector_belief

                    action = agent.act(env, obs, rng)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    next_pos = tuple(next_obs["agent_pos"])
                    hp_delta = int(next_obs["hp"][0]) - prev_hp
                    reached_goal = terminated and (int(next_obs["hp"][0]) > 0)

                    # Cập nhật belief
                    agent.update_belief(obs["grid"], prev_pos, action, next_pos, hp_delta, reached_goal)

                    # Cập nhật detector cho action_advice (chỉ để kích hoạt trigger, không dùng để đổi luật)
                    if mode == "action_advice":
                        outcome = (next_pos, hp_delta, reached_goal)
                        lik = np.array([
                            RuleShiftEnv.get_transition_distribution(obs["grid"], prev_pos, action, r, 0.10).get(outcome, 1e-6)
                            for r in RULES
                        ])
                        detector_belief = detector_belief * lik
                        detector_belief /= detector_belief.sum()

                    # Ghi nhận bước đầu tiên nhận diện đúng luật (với các luật đã đổi)
                    if rule != "R0_BASE" and adapted_at is None:
                        if RULES[int(np.argmax(agent.belief))] == rule and np.max(agent.belief) >= 0.85:
                            adapted_at = info["env_steps"]

                    obs = next_obs

                successes.append(1.0 if (terminated and int(obs["hp"][0]) > 0) else 0.0)
                steps_list.append(info["env_steps"])
                costs_list.append(info["total_cost"])
                if rule != "R0_BASE":
                    adapt_steps_list.append(adapted_at if adapted_at is not None else env.max_steps)
                else:
                    r0_false_alarms.append(1.0 if agent.probed_steps > 0 else 0.0)

        print(
            f"{('oracle_advice' if mode == 'action_advice' else mode):<16} | {100*np.mean(successes):>8.1f}% | {np.mean(steps_list):>9.2f} | "
            f"{np.mean(costs_list):>10.2f} | {np.mean(adapt_steps_list):>11.2f} | {100*np.mean(r0_false_alarms):>15.1f}%"
        )


if __name__ == "__main__":
    run_benchmark(num_episodes_per_rule=50)
