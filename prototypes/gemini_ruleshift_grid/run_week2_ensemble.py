import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
if __package__:
    from .ruleshift_env import RuleShiftEnv, RULES, WALL
else:
    from ruleshift_env import RuleShiftEnv, RULES, WALL


# ==========================================
# 1. Trích xuất đặc trưng (Egocentric 5x5)
# ==========================================
def extract_features(grid: np.ndarray, pos: tuple, action: int) -> np.ndarray:
    """
    Trích xuất vùng lưới 5x5 quanh agent (one-hot 4 kênh = 100 chiều)
    + hành động (one-hot 5 chiều) -> vector 105 chiều.
    """
    H, W = grid.shape
    r, c = int(pos[0]), int(pos[1])
    patch = np.zeros((5, 5, 4), dtype=np.float32)

    for i, dr in enumerate(range(-2, 3)):
        for j, dc in enumerate(range(-2, 3)):
            nr, nc = r + dr, c + dc
            tile = grid[nr, nc] if (0 <= nr < H and 0 <= nc < W) else WALL
            patch[i, j, int(tile)] = 1.0

    act_onehot = np.zeros(5, dtype=np.float32)
    act_onehot[int(action)] = 1.0
    return np.concatenate([patch.reshape(-1), act_onehot], axis=0)


def collect_r0_dataset(num_transitions: int = 15000, seed: int = 0):
    """Thu thập dữ liệu transition chỉ trên luật gốc R0_BASE."""
    env = RuleShiftEnv(grid_size=7, max_steps=40, aleatoric_eps=0.10)
    rng = np.random.default_rng(seed)

    X_list, Y_list = [], []
    ep = 0
    while len(X_list) < num_transitions:
        obs, _ = env.reset(seed=seed + ep, options={"target_rule": "R0_BASE", "shift_step": 0})
        ep += 1
        done = False
        while not done and len(X_list) < num_transitions:
            prev_pos = tuple(obs["agent_pos"])
            prev_hp = int(obs["hp"][0])
            action = int(rng.integers(0, 4))  # Ưu tiên hành động di chuyển

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_pos = tuple(next_obs["agent_pos"])
            next_hp = int(next_obs["hp"][0])

            x = extract_features(obs["grid"], prev_pos, action)
            y = np.array([
                float(next_pos[0] - prev_pos[0]),
                float(next_pos[1] - prev_pos[1]),
                float(next_hp - prev_hp) / 25.0,
            ], dtype=np.float32)

            X_list.append(x)
            Y_list.append(y)
            obs = next_obs

    return np.stack(X_list), np.stack(Y_list)


# ==========================================
# 2. Probabilistic MLP Ensemble (5 mạng)
# ==========================================
class ProbabilisticMLP(nn.Module):
    def __init__(self, in_dim: int = 105, hidden_dim: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-6.0, max=2.0)
        return mu, logvar


def gaussian_nll_loss(mu, logvar, target):
    inv_var = torch.exp(-logvar)
    loss = 0.5 * (inv_var * (target - mu) ** 2 + logvar)
    return loss.sum(dim=-1).mean()


class TransitionEnsemble:
    def __init__(self, num_models: int = 5, device: str = "cpu"):
        self.num_models = num_models
        self.device = device
        self.models = [ProbabilisticMLP().to(device) for _ in range(num_models)]

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 25, batch_size: int = 256):
        N = len(X)
        rng = np.random.default_rng(42)
        for m_idx, model in enumerate(self.models):
            # Bootstrap 80% dữ liệu cho mỗi mạng để tạo độ đa dạng epistemic
            idx = rng.choice(N, size=int(0.8 * N), replace=True)
            ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx]))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            for _ in range(epochs):
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    mu, logvar = model(bx)
                    loss = gaussian_nll_loss(mu, logvar, by)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            print(f"  [Train] Finished model {m_idx + 1}/{self.num_models} (final NLL: {loss.item():.4f})")

    @torch.no_grad()
    def compute_signals(self, x_np: np.ndarray, y_np: np.ndarray):
        """
        Trả về 3 tín hiệu tại một bước (hoặc batch):
          - raw_res: ||y - mu_bar||_2
          - std_res: sqrt(sum((y - mu_bar)^2 / (var_alea + var_epi + 1e-4)))
          - disagreement: sum(var_epi)
        """
        x = torch.from_numpy(x_np).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        y = torch.from_numpy(y_np).to(self.device)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        mus, vars_ = [], []
        for model in self.models:
            mu, logvar = model(x)
            mus.append(mu)
            vars_.append(torch.exp(logvar))

        mus = torch.stack(mus, dim=0)    # (M, B, 3)
        vars_ = torch.stack(vars_, dim=0)  # (M, B, 3)

        mu_bar = mus.mean(dim=0)                         # (B, 3)
        var_alea = vars_.mean(dim=0)                     # (B, 3)
        var_epi = ((mus - mu_bar.unsqueeze(0)) ** 2).mean(dim=0)  # (B, 3)

        diff_sq = (y - mu_bar) ** 2
        raw_res = torch.sqrt(diff_sq.sum(dim=-1))
        std_res = torch.sqrt((diff_sq / (var_alea + var_epi + 1e-4)).sum(dim=-1))
        disagreement = var_epi.sum(dim=-1)

        return (
            raw_res.cpu().numpy(),
            std_res.cpu().numpy(),
            disagreement.cpu().numpy(),
        )


# ==========================================
# 3. Hàm tính AUROC thuần NumPy & CUSUM
# ==========================================
def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tính AUROC không cần sklearn."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # So sánh theo hạng (Mann-Whitney U)
    neg.sort()
    # Each tied positive/negative pair contributes half a win.
    lower = np.searchsorted(neg, pos, side="left")
    upper = np.searchsorted(neg, pos, side="right")
    return float(np.mean((lower + upper) / (2 * len(neg))))


def apply_cusum(seq: np.ndarray, k: float) -> np.ndarray:
    c = np.zeros_like(seq, dtype=np.float64)
    for t in range(len(seq)):
        prev = c[t - 1] if t > 0 else 0.0
        c[t] = max(0.0, prev + (seq[t] - k))
    return c


# ==========================================
# 4. Chạy thí nghiệm 2A: So sánh 3 tín hiệu
# ==========================================
def generate_eval_episodes(ensemble: TransitionEnsemble, rule: str, num_eps: int, shift_step: int, seed_offset: int):
    """Sinh episode kiểm thử dài tối thiểu (shift_step + 10) bước để đo Delay & AUROC."""
    env = RuleShiftEnv(grid_size=7, max_steps=30, aleatoric_eps=0.10)
    rng = np.random.default_rng(seed_offset)
    episodes = []
    attempt = 0

    while len(episodes) < num_eps:
        obs, _ = env.reset(
            seed=seed_offset + attempt,
            options={"target_rule": rule, "shift_step": shift_step},
        )
        attempt += 1
        xs, ys = [], []
        done = False
        while not done:
            prev_pos = tuple(obs["agent_pos"])
            prev_hp = int(obs["hp"][0])
            action = int(rng.integers(0, 4))
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            xs.append(extract_features(obs["grid"], prev_pos, action))
            ys.append(np.array([
                float(next_obs["agent_pos"][0] - prev_pos[0]),
                float(next_obs["agent_pos"][1] - prev_pos[1]),
                float(int(next_obs["hp"][0]) - prev_hp) / 25.0,
            ], dtype=np.float32))
            obs = next_obs

        if len(xs) >= shift_step + 8:
            raw, std, dis = ensemble.compute_signals(np.stack(xs), np.stack(ys))
            episodes.append({"raw": raw, "std": std, "dis": dis})
    return episodes


def main():
    print("1. Thu thập 15,000 transitions trên R0_BASE...")
    X_train, Y_train = collect_r0_dataset(num_transitions=15000, seed=10)

    print("2. Huấn luyện Probabilistic MLP Ensemble (5 mạng)...")
    ensemble = TransitionEnsemble(num_models=5, device="cpu")
    ensemble.fit(X_train, Y_train, epochs=25, batch_size=256)

    print("3. Hiệu chuẩn ngưỡng nền k và ngưỡng tau (FAR = 5%) trên tập Validation R0_BASE...")
    val_eps = generate_eval_episodes(ensemble, rule="R0_BASE", num_eps=60, shift_step=5, seed_offset=5000)

    signal_names = ["raw", "std", "dis"]
    k_drift = {}
    tau_single = {}
    tau_cusum = {}

    for name in signal_names:
        all_vals = np.concatenate([ep[name] for ep in val_eps])
        # Mức nền CUSUM: median + 1.0 * std trên R0_BASE
        k_drift[name] = float(np.median(all_vals) + 1.0 * np.std(all_vals))

        # Ngưỡng FAR = 5% tính trên giá trị max của mỗi episode R0_BASE
        max_single = [np.max(ep[name]) for ep in val_eps]
        max_cusum = [np.max(apply_cusum(ep[name], k_drift[name])) for ep in val_eps]
        tau_single[name] = float(np.percentile(max_single, 95))
        tau_cusum[name] = float(np.percentile(max_cusum, 95))

    print("\n4. Đánh giá trên tập Test (50 episodes/luật, shift_step = 5):")
    shift_step = 5
    test_data = {
        r: generate_eval_episodes(ensemble, rule=r, num_eps=50, shift_step=shift_step, seed_offset=9000 + i * 500)
        for i, r in enumerate(RULES)
    }

    print(f"\n{'Signal':<20} | {'AUROC (All)':>11} | {'Delay R1':>9} | {'Delay R2':>9} | {'Delay R3':>9} | {'Test FAR (R0)':>13}")
    print("-" * 86)

    for name in signal_names:
        for use_cusum in [False, True]:
            label_str = f"{name.upper()} + CUSUM" if use_cusum else f"{name.upper()} (Single)"
            tau = tau_cusum[name] if use_cusum else tau_single[name]

            all_labels, all_scores = [], []
            delays = {"R1_REMAP": [], "R2_SLIPPERY": [], "R3_SOFT_SWAP": []}
            r0_alarms = []

            for r in RULES:
                for ep in test_data[r]:
                    seq = apply_cusum(ep[name], k_drift[name]) if use_cusum else ep[name]
                    if r == "R0_BASE":
                        all_labels.extend([0] * len(seq))
                        all_scores.extend(seq.tolist())
                        r0_alarms.append(1.0 if np.any(seq > tau) else 0.0)
                    else:
                        # Chỉ tính bước sau shift_step là positive (1)
                        first_shifted = max(0, shift_step - 1)
                        post = seq[first_shifted:]
                        all_labels.extend([0] * first_shifted + [1] * len(post))
                        all_scores.extend(seq.tolist())

                        triggered = np.where(post > tau)[0]
                        delay = int(triggered[0]) + 1 if len(triggered) > 0 else len(post)
                        delays[r].append(delay)

            auroc = compute_auroc(np.array(all_labels), np.array(all_scores))
            print(
                f"{label_str:<20} | {auroc:>11.3f} | "
                f"{np.mean(delays['R1_REMAP']):>9.2f} | "
                f"{np.mean(delays['R2_SLIPPERY']):>9.2f} | "
                f"{np.mean(delays['R3_SOFT_SWAP']):>9.2f} | "
                f"{100 * np.mean(r0_alarms):>12.1f}%"
            )


if __name__ == "__main__":
    main()
