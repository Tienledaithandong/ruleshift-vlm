import io
import json
import sqlite3
import hashlib
import base64
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .ruleshift_env import ACTION_DELTAS, EMPTY, GREEN_TILE as GREEN, RED_TILE as RED, REMAP_ACTION, RULES, WALL, RuleShiftEnv
else:
    from ruleshift_env import ACTION_DELTAS, EMPTY, GREEN_TILE as GREEN, RED_TILE as RED, REMAP_ACTION, RULES, WALL, RuleShiftEnv

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


# ==============================================================================
# 1. RENDER PIXEL + SET-OF-MARK (rgb_som) & KEYFRAME STRIP
# ==============================================================================
COLORS = {
    EMPTY: (243, 244, 246),   # #F3F4F6 - Trắng xám
    WALL:  (55, 65, 81),      # #374151 - Xám đậm
    GREEN: (34, 197, 94),     # #22C55E - Xanh lá
    RED:   (239, 68, 68),     # #EF4444 - Đỏ
}
AGENT_COLOR = (37, 99, 235)   # #2563EB - Xanh dương
HUD_BG = (17, 24, 39)         # #111827 - Đen xanh


def render_som_frame(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    hp: int,
    step: int,
    last_action: Optional[int] = None,
    tile_size: int = 64,
    hud_height: int = 48,
) -> Image.Image:
    """Render lưới 7x7 thành ảnh RGB 448x496 kèm nhãn tọa độ Set-of-Mark (r,c) và HUD."""
    H, W = grid.shape
    img_w, img_h = W * tile_size, H * tile_size + hud_height
    img = Image.new("RGB", (img_w, img_h), HUD_BG)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # 1. Vẽ thanh HUD
    act_str = ACTION_NAMES.get(last_action, "NONE")
    hud_text = f"HP: {hp}/100  |  Step: {step}  |  Last Action: {act_str}"
    draw.text((12, 16), hud_text, fill=(255, 255, 255), font=font)

    # 2. Vẽ các ô lưới + nhãn Set-of-Mark
    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    for r in range(H):
        for c in range(W):
            x0, y0 = c * tile_size, hud_height + r * tile_size
            x1, y1 = x0 + tile_size, y0 + tile_size
            tile = int(grid[r, c])

            draw.rectangle([x0, y0, x1, y1], fill=COLORS[tile], outline=(209, 213, 219), width=1)

            # Nhãn tọa độ SoM ở góc trên-trái mỗi ô
            text_color = (229, 231, 235) if tile == WALL else (31, 41, 55)
            draw.text((x0 + 4, y0 + 3), f"{r},{c}", fill=text_color, font=font)

            # Ký hiệu phụ (G / R) giúp VLM không nhầm màu
            if tile == GREEN:
                draw.text((x1 - 14, y1 - 14), "G", fill=(255, 255, 255), font=font)
            elif tile == RED:
                draw.text((x1 - 14, y1 - 14), "R", fill=(255, 255, 255), font=font)

    # 3. Vẽ Agent (hình tròn xanh dương + chữ A)
    cx = ac * tile_size + tile_size // 2
    cy = hud_height + ar * tile_size + tile_size // 2
    rad = tile_size // 3
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=AGENT_COLOR, outline=(255, 255, 255), width=2)
    draw.text((cx - 3, cy - 5), "A", fill=(255, 255, 255), font=font)

    return img


def make_keyframe_strip(frames: List[Image.Image], actions: List[int]) -> Image.Image:
    """
    Ghép ngang 2-3 khung hình gần nhất thành Keyframe Strip kèm mũi tên hành động:
    [Frame t-1] --(UP)--> [Frame t]
    """
    assert len(frames) >= 1
    w, h = frames[0].size
    gap = 64
    total_w = len(frames) * w + (len(frames) - 1) * gap
    strip = Image.new("RGB", (total_w, h), (31, 41, 55))
    draw = ImageDraw.Draw(strip)
    font = ImageFont.load_default()

    for i, frame in enumerate(frames):
        x_offset = i * (w + gap)
        strip.paste(frame, (x_offset, 0))
        if i < len(frames) - 1:
            ax = x_offset + w
            ay = h // 2
            act_label = ACTION_NAMES.get(actions[i], "?")
            draw.line([ax + 8, ay, ax + gap - 10, ay], fill=(250, 204, 21), width=3)
            draw.polygon(
                [(ax + gap - 10, ay - 6), (ax + gap - 2, ay), (ax + gap - 10, ay + 6)],
                fill=(250, 204, 21),
            )
            draw.text((ax + 12, ay - 18), act_label, fill=(250, 204, 21), font=font)

    return strip


# ==============================================================================
# 2. MỞ RỘNG MÔI TRƯỜNG: LUẬT R4_COMPOUND + obs_mode='rgb_som'
# ==============================================================================
RULES_V2 = [*RULES, "R4_COMPOUND"]


class RuleShiftEnvV2(RuleShiftEnv):
    """
    Kế thừa RuleShiftEnv (Tuần 1), bổ sung:
      - Luật R4_COMPOUND = R1_REMAP (đảo phím UP<->LEFT, DOWN<->RIGHT)
                         + R3_SOFT_SWAP (GREEN gây -25 HP, RED là đích).
      - obs_mode='rgb_som': trả thêm ảnh PIL Set-of-Mark trong info['rgb_som'].
    """
    rules = RULES_V2

    def __init__(self, *args, obs_mode: str = "symbolic", **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_mode = obs_mode
        self.last_action = None

    def step(self, action: int):
        self.last_action = int(action)
        obs, reward, terminated, truncated, info = super().step(action)

        if self.obs_mode == "rgb_som":
            info["rgb_som"] = render_som_frame(
                grid=obs["grid"],
                agent_pos=tuple(obs["agent_pos"]),
                hp=int(obs["hp"][0]),
                step=getattr(self, "step_count", 0),
                last_action=self.last_action,
            )
        return obs, reward, terminated, truncated, info


# ==============================================================================
# 3. FROZEN VLM REASONER + EPISTEMIC TOOLS + SQLITE DISK CACHE + MOCK MODE
# ==============================================================================
SYSTEM_PROMPT = """You are an epistemic rule reasoner for a 7x7 gridworld.
The environment dynamics may have shifted from R0_BASE (normal movement, GREEN=goal, RED=-25 HP).
Analyze the attached Keyframe Strip (Set-of-Mark coordinates) and recent transitions.

You MUST respond with a single valid JSON object calling ONE of these two Epistemic Tools:
1. {"tool": "probe_action", "args": {"action": "UP|DOWN|LEFT|RIGHT", "reason": "..."}}
   Use this if the observed transition could be an incidental slip or an unseen tile effect.
2. {"tool": "commit_rule", "args": {"movement_rule": "NORMAL|REMAP_90|SLIPPERY", "tile_semantics": "NORMAL|SWAPPED_GREEN_RED", "reasoning": "..."}}
   Use this when you have sufficient evidence to commit the new rule for the A* planner.
"""


class VLMDiskCache:
    """Cache kết quả VLM xuống SQLite theo SHA-256 của (model + prompt + image_bytes)."""
    def __init__(self, db_path: str = "vlm_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)"
        )
        self.conn.commit()

    @staticmethod
    def make_key(model: str, prompt: str, img_bytes: bytes) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(prompt.encode("utf-8"))
        h.update(img_bytes)
        return h.hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Dict[str, Any]):
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()


class FrozenVLMReasoner:
    def __init__(
        self,
        backend: str = "mock",        # "mock" để debug 0 đồng, "openai_compat" cho API thật
        model_name: str = "gpt-4o-mini",
        cache_path: str = "vlm_cache.db",
        api_client: Any = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.cache = VLMDiskCache(cache_path)
        self.api_client = api_client
        self.api_calls = 0
        self.cache_hits = 0

    def _image_to_png_bytes(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def reason(
        self,
        keyframe_strip: Image.Image,
        recent_transitions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Gọi VLM (có Disk Cache) để chọn 1 trong 2 Epistemic Tools:
          - probe_action
          - commit_rule
        """
        user_prompt = "Recent transitions:\n" + json.dumps(recent_transitions, indent=2)
        img_bytes = self._image_to_png_bytes(keyframe_strip)
        cache_key = self.cache.make_key(f"{self.backend}:{self.model_name}", SYSTEM_PROMPT + user_prompt, img_bytes)

        # 1. Kiểm tra Disk Cache trước
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # 2. Gọi backend nếu chưa có trong cache
        if self.backend == "mock":
            result = self._mock_inference(recent_transitions)
        elif self.backend == "openai_compat":
            result = self._call_openai_compatible(user_prompt, img_bytes)
            self.api_calls += 1
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.cache.set(cache_key, result)
        return result

    def _mock_inference(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bộ suy luận giả lập tất định để test toàn bộ pipeline A* + Epistemic Tools
        mà không tốn token API.
        """
        remap_votes, slip_votes, swapped_tile = 0, 0, False

        for tr in transitions:
            act = tr["action"]
            dr, dc = tr["delta_pos"]
            dhp = tr["delta_hp"]
            tile_landed = tr.get("tile_landed", EMPTY)

            exp_dr, exp_dc = ACTION_DELTAS[act]
            rem_dr, rem_dc = ACTION_DELTAS[REMAP_ACTION[act]]

            if (dr, dc) == (rem_dr, rem_dc) and (dr, dc) != (exp_dr, exp_dc):
                remap_votes += 1
            elif abs(dr) == 2 or abs(dc) == 2:
                slip_votes += 1

            if tile_landed == GREEN and dhp < 0:
                swapped_tile = True

        # Nếu mới chỉ thấy 1 bằng chứng yếu -> gọi probe_action để xác nhận chắc chắn
        if len(transitions) < 2 and remap_votes == 0 and not swapped_tile:
            return {
                "tool": "probe_action",
                "args": {
                    "action": "UP",
                    "reason": "Need another movement observation to distinguish a rule change from a slip.",
                },
            }

        movement_rule = "REMAP_90" if remap_votes > 0 else ("SLIPPERY" if slip_votes > 0 else "NORMAL")
        tile_semantics = "SWAPPED_GREEN_RED" if swapped_tile else "NORMAL"
        return {
            "tool": "commit_rule",
            "args": {
                "movement_rule": movement_rule,
                "tile_semantics": tile_semantics,
                "reasoning": f"Observed remap_votes={remap_votes}, slip_votes={slip_votes}, swapped_tile={swapped_tile}.",
            },
        }

    def _call_openai_compatible(self, user_prompt: str, img_bytes: bytes) -> Dict[str, Any]:
        """Gọi API chuẩn OpenAI/Gemini-OpenAI-compat với temperature=0.0 để tái lập."""
        if self.api_client is None:
            raise RuntimeError("Truyền api_client (vd: OpenAI()) khi dùng backend='openai_compat'.")

        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        resp = self.api_client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    ],
                },
            ],
        )
        return json.loads(resp.choices[0].message.content)


# ==============================================================================
# 4. DEMO CHẠY THỬ (RENDER + KEYFRAME STRIP + CACHE HIT)
# ==============================================================================
if __name__ == "__main__":
    # Tạo nhanh một bản đồ 7x7 mẫu
    grid = np.zeros((7, 7), dtype=np.int32)
    grid[2, 2] = WALL
    grid[3, 2] = GREEN
    grid[5, 5] = RED

    # Giả lập 1 bước dưới luật R4_COMPOUND:
    # Agent ở (3,3), bấm UP (0) -> bị đảo thành LEFT sang (3,2) là ô GREEN -> mất 25 HP!
    f0 = render_som_frame(grid, agent_pos=(3, 3), hp=100, step=5, last_action=None)
    f1 = render_som_frame(grid, agent_pos=(3, 2), hp=75,  step=6, last_action=0)
    strip = make_keyframe_strip([f0, f1], actions=[0])
    strip.save("demo_keyframe_strip.png")
    print("Đã lưu ảnh Keyframe Strip tại: demo_keyframe_strip.png")

    transitions = [
        {"step": 6, "action": 0, "delta_pos": [0, -1], "delta_hp": -25, "tile_landed": GREEN}
    ]

    reasoner = FrozenVLMReasoner(backend="mock", cache_path="vlm_cache.db")

    # Lần gọi 1: chưa có cache -> gọi inference
    out1 = reasoner.reason(strip, transitions)
    print("\nLần gọi 1 (Cache Miss):", json.dumps(out1, indent=2))

    # Lần gọi 2: cùng ảnh + transition -> đọc thẳng từ SQLite Disk Cache
    out2 = reasoner.reason(strip, transitions)
    print("\nLần gọi 2 (Cache Hit): ", json.dumps(out2, indent=2))
    print(f"\nThống kê -> API calls: {reasoner.api_calls} | Cache hits: {reasoner.cache_hits}")
