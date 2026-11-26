"""Rule-independent 224x224 RGB renderer for RuleShift-Grid.

:func:`render_grid` is deliberately handed only public geometry and public
episode state. It never receives the active rule, the shift schedule or any
noise draw, so the pixels cannot encode private information by construction.
The signpost is drawn as an illegible icon (a post, a board and two short
lines) rather than real text: the clause it stands for is public only through
the READ_SIGN tool, never through the frame.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

#: Default frame edge length. Always square and independent of grid size.
RENDER_SIZE = 224

#: Thin separators between cells. Tests assert that these exact pixels exist.
GRID_LINE_COLOR = (210, 210, 210)
#: Background of the bottom status bar.
HUD_COLOR = (30, 30, 30)

_MARGIN = 8
_LABEL_MARGIN = 14
_HUD_HEIGHT = 28

_WHITE = (255, 255, 255)
_FLOOR_COLOR = (244, 244, 244)
_WALL_COLOR = (105, 105, 105)
_LABEL_COLOR = (70, 70, 70)
_AGENT_COLOR = (40, 110, 200)
_AGENT_OUTLINE = (20, 60, 120)
_KEY_COLOR = (235, 190, 70)
_DOOR_COLOR = (140, 105, 70)
_POST_COLOR = (125, 125, 125)
_BOARD_COLOR = (255, 255, 255)
_MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (225, 90, 90),
    "blue": (90, 140, 225),
    "green": (105, 195, 110),
}


def _cell_box(x: int, y: int, cell: int, gx: int, gy: int) -> list[int]:
    left = gx + x * cell
    top = gy + y * cell
    return [left, top, left + cell - 1, top + cell - 1]


def _draw_key(draw: ImageDraw.ImageDraw, x: int, y: int, cell: int, gx: int, gy: int) -> None:
    left, top, right, bottom = _cell_box(x, y, cell, gx, gy)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    radius = max(1, cell // 4)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=_KEY_COLOR,
        outline=(150, 115, 40),
    )
    draw.rectangle([cx + radius - 1, cy - 1, right - 2, cy + 1], fill=(150, 115, 40))


def _draw_door(draw: ImageDraw.ImageDraw, x: int, y: int, cell: int, gx: int, gy: int) -> None:
    left, top, right, bottom = _cell_box(x, y, cell, gx, gy)
    inset = max(1, cell // 5)
    draw.rectangle(
        [left + inset, top + 1, right - inset, bottom - 1],
        fill=_DOOR_COLOR,
        outline=(90, 65, 40),
    )
    middle = (top + bottom) // 2
    draw.ellipse(
        [right - inset - 3, middle - 1, right - inset - 1, middle + 1],
        fill=(235, 220, 150),
    )


def _draw_signpost(
    draw: ImageDraw.ImageDraw, x: int, y: int, cell: int, gx: int, gy: int
) -> None:
    """A gray post with a white board and two short gray strokes (no text)."""

    left, top, right, bottom = _cell_box(x, y, cell, gx, gy)
    mid_x = (left + right) // 2
    board_top = top + 1
    board_bottom = top + max(2, (bottom - top) // 2)
    board_left = left + 1
    board_right = right - 1

    # The post runs from the board down to the floor.
    draw.line([mid_x, board_bottom, mid_x, bottom], fill=_POST_COLOR, width=max(1, cell // 8))
    draw.rectangle(
        [board_left, board_top, board_right, board_bottom],
        fill=_BOARD_COLOR,
        outline=_POST_COLOR,
    )
    stroke_y1 = board_top + max(1, (board_bottom - board_top) // 3)
    stroke_y2 = board_top + 2 * max(1, (board_bottom - board_top) // 3)
    stroke_left = board_left + 2
    stroke_right = max(stroke_left + 1, board_right - 2)
    draw.line([stroke_left, stroke_y1, stroke_right, stroke_y1], fill=_POST_COLOR, width=1)
    draw.line([stroke_left, stroke_y2, stroke_right, stroke_y2], fill=_POST_COLOR, width=1)


def render_grid(public_map: Any, state: Any, size: int = RENDER_SIZE) -> np.ndarray:
    """Render a deterministic RGB frame from public geometry and state only.

    No rule object, shift schedule or hidden draw is accepted as an argument,
    which is what makes identical public inputs always produce identical pixels
    regardless of which hidden rule happens to be active.
    """

    if size <= _MARGIN + _LABEL_MARGIN + _HUD_HEIGHT + _MARGIN:
        raise ValueError("size is too small for the renderer layout")

    width = int(public_map.width)
    height = int(public_map.height)
    grid_px = size - 2 * _MARGIN - _LABEL_MARGIN - _HUD_HEIGHT
    cell = grid_px // max(width, height)
    if cell < 1:
        raise ValueError("grid is too large for the renderer layout")

    gx = _MARGIN + _LABEL_MARGIN
    gy = _MARGIN + _LABEL_MARGIN
    grid_w = cell * width
    grid_h = cell * height

    image = Image.new("RGB", (size, size), _WHITE)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Floor, then walls, then the public markers/features.
    for y in range(height):
        for x in range(width):
            draw.rectangle(_cell_box(x, y, cell, gx, gy), fill=_FLOOR_COLOR)
    for wx, wy in sorted(public_map.walls):
        if 0 <= wx < width and 0 <= wy < height:
            draw.rectangle(_cell_box(wx, wy, cell, gx, gy), fill=_WALL_COLOR)
    for color, (mx, my) in sorted(public_map.markers.items()):
        if 0 <= mx < width and 0 <= my < height:
            draw.rectangle(
                _cell_box(mx, my, cell, gx, gy),
                fill=_MARKER_COLORS.get(color, (170, 170, 170)),
            )
    if public_map.key is not None:
        kx, ky = public_map.key
        if 0 <= kx < width and 0 <= ky < height:
            _draw_key(draw, kx, ky, cell, gx, gy)
    if public_map.door is not None:
        dx, dy = public_map.door
        if 0 <= dx < width and 0 <= dy < height:
            _draw_door(draw, dx, dy, cell, gx, gy)
    if public_map.signpost is not None:
        sx, sy = public_map.signpost
        if 0 <= sx < width and 0 <= sy < height:
            _draw_signpost(draw, sx, sy, cell, gx, gy)

    # Thin grid lines on top of the tiles so the separator color is exact.
    for column in range(width + 1):
        line_x = gx + column * cell
        draw.line([line_x, gy, line_x, gy + grid_h], fill=GRID_LINE_COLOR, width=1)
    for row in range(height + 1):
        line_y = gy + row * cell
        draw.line([gx, line_y, gx + grid_w, line_y], fill=GRID_LINE_COLOR, width=1)

    # Coordinate labels in the top and left margins.
    for column in range(width):
        draw.text(
            (gx + column * cell + cell / 2, _MARGIN + _LABEL_MARGIN / 2),
            str(column),
            fill=_LABEL_COLOR,
            font=font,
            anchor="mm",
        )
    for row in range(height):
        draw.text(
            (_MARGIN + _LABEL_MARGIN / 2, gy + row * cell + cell / 2),
            str(row),
            fill=_LABEL_COLOR,
            font=font,
            anchor="mm",
        )

    # Public agent position.
    ax, ay = state.position
    if 0 <= ax < width and 0 <= ay < height:
        left, top, right, bottom = _cell_box(ax, ay, cell, gx, gy)
        inset = max(1, cell // 6)
        draw.ellipse(
            [left + inset, top + inset, right - inset, bottom - inset],
            fill=_AGENT_COLOR,
            outline=_AGENT_OUTLINE,
        )

    # Bottom HUD: only public state, with no rule or schedule text.
    hud_y = size - _HUD_HEIGHT - _MARGIN
    draw.rectangle(
        [_MARGIN, hud_y, size - _MARGIN - 1, hud_y + _HUD_HEIGHT - 1],
        fill=HUD_COLOR,
    )
    hud_text = (
        f"HP={state.health} STEPS={state.steps} TOOL={state.tool_budget}"
    )
    draw.text(
        (_MARGIN + 4, hud_y + 7),
        hud_text,
        fill=_WHITE,
        font=font,
    )

    return np.asarray(image, dtype=np.uint8).copy()


def image_digest(image: np.ndarray) -> str:
    """Short stable digest of the raw frame bytes, never the pixels themselves."""

    return "sha256:" + hashlib.sha256(image.tobytes()).hexdigest()[:16]


__all__ = ["GRID_LINE_COLOR", "HUD_COLOR", "RENDER_SIZE", "image_digest", "render_grid"]
