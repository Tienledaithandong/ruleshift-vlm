"""CPU-only visual decision cards for the VLM probe-versus-commit benchmark.

The card is self-contained: map, candidate-rule descriptions, prior/posterior
bars, observed evidence, remaining budget, and tool costs are all drawn from
public fields. No model is loaded and no inference happens here.

Two controls are intentionally separate:

* nuisance: the same public facts in a different palette and panel border. It
  must not change the reference label or the evidence ledger.
* diagnostic: a semantic perturbation, currently a different observed blue
  role, with a freshly conditioned posterior and a fresh reference label. It is
  a paired decision context, not a cosmetic edit.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ruleshift.envs.grid.render import RENDER_SIZE, image_digest, render_grid
from ruleshift.evaluator.vlm_decision import (
    DecisionContext,
    Evidence,
    ROLES,
)
from ruleshift.public_api.grid_rules import RULE_BY_ID

CARD_WIDTH = 512
CARD_HEIGHT = 320
MAP_SIZE = RENDER_SIZE
PANEL_X = MAP_SIZE
PANEL_WIDTH = CARD_WIDTH - PANEL_X

_PALETTES: dict[str, dict[str, Any]] = {
    "default": {
        "background": (246, 244, 238),
        "panel": (28, 31, 36),
        "panel_text": (238, 238, 232),
        "muted": (168, 174, 182),
        "border": (92, 100, 112),
        "accent": (250, 196, 82),
        "bar": (90, 150, 230),
    },
    "nuisance": {
        "background": (236, 241, 245),
        "panel": (36, 29, 29),
        "panel_text": (245, 232, 220),
        "muted": (188, 176, 164),
        "border": (112, 96, 84),
        "accent": (184, 226, 164),
        "bar": (220, 142, 92),
    },
}

_ROLE_GLYPHS = {"goal": "G", "hazard": "H", "safe": "S"}


@dataclass(frozen=True)
class _RenderState:
    position: tuple[int, int]
    health: int = 3
    steps: int = 0
    tool_budget: int = 0


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _wrap(text: str, width: int = 36) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def _draw_sign_icon(
    draw: ImageDraw.ImageDraw,
    context: DecisionContext,
    x: int,
    y: int,
    palette: dict[str, Any],
) -> None:
    draw.rectangle([x, y + 6, x + 4, y + 28], fill=(150, 150, 150))
    draw.rectangle([x + 2, y, x + 26, y + 14], fill=(250, 250, 250), outline=(90, 90, 90))
    draw.line([x + 6, y + 5, x + 22, y + 5], fill=(120, 120, 120))
    draw.line([x + 6, y + 9, x + 22, y + 9], fill=(120, 120, 120))
    sign = [item for item in context.evidence if item.kind == "sign"]
    if sign:
        item = sign[0]
        label = "R NOT H" if "hazard" in item.value else "R NOT G"
        draw.text((x + 32, y + 1), f"S{item.side} {label}", fill=palette["panel_text"], font=_font())
    else:
        draw.text((x + 32, y + 1), "SIGN ?", fill=palette["muted"], font=_font())


def _draw_blue_icon(
    draw: ImageDraw.ImageDraw,
    context: DecisionContext,
    x: int,
    y: int,
    palette: dict[str, Any],
) -> None:
    draw.rectangle([x, y, x + 30, y + 26], fill=(78, 130, 214), outline=(40, 75, 130))
    observed = [item for item in context.evidence if item.kind == "blue"]
    if observed:
        glyph = _ROLE_GLYPHS[observed[0].value]
        draw.text((x + 11, y + 6), glyph, fill=(255, 255, 255), font=_font())
        draw.text(
            (x + 36, y + 7),
            "BLUE " + glyph,
            fill=palette["panel_text"],
            font=_font(),
        )
    else:
        draw.text((x + 11, y + 6), "?", fill=(255, 255, 255), font=_font())
        draw.text((x + 36, y + 7), "BLUE ?", fill=palette["muted"], font=_font())


def _draw_budget(
    draw: ImageDraw.ImageDraw,
    context: DecisionContext,
    x: int,
    y: int,
    palette: dict[str, Any],
) -> None:
    draw.text((x, y), "TOOL BUDGET", fill=palette["muted"], font=_font())
    total = max(context.tool_budget, 0)
    left = max(context.remaining_budget, 0)
    for index in range(total):
        cx = x + index * 12
        filled = index < left
        draw.ellipse(
            [cx, y + 14, cx + 8, y + 22],
            fill=palette["accent"] if filled else palette["panel"],
            outline=palette["accent"] if filled else palette["border"],
        )
    draw.text(
        (x, y + 30),
        f"COSTS S{context.costs.sign} B{context.costs.blue}",
        fill=palette["panel_text"],
        font=_font(),
    )
    draw.text(
        (x, y + 42),
        f"LEFT {left}/{total}",
        fill=palette["muted"],
        font=_font(),
    )
    draw.text(
        (x, y + 54),
        f"ERR {context.inspect_error:.2f}",
        fill=palette["muted"],
        font=_font(),
    )
    draw.text(
        (x, y + 66),
        f"HORIZON {context.max_probes}",
        fill=palette["muted"],
        font=_font(),
    )


def _draw_rule_panel(
    draw: ImageDraw.ImageDraw,
    context: DecisionContext,
    palette: dict[str, Any],
) -> None:
    font = _font()
    draw.rectangle(
        [PANEL_X, 0, CARD_WIDTH - 1, CARD_HEIGHT - 1],
        fill=palette["panel"],
        outline=palette["border"],
    )
    draw.text((PANEL_X + 12, 10), "CANDIDATE RULES", fill=palette["accent"], font=font)
    draw.text(
        (PANEL_X + 12, 28),
        "bar = posterior   dot = prior",
        fill=palette["muted"],
        font=font,
    )
    top = 46
    for index, rule_id in enumerate(context.candidate_rules):
        rule = RULE_BY_ID[rule_id]
        probability = float(context.posterior[index])
        prior = float(context.prior[index])
        y = top + index * 42
        draw.text((PANEL_X + 12, y), rule_id, fill=palette["panel_text"], font=font)
        draw.text(
            (PANEL_X + 146, y),
            f"p={probability:.2f}",
            fill=palette["panel_text"],
            font=font,
        )
        lines = _wrap(rule.description, 40)
        for line_index, line in enumerate(lines[:2]):
            draw.text(
                (PANEL_X + 12, y + 12 + line_index * 10),
                line,
                fill=palette["muted"],
                font=font,
            )
        bar_y = y + 33
        draw.rectangle(
            [PANEL_X + 12, bar_y, PANEL_X + 260, bar_y + 5],
            fill=palette["background"],
            outline=palette["border"],
        )
        draw.rectangle(
            [PANEL_X + 12, bar_y, PANEL_X + 12 + int(248 * probability), bar_y + 5],
            fill=palette["bar"],
        )
        prior_x = PANEL_X + 12 + int(248 * prior)
        draw.ellipse(
            [prior_x - 2, bar_y - 1, prior_x + 2, bar_y + 4],
            fill=palette["accent"],
            outline=palette["accent"],
        )


def render_decision_card(
    context: DecisionContext,
    *,
    variant: str = "base",
) -> np.ndarray:
    """Render a deterministic self-contained RGB card."""

    if variant not in _PALETTES and variant != "base":
        raise ValueError(f"unknown visual variant: {variant}")
    palette = _PALETTES["default" if variant == "base" else variant]
    state = _RenderState(
        position=context.agent_position,
        health=3,
        steps=0,
        tool_budget=context.remaining_budget,
    )
    frame = render_grid(context.public_map, state, size=MAP_SIZE)
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), palette["background"])
    image.paste(Image.fromarray(frame, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [0, MAP_SIZE, MAP_SIZE - 1, CARD_HEIGHT - 1],
        fill=palette["panel"],
        outline=palette["border"],
    )
    _draw_sign_icon(draw, context, 12, MAP_SIZE + 12, palette)
    _draw_blue_icon(draw, context, 12, MAP_SIZE + 48, palette)
    _draw_budget(draw, context, 148, MAP_SIZE + 12, palette)
    _draw_rule_panel(draw, context, palette)
    return np.asarray(image, dtype=np.uint8).copy()


def visual_digest(image: np.ndarray) -> str:
    return image_digest(image)


def blue_role_perturbation(context: DecisionContext, role: str) -> DecisionContext:
    """Return a semantic blue-role variant with a fresh exact posterior.

    This helper is intentionally in the visual module only for the paired
    diagnostic control. It never adds hidden information to the card.
    """

    if role not in ROLES:
        raise ValueError("role must be goal/hazard/safe")
    evidence = [item for item in context.evidence if item.kind != "blue"]
    evidence.append(Evidence("blue", role))
    costs = context.costs
    spent = sum(item.cost(costs) for item in evidence)
    if spent > context.tool_budget:
        raise ValueError("perturbed blue evidence exceeds the tool budget")
    from ruleshift.evaluator.vlm_decision import _posterior_for

    posterior = _posterior_for(context.prior, evidence, context.inspect_error)
    if posterior is None:
        raise ValueError("perturbed blue evidence is impossible under the prior")
    from dataclasses import replace

    return replace(
        context,
        context_id=context.context_id + "-diag-blue-" + role,
        evidence=tuple(evidence),
        posterior=posterior,
        stratum="diagnostic_blue",
    )
