"""Display settings shared by static plots, first-person rendering and browser views.

Edit default_theme.json or call apply_theme(load_theme(path)) with a partial JSON
file. Numerical simulation modules do not import this module.
"""
from __future__ import annotations
from copy import deepcopy
from functools import wraps
import json
from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parent
_DEFAULT = json.loads((_ROOT / "default_theme.json").read_text(encoding="utf-8"))
THEME = deepcopy(_DEFAULT)
PALETTE = THEME["palette"]
COLORS = THEME["colors"]
MATPLOTLIB = THEME["matplotlib"]
SCENARIO_COLORS = THEME["scenario_colors"]
_COLOR_KEYS = {
    "#243447": "text",
    "#536478": "muted",
    "#68788B": "subtle",
    "#FFFFFF": "background",
    "#F5F7FA": "page",
    "#DCE3EB": "border",
    "#E2E7EC": "grid",
    "#C5CED8": "zero",
    "#475569": "axis",
    "#BCC8D6": "control_border",
    "#0072B2": "accent",
    "#E5E7EB": "missing",
    "#B8BEC6": "ground",
    "#202B38": "sensor",
    "#e6550d": "obstacle",
    "#31a354": "free_path",
    "#6baed6": "ground_points",
    "#9ecae1": "pale_ground",
    "gold": "camera_fov",
    "yellow": "heading",
    "#00c000": "goal_direction",
    "purple": "range_circle",
    "#3182bd": "speed",
    "red": "collision",
    "black": "black",
    "white": "white",
    "0.35": "object_fill",
    "#333": "object_line",
    "rgba(80,80,80,0.85)": "object_fill_web",
    "rgba(49,163,84,0.55)": "free_path_web",
    "rgba(0,0,0,0.5)": "overlay",
    "orange": "planner_clearance",
    "#E6E8EB": "figure_grid",
    "#222222": "figure_text",
    "#666666": "figure_axis",
    "#444444": "figure_tick",
    "#555555": "figure_muted",
    "#111111": "figure_sensor",
    "#F1F4F7": "angular_background",
    "#64748B": "empty_text",
    "k": "black",
    "w": "white",
    "r": "collision",
    "y": "heading",
    "g": "free_path",
    "#333333": "object_line",
    "#000000": "black",
    "#ffffff": "background",
    "#D55E00": "color_d55e00",
    "#009E73": "color_009e73",
    "#CC79A7": "color_cc79a7",
    "#E69F00": "color_e69f00",
    "#56B4E9": "color_56b4e9",
    "#6B4C9A": "color_6b4c9a",
    "#1f77b4": "color_1f77b4",
    "#2ca02c": "color_2ca02c",
    "#d62728": "color_d62728",
}


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_theme(path=None):
    """Return defaults merged with an optional partial JSON theme file."""
    result = deepcopy(_DEFAULT)
    if path is not None:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("a visualization theme must be a JSON object")
        _merge(result, overrides)
    return result


def apply_theme(settings):
    """Apply display settings, preserving references already imported by renderers."""
    result = _merge(deepcopy(_DEFAULT), settings)
    if not result["palette"]:
        raise ValueError("the visualization palette must contain at least one color")
    PALETTE[:] = result["palette"]
    for name, target in (
        ("colors", COLORS),
        ("matplotlib", MATPLOTLIB),
        ("scenario_colors", SCENARIO_COLORS),
    ):
        target.clear()
        target.update(result[name])
        result[name] = target
    result["palette"] = PALETTE
    THEME.clear()
    THEME.update(result)
    return THEME


def color(default):
    """Resolve an existing renderer color through the shared semantic palette."""
    return COLORS.get(_COLOR_KEYS.get(default, default), default)


def colormap(name):
    return THEME["colormaps"][name]


def mpl_style(func):
    @wraps(func)
    def render(*args, **kwargs):
        import matplotlib.pyplot as plt

        with plt.rc_context(MATPLOTLIB):
            return func(*args, **kwargs)

    return render


def asset(name):
    """Read a visualization-owned template or static asset."""
    return (_ROOT / name).read_text(encoding="utf-8")


def stylesheet(name):
    text = asset("static/" + name)
    # Only styling tokens are replaced; numerical inputs and viewer data are untouched.
    pattern = r"#[0-9A-Fa-f]{6}\b|rgba\([^)]*\)|(?<![\w-])(?:white|black|gold|purple|orange)(?![\w-])"
    text = re.sub(pattern, lambda match: color(match.group(0)), text)
    text = text.replace("Arial,Helvetica,sans-serif", THEME["web"]["font_family"])
    text = text.replace("Arial,sans-serif", THEME["web"]["font_family"])
    text = text.replace("system-ui,sans-serif", THEME["web"]["live_font_family"])
    return text


def web_settings():
    # Explicit stops also support scales provided by Plotly.py that are not
    # registered by name in the JavaScript bundle.
    from plotly.colors import get_colorscale

    maps = {}
    for key, value in THEME["colormaps"].items():
        maps[key] = get_colorscale(value) if isinstance(value, str) else value
    return {
        "colors": dict(COLORS),
        "colormaps": maps,
        "occupancy_reverse": False,
        "font_family": THEME["web"]["live_font_family"],
    }
