"""9:16 vertical crop for shorts."""

from __future__ import annotations

from ..config import VerticalConfig


def vertical_filter(config: VerticalConfig) -> str:
    """ffmpeg filter chain cropping a landscape frame to a vertical one.

    Crops to the target aspect first, then scales, so the output keeps full
    resolution instead of being upscaled from a letterboxed frame. ``focus_x``
    slides the crop window horizontally -- a streamer is rarely centred, and a
    dead-centre crop often cuts them in half.
    """
    aspect = config.width / config.height
    focus = min(1.0, max(0.0, config.focus_x))
    # Crop the tallest window with the target aspect that fits, positioned by
    # focus_x across the leftover width. Expressions are evaluated by ffmpeg
    # against the real input dimensions.
    crop_w = f"min(iw\\,ih*{aspect:.6f})"
    crop_x = f"(iw-{crop_w})*{focus:.4f}"
    return (
        f"crop={crop_w}:ih:{crop_x}:0,"
        f"scale={config.width}:{config.height}:force_original_aspect_ratio=increase,"
        f"crop={config.width}:{config.height}"
    )
