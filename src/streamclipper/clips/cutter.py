"""Clip cutting with ffmpeg.

Planning is separated from execution: ``plan_cut`` is pure -- given the
segments on disk, the requested time range and the keyframe positions, it
decides which files to read, where to seek, and whether a stream copy is
acceptable. Only ``ClipCutter.cut`` touches ffmpeg. That split is what makes
the interesting decisions (keyframe snapping, padding, multi-segment spans)
testable without any media.

Two cut modes:

* **copy** -- no re-encode, so it is near-instant, but the cut can only begin
  on a keyframe. We snap *backwards* to the nearest preceding keyframe, never
  forwards, because a clip that starts slightly early keeps its opening words
  while one that starts late loses them.
* **reencode** -- frame accurate at the cost of encoding time.

``auto`` picks copy unless the backwards snap would drift more than
``snap_tolerance`` seconds, which is the case that would otherwise ship a
clip with several seconds of unrelated lead-in.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..config import ClipsConfig
from ..errors import ClipCutError
from ..logging_setup import get_logger
from ..state.models import Segment
from ..util.proc import require_binary, run
from .vertical import vertical_filter

log = get_logger(__name__)


@dataclass
class CutPlan:
    """Everything ffmpeg needs, decided before ffmpeg is involved."""

    sources: list[str] = field(default_factory=list)
    # Stream time at the first frame of the concatenated sources.
    timeline_start: float = 0.0
    # Seek offset into the concatenated sources, after keyframe snapping.
    seek: float = 0.0
    duration: float = 0.0
    mode: str = "copy"
    # Stream times the clip actually covers, padding and snapping included.
    actual_start: float = 0.0
    actual_end: float = 0.0
    # How far the keyframe snap moved the start. 0.0 when re-encoding.
    snap_drift: float = 0.0
    requested_start: float = 0.0
    requested_end: float = 0.0

    @property
    def needs_concat(self) -> bool:
        return len(self.sources) > 1


def snap_to_keyframe(
    keyframes: Sequence[float], target: float, tolerance: float
) -> tuple[float, float]:
    """Nearest keyframe at or before ``target``.

    Returns ``(snapped, drift)``. With no keyframe information, or nothing
    before the target, the target is returned unchanged and the drift is
    reported as infinite so ``auto`` mode falls through to re-encoding rather
    than silently cutting at a non-keyframe.
    """
    if not keyframes:
        return target, float("inf")

    preceding = [k for k in keyframes if k <= target + 1e-6]
    if not preceding:
        # Target sits before the first keyframe: starting there is exact.
        first = min(keyframes)
        if abs(first - target) <= tolerance:
            return first, abs(first - target)
        return target, float("inf")

    snapped = max(preceding)
    return snapped, target - snapped


def plan_cut(
    segments: Sequence[Segment],
    start: float,
    end: float,
    config: ClipsConfig,
    keyframes: Sequence[float] = (),
) -> CutPlan:
    """Decide how to cut ``[start, end]`` out of the recorded segments.

    ``keyframes`` are absolute stream times. Raises ClipCutError when no
    recorded segment covers the range -- typically because retention already
    reclaimed it.
    """
    if end <= start:
        raise ClipCutError(f"Clip end ({end}) must be after start ({start})")

    padded_start = max(0.0, start - config.pad_before)
    padded_end = end + config.pad_after

    covering = [s for s in segments if s.overlaps(padded_start, padded_end)]
    if not covering:
        raise ClipCutError(
            f"No recorded segment covers {padded_start:.1f}-{padded_end:.1f}s. "
            "The raw footage may have been deleted by the retention policy."
        )
    covering.sort(key=lambda s: s.start)

    # Clamp to what was actually recorded; a clip cannot extend past the tape.
    timeline_start = covering[0].start
    timeline_end = covering[-1].end
    padded_start = max(padded_start, timeline_start)
    padded_end = min(padded_end, timeline_end)
    if padded_end <= padded_start:
        raise ClipCutError(
            f"Clip range {start:.1f}-{end:.1f}s lies outside the recorded footage"
        )

    mode = config.mode
    seek = padded_start - timeline_start
    drift = 0.0

    if mode in {"copy", "auto"}:
        # Keyframes are absolute stream times; snap there, then convert.
        snapped_abs, drift = snap_to_keyframe(keyframes, padded_start, config.snap_tolerance)
        if mode == "auto" and drift > config.snap_tolerance:
            mode = "reencode"
            drift = 0.0
        else:
            mode = "copy"
            padded_start = max(timeline_start, snapped_abs)
            seek = padded_start - timeline_start
            # Infinite drift means there was nothing to snap to, so the start
            # was left where it was -- that is a drift of zero, not infinity.
            if drift == float("inf"):
                drift = 0.0

    return CutPlan(
        sources=[s.path for s in covering],
        timeline_start=timeline_start,
        seek=max(0.0, seek),
        duration=max(0.1, padded_end - padded_start),
        mode=mode,
        actual_start=padded_start,
        actual_end=padded_end,
        snap_drift=drift,
        requested_start=start,
        requested_end=end,
    )


def build_ffmpeg_args(
    plan: CutPlan,
    output: Path,
    config: ClipsConfig,
    concat_file: Path | None = None,
    subtitle_file: Path | None = None,
    vertical: bool = False,
) -> list[str]:
    """The ffmpeg command line for a plan.

    Input-side ``-ss`` (before ``-i``) makes the seek fast; on a copy that is
    exact because the plan already snapped to a keyframe.
    """
    args = [require_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y"]

    if plan.needs_concat:
        if concat_file is None:
            raise ClipCutError("A multi-segment plan needs a concat list file")
        args += ["-f", "concat", "-safe", "0", "-ss", f"{plan.seek:.3f}", "-i", str(concat_file)]
    else:
        args += ["-ss", f"{plan.seek:.3f}", "-i", plan.sources[0]]

    args += ["-t", f"{plan.duration:.3f}"]

    filters: list[str] = []
    if vertical:
        filters.append(vertical_filter(config.vertical))
    if subtitle_file is not None:
        escaped = str(subtitle_file).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        style = config.subtitle_style
        filters.append(
            f"subtitles='{escaped}'" + (f":force_style='{style}'" if style else "")
        )

    if filters or plan.mode == "reencode":
        # Any filter forces a re-encode; a copy cannot have its pixels touched.
        if filters:
            args += ["-vf", ",".join(filters)]
        args += [
            "-c:v", config.video_codec,
            "-crf", str(config.crf),
            "-preset", config.preset,
            "-c:a", config.audio_codec,
            # Widest player compatibility, and required for odd-sized crops.
            "-pix_fmt", "yuv420p",
        ]
    else:
        args += ["-c", "copy"]

    args += ["-movflags", "+faststart", "-avoid_negative_ts", "make_zero", str(output)]
    return args


class ClipCutter:
    """Executes cut plans."""

    def __init__(self, config: ClipsConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cut(
        self,
        plan: CutPlan,
        output: Path,
        subtitle_file: Path | None = None,
        vertical: bool = False,
    ) -> Path:
        """Run one plan. Returns the output path."""
        output.parent.mkdir(parents=True, exist_ok=True)
        concat_file: Path | None = None
        if plan.needs_concat:
            concat_file = output.with_suffix(".concat.txt")
            concat_file.write_text(
                "".join(
                    # Single quotes are the concat demuxer's escape; a path
                    # containing one would otherwise break the list.
                    "file '{}'\n".format(str(Path(src).resolve()).replace("'", r"'\''"))
                    for src in plan.sources
                ),
                encoding="utf-8",
            )

        args = build_ffmpeg_args(
            plan, output, self.config, concat_file, subtitle_file, vertical
        )
        try:
            run(args, timeout=1800)
        except subprocess.CalledProcessError as exc:
            raise ClipCutError(
                f"ffmpeg failed cutting {output.name}: {(exc.stderr or '')[-800:]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClipCutError(f"ffmpeg timed out cutting {output.name}") from exc
        finally:
            if concat_file is not None:
                concat_file.unlink(missing_ok=True)

        if not output.exists() or output.stat().st_size == 0:
            raise ClipCutError(f"ffmpeg produced no output for {output.name}")

        log.info(
            "clip.cut",
            extra={
                "output": output.name,
                "mode": plan.mode,
                "duration": round(plan.duration, 2),
                "snap_drift": round(plan.snap_drift, 3),
                "sources": len(plan.sources),
            },
        )
        return output
