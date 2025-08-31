#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/**
 * @file concatenate_videos.py
 * @brief Concatenate videos after normalizing them to a common target (max resolution, fps, etc).
 *
 * The script:
 *   1) Reads media characteristics using ffprobe for each input file.
 *   2) Chooses a target video profile:
 *        - Resolution = largest by pixel area among inputs.
 *        - FPS        = highest among inputs (keeps 30000/1001 form if present).
 *        - Pixel fmt  = yuv420p (safe widely supported default; override with --pix-fmt).
 *      Audio profile:
 *        - Sample rate  = highest among inputs (default cap 48000; override with --audio-rate).
 *        - Channels     = highest among inputs but capped to stereo by default; override with --audio-channels.
 *   3) Builds an ffmpeg filtergraph:
 *        - Optional per-input trim on video/audio.
 *        - Aspect-ratio safe scale + pad (default) or crop/stretch.
 *        - Monotonic audio PTS (asetpts=N/SR/TB) and resample to target.
 *        - Concat all normalized segments.
 *   4) Encodes:
 *        - GPU NVENC (hevc_nvenc) with modes: lossless | vbr | cqp.
 *        - CPU x265  (libx265)    with modes: lossless | crf.
 *
 * Example:
 *   python concat_normalize.py -o stitched_concat_nvenc.mkv \
 *     --use-gpu --video-quality vbr --cq 19 --b-v 30M --maxrate 60M \
 *     --inputs ../../stitched_output-with-audio.mp4 short_livebarn_stitched.mp4 ../../../dh-tv-12-2b/stitched_output-with-audio.mp4 \
 *     --clip "-00:20" --clip "00:00:05-00:10:00" --clip ""  # per-input trims (optional)
 *
 * Requirements:
 *   - ffmpeg and ffprobe available in PATH.
 *
 * @author
 * @date 2025-08-31
 */
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------- Data structures -----------------------------

@dataclass(frozen=True)
class VideoStreamInfo:
    """@brief Parsed video stream details from ffprobe.
    @param width Frame width in pixels.
    @param height Frame height in pixels.
    @param avg_frame_rate Rational string as reported by ffprobe (e.g. '30000/1001' or '24/1').
    @param fps_float Frame rate as float (derived from avg_frame_rate).
    @param pix_fmt Pixel format (e.g. 'yuv420p').
    @param bit_rate Video bit_rate in bits/second if reported (else 0).
    """
    width: int
    height: int
    avg_frame_rate: str
    fps_float: float
    pix_fmt: str
    bit_rate: int


@dataclass(frozen=True)
class AudioStreamInfo:
    """@brief Parsed audio stream details from ffprobe.
    @param sample_rate Sample rate (Hz).
    @param channels Number of audio channels.
    @param channel_layout Channel layout string if present (e.g. 'stereo', 'mono').
    @param bit_rate Audio bit_rate in bits/second if reported (else 0).
    """
    sample_rate: int
    channels: int
    channel_layout: Optional[str]
    bit_rate: int


@dataclass(frozen=True)
class MediaInfo:
    """@brief Combined per-file info including duration.
    @param path Filesystem path.
    @param duration Duration in seconds (float).
    @param v Video stream info if present.
    @param a Audio stream info if present.
    """
    path: Path
    duration: float
    v: Optional[VideoStreamInfo]
    a: Optional[AudioStreamInfo]


@dataclass(frozen=True)
class TargetProfile:
    """@brief Target audio/video normalization parameters.
    @param width Target width (px).
    @param height Target height (px).
    @param fps_rational FPS rational string (e.g., '30000/1001').
    @param pix_fmt Pixel format (e.g., 'yuv420p').
    @param audio_rate Audio sample rate (Hz).
    @param audio_channels Target number of channels (1 or 2 by default).
    """
    width: int
    height: int
    fps_rational: str
    pix_fmt: str
    audio_rate: int
    audio_channels: int


# ----------------------------- Utilities -----------------------------------

def require_binary(name: str) -> None:
    """@brief Ensure a binary exists in PATH.
    @param name Executable name.
    @throws SystemExit if not found.
    """
    if shutil.which(name) is None:
        sys.exit(f"Error: '{name}' not found in PATH.")


def ffprobe_json(path: Path) -> Dict[str, Any]:
    """@brief Run ffprobe and return parsed JSON metadata.
    @param path Input media path.
    @return ffprobe JSON as dict.
    @throws CalledProcessError on ffprobe failure.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    return json.loads(res.stdout)


def parse_fps_rational(r: str) -> float:
    """@brief Convert 'num/den' to float with safety.
    @param r Rational string like '30000/1001'.
    @return float fps.
    """
    try:
        num, den = r.split("/")
        n = int(num)
        d = int(den)
        if d == 0:
            return 0.0
        return n / d
    except Exception:
        # Sometimes ffprobe may give '0/0' or empty; return 0 as fallback.
        return 0.0


def best_rational_from_float(value: float, max_den: int = 1001) -> str:
    """@brief Approximate a float as a rational string.
    @param value Floating point value (e.g., 29.97).
    @param max_den Maximum denominator.
    @return Rational as 'num/den'.
    """
    if value <= 0:
        return "0/1"
    frac = Fraction(value).limit_denominator(max_den)
    return f"{frac.numerator}/{frac.denominator}"


def parse_time_to_seconds(s: str) -> float:
    """@brief Parse time string to seconds.
    Accepts 'SS', 'MM:SS', 'HH:MM:SS', with optional fractional seconds.
    @param s Time string.
    @return Seconds as float.
    @throws ValueError on bad format.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty time string")
    parts = s.split(":")
    parts_f = [float(p) for p in parts]
    if len(parts_f) == 1:
        return parts_f[0]
    if len(parts_f) == 2:
        mm, ss = parts_f
        return mm * 60 + ss
    if len(parts_f) == 3:
        hh, mm, ss = parts_f
        return hh * 3600 + mm * 60 + ss
    raise ValueError(f"Unrecognized time format: {s}")


def area(width: int, height: int) -> int:
    """@brief Compute pixel area."""
    return width * height


def pick_target_profile(
    infos: Sequence[MediaInfo],
    pix_fmt: str = "yuv420p",
    force_fps: Optional[str] = None,
    force_res: Optional[Tuple[int, int]] = None,
    max_audio_channels: int = 2,
    force_audio_rate: Optional[int] = None,
) -> TargetProfile:
    """@brief Decide target normalization parameters from inputs.
    @param infos MediaInfo list.
    @param pix_fmt Target pixel format (default yuv420p).
    @param force_fps Optional override FPS rational ('num/den' or float string).
    @param force_res Optional override resolution (width, height).
    @param max_audio_channels Cap for audio channels (default 2).
    @param force_audio_rate Optional override for sample rate (Hz).
    @return TargetProfile
    """
    # Defaults
    sel_w, sel_h = 1920, 1080
    sel_fps: str = "30/1"
    sel_arate = 48000
    sel_ach = 2

    # Accumulate best across inputs
    best_area = -1
    best_fps = 0.0
    best_fps_r = "0/1"
    best_arate = 0
    best_ach = 0

    for mi in infos:
        if mi.v:
            a_ = area(mi.v.width, mi.v.height)
            if a_ > best_area:
                best_area = a_
                sel_w, sel_h = mi.v.width, mi.v.height
            if mi.v.fps_float > best_fps:
                best_fps = mi.v.fps_float
                best_fps_r = mi.v.avg_frame_rate or best_rational_from_float(best_fps)

        if mi.a:
            if mi.a.sample_rate > best_arate:
                best_arate = mi.a.sample_rate
            if mi.a.channels > best_ach:
                best_ach = mi.a.channels

    # Apply discovered
    sel_fps = best_fps_r if best_fps_r != "0/1" else best_rational_from_float(max(best_fps, 30.0))
    sel_arate = best_arate if best_arate > 0 else 48000
    sel_ach = max(1, min(max(best_ach, 1), max_audio_channels))

    # Overrides
    if force_res is not None:
        sel_w, sel_h = force_res
    if force_fps is not None:
        # Accept either rational or float
        if "/" in force_fps:
            sel_fps = force_fps
        else:
            sel_fps = best_rational_from_float(float(force_fps))
    if force_audio_rate is not None:
        sel_arate = force_audio_rate

    return TargetProfile(
        width=sel_w, height=sel_h, fps_rational=sel_fps, pix_fmt=pix_fmt,
        audio_rate=sel_arate, audio_channels=sel_ach
    )


# ----------------------------- Probing -------------------------------------

def probe_media(path: Path) -> MediaInfo:
    """@brief Probe a media file with ffprobe.
    @param path File path.
    @return MediaInfo with optional streams populated.
    """
    meta = ffprobe_json(path)
    # Duration
    fmt = meta.get("format", {})
    duration = float(fmt.get("duration", 0.0))

    vinfo: Optional[VideoStreamInfo] = None
    ainfo: Optional[AudioStreamInfo] = None

    for st in meta.get("streams", []):
        codec_type = st.get("codec_type", "")
        if codec_type == "video" and vinfo is None:
            width = int(st.get("width", 0) or 0)
            height = int(st.get("height", 0) or 0)
            afr = st.get("avg_frame_rate", "") or st.get("r_frame_rate", "0/1")
            fps = parse_fps_rational(afr)
            pix_fmt = st.get("pix_fmt", "yuv420p") or "yuv420p"
            vbit = int(st.get("bit_rate", 0) or 0)
            vinfo = VideoStreamInfo(width, height, afr, fps, pix_fmt, vbit)

        elif codec_type == "audio" and ainfo is None:
            sr = int(st.get("sample_rate", 0) or 0)
            ch = int(st.get("channels", 0) or 0)
            cl = st.get("channel_layout")
            abit = int(st.get("bit_rate", 0) or 0)
            ainfo = AudioStreamInfo(sr, ch, cl, abit)

    return MediaInfo(path=path, duration=duration, v=vinfo, a=ainfo)


# ----------------------------- Filtergraph ----------------------------------

def build_filtergraph(
    infos: Sequence[MediaInfo],
    target: TargetProfile,
    clips: Sequence[Optional[str]],
    aspect_mode: str = "pad",
    use_gpu: bool = False,
    enforce_mono_to_stereo_only: bool = False,
) -> Tuple[str, List[str], List[str]]:
    """@brief Build the filter_complex string and output label maps.
    @param infos Probed media infos.
    @param target Target profile.
    @param clips Per-input clip spec 'START-END', '-END', 'START-' or '' for full. May be None/empty.
    @param aspect_mode One of: 'pad' (fit then pad), 'crop' (fill then crop), 'stretch' (exact resize, no AR keep).
    @param use_gpu If True, use 'scale_npp' else 'scale'.
    @param enforce_mono_to_stereo_only If True, only upmix mono->stereo; if multi-channel, downmix to stereo anyway. (Kept for future flexibility.)
    @return (filter_complex, v_labels, a_labels) to feed concat.
    """
    assert len(infos) == len(clips), "clips must have same length as inputs"

    parts: List[str] = []
    vlabels: List[str] = []
    alabels: List[str] = []

    w, h = target.width, target.height
    fps = target.fps_rational
    pixfmt = target.pix_fmt
    arate = target.audio_rate
    ach = target.audio_channels

    # Helper to format clip filters
    def parse_clip_spec(spec: Optional[str], total_dur: float) -> Tuple[Optional[float], Optional[float]]:
        if not spec:
            return (None, None)
        s = spec.strip()
        if not s:
            return (None, None)
        # Allowed forms: "start-end", "-end", "start-"
        if "-" in s:
            pre, post = s.split("-", 1)
            start = parse_time_to_seconds(pre) if pre.strip() else 0.0
            end = parse_time_to_seconds(post) if post.strip() else total_dur
            end = min(end, total_dur)
            if end < start:
                start, end = end, start  # swap to be safe
            return (start, end)
        else:
            # Single number means 'start-' from that time
            start = parse_time_to_seconds(s)
            return (start, None)

    # Choose scaling strategy
    if aspect_mode not in {"pad", "crop", "stretch"}:
        raise ValueError("--aspect-mode must be pad|crop|stretch")

    for idx, (mi, clip_spec) in enumerate(zip(infos, clips)):
        base_v = f"[{idx}:v]"
        base_a = f"[{idx}:a]"
        vchain = base_v
        achain: Optional[str] = base_a

        start_sec, end_sec = parse_clip_spec(clip_spec, mi.duration)
        # --- Video trim and reset PTS
        if start_sec is not None or end_sec is not None:
            # Use trim in filtergraph to keep A/V aligned
            if start_sec is None:
                start_sec = 0.0
            if end_sec is None:
                end_sec = mi.duration
            vchain += f"trim=start={start_sec}:end={end_sec},setpts=PTS-STARTPTS"

            # Audio chain may be absent if no audio stream
            if mi.a is not None:
                achain = base_a + f"atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS"
        else:
            # No trim, but keep original PTS as-is (we'll normalize later)
            vchain += "null"
            if mi.a is not None:
                achain = base_a + "anull"

        # --- Video normalize: scale + AR handling + fps + pix_fmt + sar
        if aspect_mode == "stretch":
            scale_f = "scale_npp" if use_gpu else "scale"
            vchain += f",{scale_f}={w}:{h}"
        elif aspect_mode == "pad":
            scale_f = "scale_npp" if use_gpu else "scale"
            # Fit inside, then pad around to exact size
            vchain += f",{scale_f}={w}:{h}:force_original_aspect_ratio=decrease"
            vchain += f",pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        else:  # crop
            # Fill (increase) then center-crop to exact size
            scale_f = "scale_npp" if use_gpu else "scale"
            vchain += f",{scale_f}={w}:{h}:force_original_aspect_ratio=increase"
            vchain += f",crop={w}:{h}:(iw-{w})/2:(ih-{h})/2"

        vchain += f",fps={fps},format={pixfmt},setsar=1"

        vout = f"[v{idx}]"
        vchain += vout
        parts.append(vchain)
        vlabels.append(vout)

        # --- Audio normalize (or synthesize silence if missing)
        if mi.a is None:
            # Synthesize silence equal to the (trimmed) video duration
            # Determine the segment duration for this input
            if start_sec is None:
                dur = mi.duration
            else:
                end_eff = end_sec if end_sec is not None else mi.duration
                dur = max(0.0, end_eff - start_sec)
            # Generate silent stereo at target rate for that duration
            achain = f"anullsrc=r={arate}:cl={'stereo' if ach >= 2 else 'mono'},atrim=0:{dur},asetpts=N/SR/TB[a{idx}]"
        else:
            # Resample to target, map to target layout; force monotonic pts
            # 'ocl' sets output channel layout (FFmpeg alias for out_channel_layout)
            target_layout = "stereo" if ach >= 2 else "mono"
            # Make sure achain exists (could be 'anull' or trimmed path from above)
            if "anull" in (achain or ""):
                # Replace 'anull' with the real chain
                achain = base_a
                if start_sec is not None or end_sec is not None:
                    # if we had a trim on video but 'anull' here due to no earlier trim,
                    # align audio trim as well for safety
                    s = start_sec or 0.0
                    e = end_sec if end_sec is not None else mi.duration
                    achain += f"atrim=start={s}:end={e},asetpts=PTS-STARTPTS"
            # Finally resample/upmix/downmix and ensure monotonic PTS
            achain += f",aresample={arate}:ocl={target_layout},asetpts=N/SR/TB[a{idx}]"

        parts.append(achain)
        alabels.append(f"[a{idx}]")

    # Concat
    concat = "".join(vlabels + alabels) + f"concat=n={len(infos)}:v=1:a=1[v][a]"
    parts.append(concat)

    filter_complex = ";".join(parts)
    return filter_complex, ["[v]"], ["[a]"]


# ----------------------------- Command builder ------------------------------

def build_ffmpeg_command(
    inputs: Sequence[Path],
    filter_complex: str,
    vmap: Sequence[str],
    amap: Sequence[str],
    target: TargetProfile,
    output: Path,
    use_gpu: bool,
    video_quality: str,
    preset: str,
    cq: Optional[int],
    b_v: Optional[str],
    maxrate: Optional[str],
    crf: Optional[int],
    audio_bitrate: str,
    overwrite: bool,
) -> List[str]:
    """@brief Compose the final ffmpeg CLI command.
    @param inputs Input paths.
    @param filter_complex Built filtergraph.
    @param vmap Video map labels.
    @param amap Audio map labels.
    @param target Target profile (used for audio settings here).
    @param output Output path.
    @param use_gpu Whether to use NVENC.
    @param video_quality One of {'lossless','vbr','cqp','crf'} (crf only valid for x265).
    @param preset Encoder preset ('p1'..'p7' for NVENC, 'ultrafast'..'placebo' for x265).
    @param cq Constant quality for NVENC VBR/CQP modes (optional).
    @param b_v Target average video bitrate (e.g., '30M') (optional).
    @param maxrate Max rate for NVENC VBR (e.g., '60M') (optional).
    @param crf CRF value for libx265 when video_quality=crf (optional).
    @param audio_bitrate Audio bitrate string (e.g., '192k').
    @param overwrite Allow overwrite (-y).
    @return List of CLI tokens for subprocess.
    """
    cmd: List[str] = ["ffmpeg"]
    # Enable CUDA hwaccel for decoding to save CPU when use_gpu is True
    if use_gpu:
        cmd += ["-hwaccel", "cuda"]
    # Safer timestamps for muxer
    cmd += ["-fflags", "+genpts"]

    # Inputs
    for p in inputs:
        cmd += ["-i", str(p)]

    # Filtergraph
    cmd += ["-filter_complex", filter_complex]

    # Map
    for m in vmap:
        cmd += ["-map", m]
    for m in amap:
        cmd += ["-map", m]

    # Video encoder
    if use_gpu:
        cmd += ["-c:v", "hevc_nvenc", "-preset", preset, "-profile:v", "main"]
        if video_quality == "lossless":
            cmd += ["-tune", "lossless", "-rc", "constqp", "-qp", "0"]
        elif video_quality == "cqp":
            qp_val = str(cq if cq is not None else 19)
            cmd += ["-rc", "constqp", "-qp", qp_val]
        elif video_quality == "vbr":
            cq_val = str(cq if cq is not None else 19)
            # If b_v / maxrate omitted, choose reasonable defaults from target size
            b = b_v if b_v else "30M"
            mr = maxrate if maxrate else "60M"
            cmd += ["-rc", "vbr", "-cq", cq_val, "-b:v", b, "-maxrate", mr]
        else:
            sys.exit("Invalid --video-quality for NVENC. Use lossless|vbr|cqp.")
    else:
        cmd += ["-c:v", "libx265", "-preset", preset]
        if video_quality == "lossless":
            cmd += ["-x265-params", "lossless=1"]
        elif video_quality == "crf":
            crf_val = str(crf if crf is not None else 22)
            cmd += ["-crf", crf_val]
        else:
            sys.exit("Invalid --video-quality for x265. Use lossless|crf.")

    # Audio encoder (AAC; change to flac+mkv if you need truly lossless audio)
    cmd += [
        "-c:a", "aac",
        "-ar", str(target.audio_rate),
        "-ac", str(target.audio_channels),
        "-b:a", audio_bitrate,
    ]

    # Faststart is meaningful for MP4
    if output.suffix.lower() == ".mp4":
        cmd += ["-movflags", "+faststart"]

    # Overwrite?
    if overwrite:
        cmd += ["-y"]

    cmd += [str(output)]
    return cmd


# ----------------------------- CLI -----------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """@brief CLI arguments parser.
    @param argv Optional args vector.
    @return Parsed Namespace.
    """
    p = argparse.ArgumentParser(
        description="Normalize (scale/fps/audio) and concatenate inputs to a single HEVC output."
    )
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input video files (2 or more)."
    )
    p.add_argument(
        "--clip", dest="clips", action="append", default=[],
        help="Optional per-input clip 'START-END', '-END', 'START-' or a single 'START'. "
             "Provide one --clip per input; omit or use empty string for full."
    )
    p.add_argument(
        "-o", "--output", required=True, help="Output file path (.mkv or .mp4)."
    )
    p.add_argument(
        "--use-gpu", action="store_true",
        help="Use NVIDIA NVENC (hevc_nvenc) and scale_npp."
    )
    p.add_argument(
        "--aspect-mode", choices=["pad", "crop", "stretch"], default="pad",
        help="How to fit AR to target resolution (default: pad)."
    )
    p.add_argument(
        "--pix-fmt", default="yuv420p",
        help="Target pixel format (default: yuv420p)."
    )
    p.add_argument(
        "--force-fps", default=None,
        help="Override target FPS (float or rational, e.g., 29.97 or 30000/1001)."
    )
    p.add_argument(
        "--force-res", default=None,
        help="Override resolution as WIDTHxHEIGHT (e.g., 8192x3052)."
    )
    p.add_argument(
        "--audio-rate", type=int, default=None,
        help="Override audio sample rate (Hz). If unset, use highest among inputs."
    )
    p.add_argument(
        "--audio-channels", type=int, default=2,
        help="Target audio channels (default 2 = stereo)."
    )
    p.add_argument(
        "--video-quality", choices=["lossless", "vbr", "cqp", "crf"], default="vbr",
        help="Quality mode (NVENC: lossless|vbr|cqp, x265: lossless|crf)."
    )
    p.add_argument(
        "--preset", default="p4",
        help="Encoder preset (NVENC p1..p7; x265 ultrafast..placebo)."
    )
    p.add_argument(
        "--cq", type=int, default=19, help="NVENC CQ/QP value for vbr/cqp modes."
    )
    p.add_argument(
        "--b-v", dest="b_v", default=None, help="NVENC target bitrate for vbr (e.g., 30M)."
    )
    p.add_argument(
        "--maxrate", default=None, help="NVENC maxrate for vbr (e.g., 60M)."
    )
    p.add_argument(
        "--crf", type=int, default=22, help="x265 CRF value for --video-quality=crf."
    )
    p.add_argument(
        "--audio-bitrate", default="192k", help="AAC bitrate (e.g., 128k, 192k)."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the ffmpeg command and exit."
    )
    p.add_argument(
        "-y", "--overwrite", action="store_true", help="Overwrite output if exists."
    )
    return p.parse_args(argv)


# ----------------------------- Main -----------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """@brief Program entry.
    @param argv Arg list.
    @return Exit code.
    """
    args = parse_args(argv)

    require_binary("ffmpeg")
    require_binary("ffprobe")

    inputs = [Path(s) for s in args.inputs]
    if len(inputs) < 2:
        sys.exit("Please provide at least two --inputs files.")

    # Align clips list length to inputs
    clips: List[Optional[str]] = []
    for i, path in enumerate(inputs):
        if i < len(args.clips):
            clips.append(args.clips[i])
        else:
            clips.append(None)

    # Probe all inputs
    infos: List[MediaInfo] = [probe_media(p) for p in inputs]

    # Optional forced resolution
    force_res_tuple: Optional[Tuple[int, int]] = None
    if args.force_res:
        try:
            w_s, h_s = args.force_res.lower().split("x")
            force_res_tuple = (int(w_s), int(h_s))
        except Exception as e:
            sys.exit(f"--force-res expects WIDTHxHEIGHT, e.g., 8192x3052; got: {args.force_res}")

    # Decide target
    target = pick_target_profile(
        infos,
        pix_fmt=args.pix_fmt,
        force_fps=args.force_fps,
        force_res=force_res_tuple,
        max_audio_channels=max(1, args.audio_channels),
        force_audio_rate=args.audio_rate,
    )

    # Build filtergraph
    filter_complex, vmap, amap = build_filtergraph(
        infos=infos,
        target=target,
        clips=clips,
        aspect_mode=args.aspect_mode,
        use_gpu=bool(args.use_gpu),
    )

    # Compose command
    output = Path(args.output)
    cmd = build_ffmpeg_command(
        inputs=inputs,
        filter_complex=filter_complex,
        vmap=vmap,
        amap=amap,
        target=target,
        output=output,
        use_gpu=bool(args.use_gpu),
        video_quality=args.video_quality,
        preset=args.preset,
        cq=args.cq,
        b_v=args.b_v,
        maxrate=args.maxrate,
        crf=args.crf,
        audio_bitrate=args.audio_bitrate,
        overwrite=bool(args.overwrite),
    )

    # Display helpful summary
    print("=== Target Profile ===")
    print(f"  Resolution: {target.width}x{target.height}")
    print(f"  FPS:        {target.fps_rational}")
    print(f"  Pixel fmt:  {target.pix_fmt}")
    print(f"  Audio:      {target.audio_rate} Hz, {target.audio_channels} ch")
    print("=== Filtergraph ===")
    print(filter_complex)
    print("=== ffmpeg Command ===")
    print(" ".join(shlex_quote(t) for t in cmd))

    if args.dry_run:
        return 0

    # Run ffmpeg
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr if isinstance(e.stderr, str) else "")
        sys.exit(e.returncode)

    return 0


# ----------------------------- Helpers --------------------------------------

def shlex_quote(s: str) -> str:
    """@brief Minimal shell-quote for display purposes only.
    @param s String to quote.
    @return Quoted string for pretty-printing.
    """
    # Simple safe quote (not using shlex because of cross-platform display)
    if not s or any(c in s for c in ' \t\n"\'`$&|;<>(){}[]*?'):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


if __name__ == "__main__":
    raise SystemExit(main())
