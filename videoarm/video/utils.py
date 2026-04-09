"""
Video frame extraction utilities for VideoARM.
"""

import uuid
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from videoarm.config.settings import TEMP_DIR


def _resize_to_short_side(frame, target_short_side: int = 256):
    """Scale a frame so its shorter dimension equals target_short_side."""
    if target_short_side is None or target_short_side <= 0:
        return frame
    h, w = frame.shape[:2]
    short = min(h, w)
    if short <= target_short_side:
        return frame
    scale = target_short_side / float(short)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_cropped_frame_paths(
    video_path: Union[str, Path],
    start_frame: Optional[int],
    end_frame: Optional[int],
    num_frames: int,
    session_id: Optional[str] = None,
    target_short_side: int = 256,
    silent: bool = False,
) -> List[str]:
    """
    Uniformly sample ``num_frames`` frames from [start_frame, end_frame] and
    save them as JPEG files under ``tmp/<session_id>/``.

    Each frame has its global frame index overlaid in white at the top-left.

    Args:
        video_path:        Path to the source video file.
        start_frame:       First frame index (inclusive); None → 0.
        end_frame:         Last frame index (inclusive); None → last frame.
        num_frames:        Number of frames to extract.
        session_id:        Sub-directory tag inside tmp/; auto-generated if None.
        target_short_side: Resize shorter edge to this many pixels.

    Returns:
        List of absolute paths to the saved JPEG files.
    """
    video_path = Path(video_path)
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    # Resolve frame boundaries
    if start_frame is None or end_frame is None:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        start_frame = start_frame or 0
        end_frame = end_frame or (total - 1)

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid frame range.")

    available = end_frame - start_frame + 1
    actual = min(num_frames, available)
    if actual < num_frames:
        print(
            f"Requested {num_frames} frames but only {available} available "
            f"in [{start_frame}, {end_frame}]; using {actual}."
        )

    if actual == 1:
        indices = [start_frame]
    else:
        indices = [
            int(start_frame + (end_frame - start_frame) * i / (actual - 1))
            for i in range(actual)
        ]

    output_dir = TEMP_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: List[str] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 256
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 256

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((fh, fw, 3), dtype=np.uint8)

        # Overlay global frame index (white text, top-left)
        cv2.putText(frame, str(idx), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = _resize_to_short_side(frame, target_short_side)
        out_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        frame_paths.append(str(out_path.resolve()))

    cap.release()

    for p in frame_paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Frame file not found after extraction: {p}")

    if not silent:
        print(f"│  Frames : {len(frame_paths)} frames extracted [{start_frame}-{end_frame}]")
    return frame_paths
