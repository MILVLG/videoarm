"""
VideoARMAgent — coarse-to-fine video reasoning agent.

Implements the observe → think → act → memorize loop described in the paper:

  "VideoARM: Agentic Reasoning-over-Hierarchical-Memory
   for Long-Form Video Understanding"
  https://arxiv.org/abs/2512.12360

Architecture overview
---------------------
The agent maintains a **Hierarchical and Multimodal Memory (HM³)** with three
tiers that are injected as a JSON block into every controller prompt:

  hm3["scene_snapshots"]   — long-term perception pool entries
                              (frame intervals + captions from Scene Snapper)
  hm3["audio_transcripts"] — transcript records from Audio Transcriber
  hm3["clip_analyses"]     — fine-grained QA records from Clip Analyzer

Three tools are available to the controller:

  scene_snapper      (Multimodal Understanding Tool)
      Navigates to user-specified frame ranges, samples frames, and generates
      a scene caption V_C.  Updates the long-term perception pool P_l.

  audio_transcriber  (Multimodal Understanding Tool)
      Extracts and transcribes audio from specified frame ranges via whisper-1.
      Provides transcript A_C complementing visual signals.

  clip_analyzer      (Multimodal Understanding Tool)
      Uniformly samples frames from a local interval and answers a sub-question
      Q_sub, returning answer A_sub and confidence score S_sub.
      Enables fine-grained spatial/temporal verification.

Message lifecycle
-----------------
After each tool call the conversation is *rebuilt* from scratch:
  [System] + [User: video info + HM³ JSON + question]

This keeps the context window lean while preserving all accumulated evidence
in the HM³ JSON.
"""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from videoarm.api.client import call_openai_model_with_tools, handle_json_parsing_error
from videoarm.config.model_config import get_config
from videoarm.config.settings import RESULTS_DIR, TEMP_DIR
from videoarm.video.utils import get_cropped_frame_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _warn(msg: str) -> None:
    """Print a yellow warning line with │ prefix."""
    print(f"{_YELLOW}│ ⚠  {msg}{_RESET}")


def _err(msg: str) -> None:
    """Print a red error line with │ prefix."""
    print(f"{_RED}│ ✗  {msg}{_RESET}")


def _is_content_filter_error(message: str) -> bool:
    """
    Return True only for genuine content-policy rejections
    (not transient overload / rate-limit errors).
    """
    text = message.lower()
    if any(k in text for k in ("rate limit", "retry", "overloaded")):
        return False
    return any(
        k in text
        for k in (
            "content management policy",
            "content filter",
            "content_filter",
            "safety system",
            "rejected as a result of our safety",
        )
    )


# ---------------------------------------------------------------------------
# VideoARMAgent
# ---------------------------------------------------------------------------


class VideoARMAgent:
    """
    Coarse-to-fine video reasoning agent with hierarchical multimodal memory.

    Usage::

        agent = VideoARMAgent()
        answer = agent.ask("video.mp4", "What shark species appears in the video?")

    For multiple-choice questions::

        answer = agent.ask(
            "video.mp4",
            "A. Hammerhead  B. Bull shark  C. Tiger shark  D. Mako",
            is_multiple_choice=True,
        )
    """

    def __init__(self, model_name: str = None) -> None:
        self.config = get_config()
        self.model_name = model_name or self.config.get_model("controller")
        # Runtime state (reset per question)
        self.video_info: Dict[str, Any] = {}
        self.session_id: str = ""
        self.conversation_history: List[Dict] = []
        self.tool_calls_log: List[Dict] = []
        self.analysis_iterations_used: int = 0
        self.video_has_audio: bool = True
        self.current_question_hash: Optional[str] = None

        # HM³ — Hierarchical and Multimodal Memory
        # Populated during the reasoning loop; injected into every controller turn.
        self.hm3: Dict[str, List] = self._empty_hm3()

        # Tool definitions (JSON schema for function calling)
        self.tools_registry = self._build_tools_registry()

        # Retry parameters
        self.tool_retry_attempts = (
            self.config.get_pipeline_config("tool_retry_attempts") or 2
        )
        self.tool_retry_delay_base = (
            self.config.get_pipeline_config("tool_retry_delay_base") or 2
        )

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        video_path: str,
        question: str,
        is_multiple_choice: bool = False,
        choice_format: str = "letter",  # "letter" → A/B/C/D, "number" → 0/1/2/3/4
        save_result: bool = True,
    ) -> str:
        """
        Answer a question about a video using adaptive coarse-to-fine reasoning.

        Args:
            video_path:          Path to a local video file.
            question:            The question to answer.
            is_multiple_choice:  If True, the answer is a single option letter/number.
            choice_format:       "letter" (A-D) or "number" (0-4).
            save_result:         Whether to persist the QA trace to disk.

        Returns:
            The agent's answer as a string.
        """
        print("=" * 60)
        print("VideoARM Agent")
        print("=" * 60)
        print(f"  Video    : {video_path}")
        print(f"  Question : {question}")
        print(f"  Model    : {self.model_name}")

        start = time.time()

        self._initialize_video(video_path)
        self.session_id = str(int(time.time() * 1000))[-8:]
        self.hm3 = self._empty_hm3()
        self.current_question_hash = hashlib.md5(
            question.encode("utf-8")
        ).hexdigest()[:12]

        answer = self._reasoning_loop(
            video_path, question, is_multiple_choice, choice_format
        )

        elapsed = time.time() - start
        print("=" * 60)
        print(f"Done in {elapsed:.1f}s  |  Answer: {answer}")
        print("=" * 60)
        if save_result:
            self._save_qa_result(video_path, question, answer, elapsed)
        self._cleanup_temp_frames()
        return answer

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_hm3() -> Dict[str, List]:
        return {
            # Long-term perception pool entries (frame interval + scene caption)
            "scene_snapshots": [],
            # Audio transcript records
            "audio_transcripts": [],
            # Clip analysis records (sub-question + answer + confidence)
            "clip_analyses": [],
        }

    def _initialize_video(self, video_path: str) -> None:
        """Read basic video metadata."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.video_info = {
            "path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration": total_frames / fps if fps > 0 else 0.0,
        }
        self.video_has_audio = True
        print(f"  Duration : {self.video_info['duration']:.1f}s  |  "
              f"{total_frames} frames  |  {fps:.1f} fps")
        print("-" * 60)

    # ------------------------------------------------------------------ #
    # Core reasoning loop                                                  #
    # ------------------------------------------------------------------ #

    def _reasoning_loop(
        self,
        video_path: str,
        question: str,
        is_multiple_choice: bool,
        choice_format: str,
    ) -> str:
        """
        Main observe → think → act → memorize loop.

        The controller iterates up to N steps (step budget).  On each step:
          1. Observe  — reads HM³ (injected as JSON in the user message)
          2. Think    — generates a reasoning trace R_t
          3. Act      — invokes one tool with parameters P_t
          4. Memorize — appends tool output O_t to HM³; rebuilds messages
        """
        system_prompt = (
            "You are a helpful assistant who answers multi-step questions by "
            "sequentially invoking functions. Follow the OBSERVE → THINK → ACT → "
            "MEMORIZE loop:\n"
            "  • OBSERVE  Carefully read the Current Memory (HM³) JSON.\n"
            "  • THINK    Reason step-by-step about which function to call next.\n"
            "  • ACT      Call exactly one function that moves you closer to the answer.\n"
            "  • MEMORIZE The system updates the memory automatically after each call.\n"
            "Plan extensively before each call and reflect on every result.  Do not "
            "guess — use the tools to gather evidence.  Give the final answer only when "
            "you are confident.\n"
            "Each extracted frame displays the global frame index in white text at the "
            "top-left.  Each picture is a 3×2 mosaic of 6 frames, row-major."
        )

        messages = self._build_initial_messages(
            system_prompt, question, is_multiple_choice, choice_format
        )
        self.conversation_history = list(messages)
        self.tool_calls_log = []

        cfg = self.config.get_pipeline_config()
        max_iterations = cfg["max_iterations"]
        max_retries = cfg["api_retry_attempts"]
        retry_delay_base = cfg["api_retry_delay_base"]

        iteration = 0
        self.analysis_iterations_used = 0
        start_time = time.time()

        while iteration < max_iterations:
            iteration += 1
            self.analysis_iterations_used = iteration
            print(f"\n┌─ Iteration {iteration}/{max_iterations} "
                  f"({time.time() - start_time:.1f}s elapsed) ─────────────")

            api_key, base_url = self.config.get_api_config("controller")
            params = self.config.get_model_params("controller")
            tool_choice = "required" if iteration == 1 else "auto"

            # --- Call controller model with retries ---
            response = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = self._call_controller(
                        messages, params, api_key, base_url, tool_choice
                    )
                    if response is not None and "content" in response:
                        break
                    raise RuntimeError("Invalid response from API")
                except Exception as err:
                    _err(f"API call failed (attempt {attempt}/{max_retries}): {err}")

                    if _is_content_filter_error(str(err)):
                        _err("Content filter triggered — forcing final answer.")
                        fallback = self._force_final_answer(
                            messages, params, api_key, base_url
                        )
                        if fallback:
                            return fallback
                        raise

                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"API call failed after {max_retries} attempts: {err}"
                        )
                    wait = retry_delay_base ** attempt
                    _warn(f"Retrying in {wait}s...")
                    time.sleep(wait)

            content = response.get("content") or ""
            tool_calls = response.get("tool_calls") or []

            # Extra retries if we get an error-like non-tool response
            if not tool_calls and self._looks_like_error(content):
                _warn("Error-like content received; retrying up to 3 times...")
                for _ in range(3):
                    try:
                        r2 = self._call_controller(
                            messages, params, api_key, base_url, tool_choice
                        )
                        c2 = (r2 or {}).get("content", "")
                        t2 = (r2 or {}).get("tool_calls") or []
                        if t2 or (c2 and not self._looks_like_error(c2)):
                            content, tool_calls = c2, t2
                            response = r2
                            break
                    except Exception:
                        time.sleep(1)

            # Append assistant message
            ai_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            messages.append(ai_msg)
            self.conversation_history.append(ai_msg)
            if content:
                thinking_preview = content[:400] + ("..." if len(content) > 400 else "")
                print(f"│ Thinking: {thinking_preview}")
            if tool_calls:
                names = [tc["function"]["name"] for tc in tool_calls]
                print(f"│ Action  : {', '.join(names)}")

            # --- Execute tools ---
            if tool_calls:
                scene_update_needed = False
                updated_ranges: List[Dict] = []
                vp_num_frames = None

                for tc in tool_calls:
                    t_start = time.time()
                    result = self._execute_tool(video_path, tc)
                    t_elapsed = time.time() - t_start

                    raw_args = tc["function"].get("arguments", "")
                    parsed_args = handle_json_parsing_error(raw_args, "tool arguments") or {}

                    self.tool_calls_log.append(
                        {
                            "iteration": iteration,
                            "tool_name": tc["function"]["name"],
                            "arguments": parsed_args,
                            "result": result,
                            "execution_time": t_elapsed,
                            "timestamp": time.time(),
                        }
                    )

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                    messages.append(tool_msg)
                    self.conversation_history.append(tool_msg)

                    tool_name = tc["function"]["name"]
                    self._update_hm3(tool_name, result, parsed_args, iteration)

                    reason = parsed_args.get("reason", "")
                    if reason:
                        print(f"│  Reason : {reason}")

                    if tool_name == "scene_snapper" and "frame_ranges" in result:
                        scene_update_needed = True
                        updated_ranges = result.get("frame_ranges", [])
                        try:
                            vp_num_frames = int(parsed_args.get("num_frames", 30))
                        except Exception:
                            vp_num_frames = None
                        fps = self.video_info.get("fps", 30.0)
                        ranges_str = "  ".join(
                            f"[{r['start_frame']}-{r['end_frame']}] "
                            f"({r['start_frame']/fps:.1f}s-{r['end_frame']/fps:.1f}s)"
                            for r in updated_ranges
                        )
                        print(f"│  Ranges : {ranges_str}")
                        caption = result.get("caption", "")
                        if caption:
                            print(f"│  Caption: {caption}")
                    elif tool_name == "audio_transcriber":
                        if result.get("status") == "no_audio":
                            print("│  Result : no audio stream in this video.")
                        else:
                            transcript = result.get("transcript_text", "")
                            preview = transcript[:200] + ("..." if len(transcript) > 200 else "")
                            print(f"│  Transcript ({len(transcript)} chars): {preview}")
                    elif tool_name == "clip_analyzer":
                        conf = result.get("confidence", "?")
                        answer = result.get("answer", "")
                        fr = result.get("frame_range", {})
                        fps = self.video_info.get("fps", 30.0)
                        t0 = fr.get("start_frame", 0) / fps
                        t1 = fr.get("end_frame", 0) / fps
                        print(f"│  Range  : [{fr.get('start_frame')}-{fr.get('end_frame')}] ({t0:.1f}s-{t1:.1f}s)")
                        print(f"│  Answer : {answer[:200]}")
                        print(f"│  Conf   : {conf}")

                # Attach new frames to context after scene_snapper
                if scene_update_needed and updated_ranges:
                    try:
                        self._attach_frames_to_messages(
                            messages, video_path, updated_ranges, vp_num_frames
                        )
                    except Exception as e:
                        _err(f"Frame attachment failed: {e}")

                # Capture the latest image message before rebuilding
                latest_image_msg = self._find_latest_image_message(messages)

                # Rebuild messages with updated HM³ for next turn
                messages = self._build_initial_messages(
                    system_prompt, question, is_multiple_choice, choice_format
                )
                for m in messages:
                    self.conversation_history.append(copy.deepcopy(m))

                if latest_image_msg:
                    messages.append(latest_image_msg)
                    self.conversation_history.append(copy.deepcopy(latest_image_msg))

            else:
                # No tool calls → treat as final answer
                if self._looks_like_error(content):
                    _warn("Error-like response; continuing to next iteration.")
                    continue
                print(f"└─ Final answer: {content}")
                return content

        # Maximum iterations reached — force conclusion
        print(f"└─ Step budget ({max_iterations}) reached — forcing final answer.")
        return self._force_final_answer(messages, params, api_key, base_url) or (
            "Maximum iterations reached without a conclusive answer."
        )

    # ------------------------------------------------------------------ #
    # Message builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_initial_messages(
        self,
        system_prompt: str,
        question: str,
        is_multiple_choice: bool,
        choice_format: str,
    ) -> List[Dict]:
        """Construct the [System, User] message pair for the current turn."""
        user = (
            f"**Video Information**\n"
            f"- Total frames: {self.video_info['total_frames']}\n"
            f"- Duration: {self.video_info['duration']:.1f} seconds\n"
            f"- FPS: {self.video_info['fps']:.2f}\n\n"
            "Available tools:\n"
            "• `scene_snapper`     — navigate to frame ranges and get a scene caption.\n"
            "• `audio_transcriber` — transcribe audio from frame ranges.\n"
            "• `clip_analyzer`     — analyze a local frame range with a sub-question.\n"
            "Always base the final answer on observations and tool outputs.\n\n"
        )

        try:
            user += f"**Current Memory (HM³)**\n{json.dumps(self.hm3, ensure_ascii=False)}\n\n"
        except Exception:
            pass

        q_text = str(question or "")
        if "\nOptions:" in q_text or q_text.strip().startswith("Question:"):
            user += q_text.strip() + "\n\n"
        else:
            user += f"**Question**\n{q_text}\n\n"

        if is_multiple_choice:
            if choice_format == "number":
                user += "Respond with only the number (0, 1, 2, 3, or 4) of the correct option.\n"
            else:
                user += "Respond with only the letter (A, B, C, or D) of the correct option.\n"

        messages: List[Dict] = [{"role": "system", "content": system_prompt}]

        if not self.video_has_audio:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This video has no audio stream.  "
                        "Do not call audio_transcriber."
                    ),
                }
            )

        messages.append({"role": "user", "content": user})
        return messages

    # ------------------------------------------------------------------ #
    # Controller call helpers                                              #
    # ------------------------------------------------------------------ #

    def _call_controller(self, messages, params, api_key, base_url, tool_choice):
        sanitized = self._sanitize_messages(messages)
        kwargs = {
            "messages": sanitized,
            "model_name": self.model_name,
            "tools": self.tools_registry,
            "tool_choice": tool_choice,
            "api_key": api_key,
            "endpoints": base_url,
        }
        kwargs.update(params or {})
        return call_openai_model_with_tools(**kwargs)

    def _force_final_answer(self, messages, params, api_key, base_url) -> Optional[str]:
        """Call the controller without tools to extract a final answer."""
        try:
            r = call_openai_model_with_tools(
                messages=messages
                + [
                    {
                        "role": "system",
                        "content": (
                            "You have reached the step budget.  Based on all gathered "
                            "evidence, provide your best final answer.  Do not call any tools."
                        ),
                    }
                ],
                model_name=self.model_name,
                tools=None,
                tool_choice="none",
                api_key=api_key,
                endpoints=base_url,
                **(params or {}),
            )
            if r and r.get("content"):
                return r["content"]
        except Exception as e:
            _err(f"Forced final answer failed: {e}")
        return None

    @staticmethod
    def _sanitize_messages(messages: List[Dict]) -> List[Dict]:
        """
        Remove assistant messages whose tool_calls have no matching tool response.
        This prevents API errors caused by unresolved tool call references.
        """
        responded_ids = {
            m.get("tool_call_id")
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        sanitized = []
        for msg in messages:
            if msg.get("role") == "assistant" and isinstance(
                msg.get("tool_calls"), list
            ):
                ids = [tc.get("id") for tc in msg["tool_calls"] if isinstance(tc, dict)]
                if ids and not all(i in responded_ids for i in ids):
                    msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            sanitized.append(msg)
        return sanitized

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        if not text or not isinstance(text, str):
            return True
        t = text.strip().lower()
        if not t or t == "none":
            return True
        return any(
            k in t
            for k in ("analysis failed", "api call failed", "error:", "exception:")
        )

    # ------------------------------------------------------------------ #
    # HM³ update helpers                                                   #
    # ------------------------------------------------------------------ #

    def _update_hm3(
        self,
        tool_name: str,
        result: Dict,
        args: Dict,
        iteration: int,
    ) -> None:
        """Route tool output to the correct HM³ tier."""
        reason = (result.get("reason") or args.get("reason", "")).strip()

        if tool_name == "scene_snapper":
            caption = result.get("caption", "")
            ranges = result.get("frame_ranges", args.get("frame_ranges", []))
            for r in ranges:
                try:
                    entry: Dict[str, Any] = {
                        "iteration": iteration,
                        "frame_interval": [
                            int(r["start_frame"]),
                            int(r["end_frame"]),
                        ],
                    }
                    if reason:
                        entry["reason"] = reason
                    if caption:
                        entry["caption"] = caption
                    self.hm3["scene_snapshots"].append(entry)
                except Exception:
                    continue

        elif tool_name == "audio_transcriber":
            if result.get("status") == "no_audio":
                return
            segments = result.get("segments", [])
            if not segments:
                return
            record: Dict[str, Any] = {"iteration": iteration, "segments": []}
            if reason:
                record["reason"] = reason
            for seg in segments:
                try:
                    record["segments"].append(
                        {
                            "frame_interval": [
                                int(seg["start_frame"]),
                                int(seg["end_frame"]),
                            ],
                            "text": str(seg.get("text", "")),
                        }
                    )
                except Exception:
                    continue
            if record["segments"]:
                combined = " ".join(
                    s["text"] for s in record["segments"] if s.get("text")
                ).strip()
                if combined:
                    record["text"] = combined
                self.hm3["audio_transcripts"].append(record)

        elif tool_name == "clip_analyzer":
            if result.get("error"):
                return
            fr = result.get("frame_range", {})
            try:
                entry = {
                    "iteration": iteration,
                    "frame_interval": [
                        int(fr["start_frame"]),
                        int(fr["end_frame"]),
                    ],
                    "question": str(result.get("question", "")),
                    "answer": str(result.get("answer", "")),
                    "confidence": result.get("confidence"),
                }
                if reason:
                    entry["reason"] = reason
                self.hm3["clip_analyses"].append(entry)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Tools registry
    # (For ease of implementation, we integrate the 
    # Interval Localizer and Clip Explorer described 
    # in the paper into a parameterized implementation.)                                                    
    # ------------------------------------------------------------------ #

    def _build_tools_registry(self) -> List[Dict]:
        """Return the list of tool definitions for the OpenAI function-calling API."""
        cfg = self.config.get_pipeline_config()
        audio_max = cfg["audio_max_frames"]
        fa_max = cfg["frame_analysis_max_frames"]

        return [
            # ---------------------------------------------------------- #
            # Scene Snapper                                               #
            # Updates the long-term perception pool P_l.                 #
            # Equation (1) in paper: V_C = SceneSnapper(F), F ∈ P_l     #
            # ---------------------------------------------------------- #
            {
                "type": "function",
                "function": {
                    "name": "scene_snapper",
                    "description": (
                        "Navigate to frame ranges, extract sampled frames, and return an "
                        "auto-generated scene caption.  Updates the long-term perception "
                        "pool.  Default sample size: 30 frames."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "frame_ranges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_frame": {
                                            "type": "integer",
                                            "description": "Start frame index (global, inclusive).",
                                        },
                                        "end_frame": {
                                            "type": "integer",
                                            "description": "End frame index (global, inclusive).",
                                        },
                                    },
                                    "required": ["start_frame", "end_frame"],
                                },
                                "description": (
                                    "Frame ranges to view.  Frames are distributed "
                                    "proportionally across ranges."
                                ),
                            },
                            "num_frames": {
                                "type": "integer",
                                "description": "Frames to extract: 30/60/90/150 (default 30).",
                                "enum": [30, 60, 90, 150],
                                "default": 30,
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief rationale for invoking this tool.",
                            },
                        },
                        "required": ["frame_ranges", "reason"],
                    },
                },
            },
            # ---------------------------------------------------------- #
            # Audio Transcriber                                           #
            # Equation (2) in paper: A_C = AudioTrans(A), A ∈ P_s       #
            # ---------------------------------------------------------- #
            {
                "type": "function",
                "function": {
                    "name": "audio_transcriber",
                    "description": (
                        f"Extract and transcribe audio from specific frame ranges using "
                        f"whisper-1.  Total frames across all ranges must be under "
                        f"{audio_max} frames (≈ 25 MB audio limit)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "frame_ranges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_frame": {"type": "integer"},
                                        "end_frame": {"type": "integer"},
                                    },
                                    "required": ["start_frame", "end_frame"],
                                },
                                "description": "Frame ranges to extract audio from.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief rationale for invoking this tool.",
                            },
                        },
                        "required": ["frame_ranges", "reason"],
                    },
                },
            },
            # ---------------------------------------------------------- #
            # Clip Analyzer                                               #
            # Equation (3) in paper:                                     #
            #   A_sub, S_sub = ClipAnalyzer(F, Q_sub), F ∈ P_s          #
            # ---------------------------------------------------------- #
            {
                "type": "function",
                "function": {
                    "name": "clip_analyzer",
                    "description": (
                        f"Analyze a local frame range by asking a sub-question.  "
                        f"Uniformly samples up to {fa_max} frames and returns an "
                        f"answer with a confidence score."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "frame_range": {
                                "type": "object",
                                "properties": {
                                    "start_frame": {
                                        "type": "integer",
                                        "description": "Start frame index (global, inclusive).",
                                    },
                                    "end_frame": {
                                        "type": "integer",
                                        "description": "End frame index (global, inclusive).",
                                    },
                                },
                                "required": ["start_frame", "end_frame"],
                            },
                            "question": {
                                "type": "string",
                                "description": "Sub-question to ask about this frame range.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief rationale for invoking this tool.",
                            },
                        },
                        "required": ["frame_range", "question", "reason"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------ #
    # Tool execution dispatcher                                            #
    # ------------------------------------------------------------------ #

    def _execute_tool(self, video_path: str, tool_call: Dict) -> Dict:
        """Parse tool arguments and dispatch to the appropriate handler."""
        tool_name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments", "")

        arguments = handle_json_parsing_error(raw_args, "tool arguments")
        if arguments is None:
            return {"error": f"Failed to parse arguments for {tool_name}"}

        try:
            if tool_name == "scene_snapper":
                return self._scene_snapper(video_path, **arguments)
            elif tool_name == "audio_transcriber":
                return self._audio_transcriber(video_path, **arguments)
            elif tool_name == "clip_analyzer":
                return self._clip_analyzer(video_path, **arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except TypeError as e:
            return {"error": f"Invalid parameters for {tool_name}: {e}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

    # ------------------------------------------------------------------ #
    # Tool: Scene Snapper                                                  #
    # ------------------------------------------------------------------ #

    def _scene_snapper(
        self,
        video_path: str,
        frame_ranges: List[Dict],
        reason: str,
        num_frames: int = None,
    ) -> Dict:
        """
        Navigate to frame ranges, extract frames, and generate a scene caption.

        This tool updates the long-term perception pool P_l in the Sensory Memory.
        The generated caption V_C is stored in the Result Memory.
        """
        reason = str(reason or "").strip()
        if num_frames not in (30, 60, 90, 150):
            num_frames = 30

        fps = self.video_info.get("fps", 30.0)

        all_paths = self._extract_frames_proportional(
            video_path, frame_ranges, total_frames=num_frames, target_short_side=256
        )
        if not all_paths:
            return {"error": "Failed to extract frames", "reason": reason}

        total_dur = sum(
            (r["end_frame"] - r["start_frame"]) / fps for r in frame_ranges
        )
        caption_result = self._generate_scene_caption(frame_ranges, all_paths)
        caption = caption_result.get("caption", "")

        result = {
            "status": "success",
            "frames_loaded": len(all_paths),
            "frame_ranges": frame_ranges,
            "total_duration": f"{total_dur:.1f}s",
            "caption": caption,
            "reason": reason,
            "message": (
                f"Loaded {len(all_paths)} frames from {len(frame_ranges)} range(s). "
                f"Duration: {total_dur:.1f}s"
            ),
        }
        if caption_result.get("status") == "error":
            result["caption_error"] = caption_result.get("error")
        return result

    def _generate_scene_caption(
        self, frame_ranges: List[Dict], frame_paths: List[str]
    ) -> Dict:
        """
        Call the scene_snapper model to caption the extracted frames.
        Corresponds to Equation (1): V_C = SceneSnapper(F), F ∈ P_l
        """
        if not frame_paths:
            return {"status": "skip", "caption": ""}

        fps = self.video_info.get("fps", 30.0)
        composite_paths = self._make_composite_grids(frame_paths)

        model = self.config.get_model("scene_snapper")
        params = self.config.get_model_params("scene_snapper")
        api_key, base_url = self.config.get_api_config("scene_snapper")

        fr = frame_ranges[0] if frame_ranges else {}
        s, e = fr.get("start_frame", 0), fr.get("end_frame", 0)

        try:
            response = self._run_with_retry(
                lambda: call_openai_model_with_tools(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a video frame captioning assistant.  "
                                "Each picture is a 3×2 mosaic of 6 frames, "
                                "row-major from top-left to bottom-right."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Caption the provided frames sampled uniformly from "
                                f"frame range {s}-{e}.  Describe the main scene or "
                                f"action in one concise English sentence."
                            ),
                        },
                    ],
                    model_name=model,
                    image_paths=[str(p) for p in composite_paths],
                    api_key=api_key or None,
                    endpoints=base_url or None,
                    **(params or {}),
                )
            )

            caption = ""
            if response and isinstance(response, dict):
                raw = (response.get("content") or "").strip()
                if raw:
                    caption = raw.splitlines()[0].strip()
                    # Strip any leading "Caption:" prefix the model may add
                    if caption.lower().startswith("caption:"):
                        caption = caption[len("caption:"):].strip()

            return {"status": "success" if caption else "empty", "caption": caption}

        except Exception as e:
            _err(f"Scene caption failed: {e}")
            return {"status": "error", "caption": "", "error": str(e)}

    # ------------------------------------------------------------------ #
    # Tool: Audio Transcriber                                              #
    # ------------------------------------------------------------------ #

    def _audio_transcriber(
        self, video_path: str, frame_ranges: List[Dict], reason: str
    ) -> Dict:
        """
        Extract audio from the specified frame ranges and transcribe with whisper-1.

        Corresponds to Equation (2): A_C = AudioTrans(A), A ∈ P_s
        """
        import os
        import subprocess
        import tempfile

        reason = str(reason or "").strip()
        fps = self.video_info["fps"]
        total_frames = self.video_info["total_frames"]
        max_frames = self.config.get_pipeline_config("audio_max_frames")

        # Check for audio stream
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if not probe.stdout.strip():
            self.video_has_audio = False
            return {
                "status": "no_audio",
                "message": "This video has no audio stream.",
                "segments": [],
                "reason": reason,
            }

        # Validate frame count / estimated size
        n_frames = sum(
            min(r["end_frame"], total_frames - 1) - max(0, r["start_frame"]) + 1
            for r in frame_ranges
        )
        duration = n_frames / fps
        estimated_mb = duration * 32000 / 1_000_000
        if estimated_mb > 25 or n_frames > max_frames:
            return {
                "error": (
                    f"Audio too large: {estimated_mb:.1f} MB / {n_frames} frames "
                    f"exceeds limits (25 MB, {max_frames} frames).  "
                    f"Reduce the frame ranges."
                ),
                "reason": reason,
            }

        # Build audio segment metadata
        segs = [
            {
                "start_frame": max(0, r["start_frame"]),
                "end_frame": min(total_frames - 1, r["end_frame"]),
                "start_time": max(0, r["start_frame"]) / fps,
                "end_time": min(total_frames - 1, r["end_frame"]) / fps,
                "duration": (
                    min(total_frames - 1, r["end_frame"])
                    - max(0, r["start_frame"])
                ) / fps,
            }
            for r in frame_ranges
        ]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name

        try:
            self._extract_audio_segments(str(video_path), segs, tmp_path)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                return {"error": "Audio extraction produced an empty file.", "reason": reason}

            from openai import OpenAI

            api_key, base_url = self.config.get_api_config("audio_transcriber")
            client = OpenAI(api_key=api_key or None, base_url=base_url or None)

            def _transcribe():
                with open(tmp_path, "rb") as af:
                    return client.audio.transcriptions.create(
                        model="whisper-1", file=af, response_format="verbose_json"
                    )

            transcript = self._run_with_retry(_transcribe)

            # Map relative timestamps back to global frame indices
            result_segs = []
            if hasattr(transcript, "segments"):
                for seg in transcript.segments:
                    gf_start, gf_end = self._relative_to_global_frames(
                        seg.start, seg.end, segs, fps
                    )
                    result_segs.append(
                        {
                            "start_frame": gf_start,
                            "end_frame": gf_end,
                            "text": seg.text.strip(),
                        }
                    )

            combined = " ".join(s["text"] for s in result_segs if s["text"]).strip()
            return {
                "segments": result_segs,
                "transcript_text": combined,
                "reason": reason,
            }

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _extract_audio_segments(video_path: str, segs: List[Dict], out_path: str) -> None:
        """Extract (and optionally concatenate) audio segments using ffmpeg."""
        import os
        import subprocess
        import tempfile

        def _ffmpeg(*args):
            r = subprocess.run(list(args), capture_output=True, text=True)
            if r.returncode != 0:
                # Keep only error-related lines from stderr
                errors = [
                    l for l in r.stderr.splitlines()
                    if l.strip() and any(
                        k in l.lower()
                        for k in ("error", "invalid", "failed", "could not")
                    )
                    and not any(
                        k in l for k in ("ffmpeg version", "built with", "configuration:", "lib")
                    )
                ]
                raise RuntimeError("\n".join(errors) or r.stderr[-300:])

        base_args = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        ]

        if len(segs) == 1:
            s = segs[0]
            _ffmpeg(
                *base_args[:4], "-ss", str(s["start_time"]),
                "-t", str(s["duration"]), *base_args[4:], out_path,
            )
            return

        # Multiple segments: extract individually then concat
        seg_files = []
        concat_list = None
        try:
            for i, s in enumerate(segs):
                sf = f"{out_path}_{i}.wav"
                seg_files.append(sf)
                _ffmpeg(
                    *base_args[:4], "-ss", str(s["start_time"]),
                    "-t", str(s["duration"]), *base_args[4:], sf,
                )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                concat_list = f.name
                for sf in seg_files:
                    f.write(f"file '{sf}'\n")

            _ffmpeg(
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list, "-c", "copy", out_path,
            )
        finally:
            for sf in seg_files:
                if os.path.exists(sf):
                    os.unlink(sf)
            if concat_list and os.path.exists(concat_list):
                os.unlink(concat_list)

    @staticmethod
    def _relative_to_global_frames(
        rel_start: float,
        rel_end: float,
        audio_segs: List[Dict],
        fps: float,
    ) -> tuple:
        """Map a whisper-relative timestamp to global frame indices."""
        cumulative = 0.0
        for seg in audio_segs:
            if cumulative <= rel_start < cumulative + seg["duration"]:
                offset_s = rel_start - cumulative
                offset_e = min(rel_end - cumulative, seg["duration"])
                g_start = seg["start_time"] + offset_s
                g_end = seg["start_time"] + offset_e
                # Handle cross-segment end time
                if rel_end > cumulative + seg["duration"]:
                    remaining = rel_end - (cumulative + seg["duration"])
                    for ns in audio_segs[audio_segs.index(seg) + 1:]:
                        if remaining <= ns["duration"]:
                            g_end = ns["start_time"] + remaining
                            break
                        remaining -= ns["duration"]
                return int(g_start * fps), int(g_end * fps)
            cumulative += seg["duration"]

        if audio_segs:
            last = audio_segs[-1]
            g = int(last["end_time"] * fps)
            return g, g
        return 0, 0

    # ------------------------------------------------------------------ #
    # Tool: Clip Analyzer                                                  #
    # ------------------------------------------------------------------ #

    def _clip_analyzer(
        self,
        video_path: str,
        frame_range: Dict,
        question: str,
        reason: str,
    ) -> Dict:
        """
        Sample frames from a local interval and answer a sub-question.

        Corresponds to Equation (3): A_sub, S_sub = ClipAnalyzer(F, Q_sub), F ∈ P_s
        """
        reason = str(reason or "").strip()

        if not frame_range:
            return {"error": "No frame range provided.", "reason": reason}
        if not question:
            return {"error": "No question provided.", "reason": reason}

        max_f = self.video_info.get("total_frames", 0) - 1
        start = max(0, min(int(frame_range.get("start_frame", 0)), max_f))
        end = max(start, min(int(frame_range.get("end_frame", 0)), max_f))
        fps = self.video_info.get("fps", 30.0)

        if start == end:
            return {"error": "start_frame equals end_frame.", "reason": reason}

        n = max(1, self.config.get_pipeline_config("frame_analysis_max_frames") or 30)
        frame_paths = get_cropped_frame_paths(
            video_path=video_path,
            start_frame=start,
            end_frame=end,
            num_frames=n,
            session_id=f"{self.session_id}_ca_{start}_{end}",
            target_short_side=512,
        )
        if not frame_paths:
            return {"error": "Failed to extract frames.", "reason": reason}

        model = self.config.get_model("clip_analyzer")
        params = self.config.get_model_params("clip_analyzer")
        api_key, base_url = self.config.get_api_config("clip_analyzer")

        system = (
            f"You are an expert video frame analyst.  Analyze the provided "
            f"{len(frame_paths)} frames sampled uniformly from frame range "
            f"{start}-{end} and answer the question below.  Each frame displays "
            f"the global index in white at the top-left."
        )
        user = (
            f"Analyze this frame sequence and answer the question.\n"
            f"Include a confidence score between 0.0 and 1.0.\n"
            f"Format:\nAnswer: <answer>\nConfidence: <score>\n\n"
            f"Question: {question}"
        )

        response = self._run_with_retry(
            lambda: call_openai_model_with_tools(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model_name=model,
                image_paths=frame_paths,
                api_key=api_key or None,
                endpoints=base_url or None,
                **(params or {}),
            )
        )

        if not response or "content" not in response:
            raise RuntimeError("Empty response from clip analyzer model.")

        text = response["content"].strip()
        answer, confidence = None, None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Answer:"):
                answer = line[7:].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = max(0.0, min(1.0, float(line[11:].strip())))
                except ValueError:
                    pass

        if answer is None:
            answer = text
        if confidence is None:
            m = re.search(r"([0-9]*\.?[0-9]+)", text)
            confidence = max(0.0, min(1.0, float(m.group(1)))) if m else 0.5



        return {
            "status": "success",
            "frame_range": {
                "start_frame": start,
                "end_frame": end,
            },
            "sampled_frames": len(frame_paths),
            "answer": answer,
            "confidence": confidence,
            "question": question,
            "reason": reason,
        }

    # ------------------------------------------------------------------ #
    # Frame extraction helpers                                             #
    # ------------------------------------------------------------------ #

    def _extract_frames_proportional(
        self,
        video_path: str,
        frame_ranges: List[Dict],
        total_frames: int,
        target_short_side: int = 256,
        silent: bool = False,
    ) -> List[str]:
        """Extract total_frames frames distributed proportionally across ranges."""
        if not frame_ranges:
            return []

        total_length = sum(
            r["end_frame"] - r["start_frame"] + 1 for r in frame_ranges
        )
        all_paths: List[str] = []

        for i, r in enumerate(frame_ranges):
            start = max(0, r["start_frame"])
            end = min(self.video_info["total_frames"] - 1, r["end_frame"])
            ratio = (end - start + 1) / total_length
            count = max(1, int(total_frames * ratio))

            if i == len(frame_ranges) - 1:
                count = total_frames - len(all_paths)

            paths = get_cropped_frame_paths(
                video_path=video_path,
                start_frame=start,
                end_frame=end,
                num_frames=count,
                session_id=self.session_id,
                target_short_side=target_short_side,
                silent=silent,
            )
            all_paths.extend(paths)
            if len(all_paths) >= total_frames:
                break

        return all_paths[:total_frames]

    def _make_composite_grids(
        self, frame_paths: List[str], rows: int = 2, cols: int = 3
    ) -> List[Path]:
        """
        Tile frames into 3×2 composite grid images.

        Reduces the number of images sent to the vision model while preserving
        spatial content.  Matches the grid layout described in the paper.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return [Path(p) for p in frame_paths]

        if not frame_paths:
            return []

        tile_count = rows * cols
        parent = Path(frame_paths[0]).parent
        composites: List[Path] = []

        for idx in range(0, len(frame_paths), tile_count):
            chunk = frame_paths[idx: idx + tile_count]
            imgs = []
            for p in chunk:
                img = cv2.imread(str(p))
                if img is not None:
                    imgs.append(img)

            if not imgs:
                continue

            th = min(img.shape[0] for img in imgs)
            tw = min(img.shape[1] for img in imgs)
            resized = [
                cv2.resize(img, (tw, th)) if img.shape[:2] != (th, tw) else img
                for img in imgs
            ]
            while len(resized) < tile_count:
                resized.append(np.zeros((th, tw, 3), dtype=np.uint8))

            grid = cv2.vconcat(
                [cv2.hconcat(resized[r * cols: (r + 1) * cols]) for r in range(rows)]
            )
            out = parent / f"{self.session_id}_grid_{idx // tile_count:03d}.png"
            cv2.imwrite(str(out), grid)
            composites.append(out)

        return composites if composites else [Path(p) for p in frame_paths]

    def _attach_frames_to_messages(
        self,
        messages: List[Dict],
        video_path: str,
        frame_ranges: List[Dict],
        num_frames: Optional[int],
    ) -> None:
        """Extract frames and append a vision user message."""
        limit = self.config.get_pipeline_config("total_frames_limit")
        count = min(num_frames or limit, limit)

        new_paths = self._extract_frames_proportional(
            video_path, frame_ranges, total_frames=count, target_short_side=256, silent=True
        )
        if not new_paths:
            return

        composites = self._make_composite_grids(new_paths)
        prepared = composites if composites else [Path(p) for p in new_paths]

        # Remove stale image messages
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and any(
                    item.get("type") == "image_url"
                    for item in msg["content"]
                    if isinstance(item, dict)
                )
            ):
                messages.pop(i)

        from videoarm.api.client import local_image_to_data_url

        image_content = [
            {"type": "image_url", "image_url": {"url": local_image_to_data_url(str(p))}}
            for p in prepared
        ]
        messages.append({"role": "user", "content": image_content})

    @staticmethod
    def _find_latest_image_message(messages: List[Dict]) -> Optional[Dict]:
        """Return the last user message that contains image content."""
        for msg in reversed(messages):
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and any(
                    item.get("type") == "image_url"
                    for item in msg["content"]
                    if isinstance(item, dict)
                )
            ):
                return copy.deepcopy(msg)
        return None

    # ------------------------------------------------------------------ #
    # Retry helper                                                         #
    # ------------------------------------------------------------------ #

    def _run_with_retry(self, call_factory, attempts=None, delay_base=None):
        attempts = max(1, attempts or self.tool_retry_attempts)
        delay_base = delay_base or self.tool_retry_delay_base
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return call_factory()
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
                wait = delay_base ** attempt
                _warn(f"Tool retry {attempt}/{attempts} in {wait}s: {exc}")
                time.sleep(wait)
        raise last_exc

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _save_qa_result(
        self, video_path: str, question: str, answer: str, elapsed: float
    ) -> None:
        """Persist the QA result (conversation history + HM³) to JSON."""
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
            filename = f"qa_{q_hash}_{ts}_{self.session_id}.json"

            result = {
                "video_path": str(video_path),
                "question": question,
                "answer": answer,
                "model_name": self.model_name,
                "session_id": self.session_id,
                "timestamp": ts,
                "processing_time_seconds": elapsed,
                "video_info": self.video_info,
                "hm3": self.hm3,
                "conversation_history": self.conversation_history,
                "tool_calls_log": self.tool_calls_log,
                "tool_statistics": {
                    "total_calls": len(self.tool_calls_log),
                    "by_tool": self._tool_usage_stats(),
                    "iterations_used": self.analysis_iterations_used,
                },
            }

            out = RESULTS_DIR / filename
            tmp = out.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            tmp.replace(out)
            print(f"Result saved: {out}")

        except Exception as e:
            print(f"Failed to save QA result: {e}")

    def _tool_usage_stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.tool_calls_log:
            name = entry.get("tool_name", "unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _cleanup_temp_frames(self) -> None:
        """Remove all temporary frame directories for this session."""
        for d in TEMP_DIR.glob(f"{self.session_id}*"):
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
