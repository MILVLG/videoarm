"""
VideoARM model and pipeline configuration.

Model names are set in code (DEFAULT_MODELS).  Most parameters can be
overridden at runtime via environment variables prefixed with VIDEOARM_.
"""

import os
from typing import Any, Dict, Tuple


class ModelConfig:
    """Manages model names, API credentials, and pipeline hyper-parameters."""

    # ------------------------------------------------------------------ #
    # Default model assignments (matching paper's implementation details) #
    # ------------------------------------------------------------------ #
    DEFAULT_MODELS: Dict[str, str] = {
        # Controller and Temporal Scoping Tools (paper: OpenAI o3)
        "controller": "o3",
        # Multimodal Understanding Tools — visual analysis (paper: GPT-4.1)
        "clip_analyzer": "gpt-4.1",
        # Scene Snapper caption model
        "scene_snapper": "gpt-4.1",
        # Audio Transcriber (paper: whisper-1)
        "audio_transcriber": "whisper-1",
    }

    # ------------------------------------------------------------------ #
    # Per-model generation parameters                                     #
    # ------------------------------------------------------------------ #
    MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
        "controller": {"max_tokens": 10000},
        "clip_analyzer": {"max_tokens": 1000},
        "scene_snapper": {"max_tokens": 1000},
    }

    # ------------------------------------------------------------------ #
    # Pipeline hyper-parameters                                           #
    # ------------------------------------------------------------------ #
    PIPELINE_CONFIG: Dict[str, Any] = {
        # Step budget N (paper Section 4.2)
        "max_iterations": 10,
        # Total frames that may be passed to the controller per turn
        "total_frames_limit": 240,
        # Max frames per single tool call (used by scene_snapper)
        "max_frames_per_tool": 150,
        # Frames sampled by Clip Analyzer (paper: up to 50)
        "frame_analysis_max_frames": 50,
        # Max total frames for Audio Transcriber (≈ 25 MB audio bound)
        "audio_max_frames": 15000,
        # API retry settings
        "api_retry_attempts": 4,
        "api_retry_delay_base": 4,
        # Tool-level retry settings
        "tool_retry_attempts": 2,
        "tool_retry_delay_base": 2,
    }

    # ------------------------------------------------------------------ #
    # Video processing                                                    #
    # ------------------------------------------------------------------ #
    VIDEO_CONFIG: Dict[str, Any] = {
        "video_resolution": 1080,
    }

    def __init__(self) -> None:
        self._models = self._load_models()
        self._load_api_config()
        self._load_env_overrides()

    # ------------------------------------------------------------------ #
    # Internal loaders                                                    #
    # ------------------------------------------------------------------ #

    def _load_models(self) -> Dict[str, str]:
        models = self.DEFAULT_MODELS.copy()

        overrides = {
            "controller": os.getenv("VIDEOARM_MODEL_CONTROLLER"),
            "clip_analyzer": os.getenv("VIDEOARM_MODEL_CLIP_ANALYZER"),
            "scene_snapper": os.getenv("VIDEOARM_MODEL_SCENE_SNAPPER"),
            "audio_transcriber": os.getenv("VIDEOARM_MODEL_AUDIO_TRANSCRIBER"),
        }
        for component, value in overrides.items():
            if value:
                models[component] = value
        return models

    def _load_api_config(self) -> None:
        self.default_api_key = os.getenv("OPENAI_API_KEY", "")
        self.default_base_url = os.getenv("OPENAI_BASE_URL", "")
        self.component_api_keys: Dict[str, str] = {}
        self.component_base_urls: Dict[str, str] = {}

        for component in self._models:
            env = component.upper()
            key = os.getenv(f"VIDEOARM_API_KEY_{env}")
            url = os.getenv(f"VIDEOARM_BASE_URL_{env}")
            if key:
                self.component_api_keys[component] = key
            if url:
                self.component_base_urls[component] = url

    def _load_env_overrides(self) -> None:
        int_vars = {
            "max_iterations": "VIDEOARM_MAX_ITERATIONS",
            "total_frames_limit": "VIDEOARM_TOTAL_FRAMES_LIMIT",
            "max_frames_per_tool": "VIDEOARM_MAX_FRAMES_PER_TOOL",
            "frame_analysis_max_frames": "VIDEOARM_FRAME_ANALYSIS_MAX_FRAMES",
            "audio_max_frames": "VIDEOARM_AUDIO_MAX_FRAMES",
            "api_retry_attempts": "VIDEOARM_API_RETRY_ATTEMPTS",
            "api_retry_delay_base": "VIDEOARM_API_RETRY_DELAY_BASE",
            "tool_retry_attempts": "VIDEOARM_TOOL_RETRY_ATTEMPTS",
            "tool_retry_delay_base": "VIDEOARM_TOOL_RETRY_DELAY_BASE",
        }
        for param, env_var in int_vars.items():
            val = os.getenv(env_var)
            if val:
                self.PIPELINE_CONFIG[param] = int(val)

    # ------------------------------------------------------------------ #
    # Public accessors                                                    #
    # ------------------------------------------------------------------ #

    def get_model(self, component: str) -> str:
        """Return the model name for a given component."""
        return self._models.get(component, self.DEFAULT_MODELS.get(component, "gpt-4.1"))

    def get_model_params(self, component: str) -> Dict[str, Any]:
        """Return generation parameters for a component."""
        return self.MODEL_PARAMS.get(component, {})

    def get_api_config(self, component: str = None) -> Tuple[str, str]:
        """
        Return ``(api_key, base_url)`` for a component.

        Falls back to the global OPENAI_API_KEY / OPENAI_BASE_URL if no
        component-specific override is found.
        """
        if not component:
            return self.default_api_key, self.default_base_url

        env = component.upper()
        # Lazy-load from env (supports runtime changes)
        if component not in self.component_api_keys:
            val = os.getenv(f"VIDEOARM_API_KEY_{env}")
            if val:
                self.component_api_keys[component] = val
        if component not in self.component_base_urls:
            val = os.getenv(f"VIDEOARM_BASE_URL_{env}")
            if val:
                self.component_base_urls[component] = val

        api_key = self.component_api_keys.get(component, self.default_api_key)
        base_url = self.component_base_urls.get(component, self.default_base_url)
        return api_key, base_url

    def get_pipeline_config(self, param: str = None) -> Any:
        """Return a pipeline parameter, or the entire config dict."""
        if param:
            return self.PIPELINE_CONFIG.get(param)
        return self.PIPELINE_CONFIG.copy()

    def get_all_models(self) -> Dict[str, str]:
        return self._models.copy()

    def update_model(self, component: str, model_name: str) -> None:
        self._models[component] = model_name


# ------------------------------------------------------------------ #
# Global singleton                                                    #
# ------------------------------------------------------------------ #

_config: ModelConfig | None = None


def get_config() -> ModelConfig:
    global _config
    if _config is None:
        _config = ModelConfig()
    return _config


def get_model(component: str) -> str:
    return get_config().get_model(component)


def get_model_params(component: str) -> Dict[str, Any]:
    return get_config().get_model_params(component)


def get_api_config(component: str = None) -> Tuple[str, str]:
    return get_config().get_api_config(component)


def get_pipeline_config(param: str = None) -> Any:
    return get_config().get_pipeline_config(param)
