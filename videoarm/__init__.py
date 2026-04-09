"""
VideoARM: Agentic Reasoning-over-Hierarchical-Memory for Long-Form Video Understanding.

Paper: https://arxiv.org/abs/2512.12360
"""

from videoarm.core.agent import VideoARMAgent


def ask(video_path: str, question: str, multiple_choice: bool = False) -> str:
    """Convenience wrapper: answer a question about a video."""
    agent = VideoARMAgent()
    return agent.ask(video_path, question, is_multiple_choice=multiple_choice)


__all__ = ["VideoARMAgent", "ask"]
