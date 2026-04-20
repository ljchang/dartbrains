"""
Notebook UI helpers for marimo tutorials.
"""

import marimo as mo


def youtube(video_id: str) -> mo.Html:
    """Embed a YouTube video by video ID.

    Usage:
        from Code.notebook_utils import youtube
        youtube("dQw4w9WgXcQ")
    """
    return mo.Html(
        f'<iframe width="560" height="315" '
        f'src="https://www.youtube.com/embed/{video_id}" '
        f'frameborder="0" allowfullscreen></iframe>'
    )
