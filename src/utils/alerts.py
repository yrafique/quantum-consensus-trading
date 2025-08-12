"""
trading_system.alerts
=====================

Utilities for sending notifications to the user when certain trading
conditions are met.  On macOS this can integrate with the native
notification centre via the ``pync`` library or AppleScript.  In
environments where notifications are unavailable (such as this
execution context), messages fall back to standard output.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

from .config import ENABLE_NOTIFICATIONS


def send_notification(title: str, message: str) -> None:
    """Send a desktop notification or print to stdout.

    Parameters
    ----------
    title : str
        Title of the notification.
    message : str
        Body text of the notification.
    """
    if not ENABLE_NOTIFICATIONS:
        print(f"[NOTIFY] {title}: {message}")
        return
    # Attempt to use pync if available
    try:
        import pync

        pync.notify(message, title=title)
        return
    except ImportError:
        pass
    # Fallback: use AppleScript on macOS
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run([
            "osascript",
            "-e",
            script,
        ], check=True)
    except Exception:
        # As a last resort, print to stdout
        print(f"[NOTIFY] {title}: {message}")


__all__ = ["send_notification"]