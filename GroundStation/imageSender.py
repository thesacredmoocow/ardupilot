import os
import threading
import time
from typing import Optional

import cv2
import imagezmq  # make sure `pip install imagezmq` is done in your environment
import numpy as np


class _ImageSenderThread:
    """
    Background sender that publishes the latest frame over imagezmq at a fixed rate.
    """

    def __init__(self, name: str, endpoint: str, send_hz: float = 10.0) -> None:
        self._name = name
        self._endpoint = endpoint
        self._interval = 1.0 / float(send_hz)

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stop = threading.Event()

        self._sender = imagezmq.ImageSender(connect_to=self._endpoint)
        self._thread = threading.Thread(target=self._loop, name="ImageSender", daemon=True)
        self._thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        """Store the most recent frame to be sent by the background thread."""
        if frame is None:
            return
        with self._lock:
            self._latest_frame = frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        next_send = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_send:
                time.sleep(next_send - now)
                continue
            next_send = now + self._interval

            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is None:
                continue

            # Ensure frame is a contiguous BGR image
            frame = np.ascontiguousarray(frame)
            try:
                self._sender.send_image(self._name, frame)
            except Exception:
                # Swallow send errors; they should not kill the main process
                pass


_sender_thread: Optional[_ImageSenderThread] = None


def init_image_sender(name: str = "slam",
                      endpoint: Optional[str] = None,
                      send_hz: float = 10.0) -> None:
    """
    Initialise the global image sender thread.

    - name: stream name reported to imagezmq receiver
    - endpoint: imagezmq endpoint, e.g. 'tcp://192.168.0.108:5555'
      Defaults to IMAGEZMQ_ENDPOINT env var or 'tcp://127.0.0.1:5555'.
    - send_hz: maximum send rate (default 10 Hz)
    """
    global _sender_thread
    if _sender_thread is not None:
        return

    if endpoint is None:
        endpoint = os.environ.get("IMAGEZMQ_ENDPOINT", "tcp://192.168.0.116:5555")

    _sender_thread = _ImageSenderThread(name=name, endpoint=endpoint, send_hz=send_hz)


def update_latest_frame(frame: np.ndarray) -> None:
    """
    Pass the latest frame to the sender.
    The frame will be transmitted at up to the configured send_hz rate.
    """
    if _sender_thread is None:
        return
    _sender_thread.update_frame(frame)


def stop_image_sender() -> None:
    """Stop the background sender thread if it is running."""
    global _sender_thread
    if _sender_thread is None:
        return
    _sender_thread.stop()
    _sender_thread = None

