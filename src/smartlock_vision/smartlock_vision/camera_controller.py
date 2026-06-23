"""Camera control module for smartlock vision system.

This module handles:
- Camera initialization and configuration
- Frame capture and streaming
- Power management (camera enabled/disabled based on IR trigger)

Typical usage:
    camera = CameraController(device=0, width=640, height=480)
    camera.start()
    frame = camera.read_frame()
    camera.stop()
"""

import time
import threading
from typing import Any, Optional
import cv2
import numpy as np

from smartlock_vision.vision_utils import get_default_logger


class CameraController:
    """Control and capture camera frames.

    Attributes:
        device (int): Camera device index (0 = primary camera).
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
        fps (int): Target frames per second.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[Any] = None,
    ):
        """Initialize camera controller.

        Args:
            device: Camera device index.
            width: Frame width.
            height: Frame height.
            fps: Target frames per second.
        """
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or get_default_logger(__name__)

        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

    def start(self) -> bool:
        """Start camera capture thread.

        Returns:
            True if started successfully, False otherwise.
        """
        if self.is_running:
            self.logger.warning("Camera already running")
            return False

        if not self._open():
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        self.logger.info("Camera capture started")
        return True

    def stop(self) -> None:
        """Stop camera capture thread."""
        if not self.is_running:
            return

        self.is_running = False

        if self.thread:
            self.thread.join(timeout=2.0)

        self._close()
        self.logger.info("Camera capture stopped")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read latest captured frame.

        Returns:
            Current frame as numpy array (BGR), or None if no frame available.
        """
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame

    def _open(self) -> bool:
        """Open camera device.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        try:
            self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

            if not self.capture.isOpened():
                self.logger.error(f"Failed to open camera device {self.device}")
                return False

            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.logger.info(f"Opened camera {self.device}")
            return True
        except Exception as e:
            self.logger.error(f"Error opening camera: {e}")
            return False

    def _close(self) -> None:
        """Close camera device."""
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None
            self.logger.info("Camera closed")
        self.current_frame = None

    def _capture_loop(self) -> None:
        """Main camera capture loop."""
        self.logger.info("Starting camera capture loop")

        if self.capture is None:
            self.logger.error("Camera capture device is None. Exiting loop.")
            self.is_running = False
            return

        while self.is_running:
            try:
                ret, frame = self.capture.read()

                if not ret:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue

                # Store current frame
                with self.frame_lock:
                    self.current_frame = frame
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

        self.logger.info("Camera capture loop stopped")
