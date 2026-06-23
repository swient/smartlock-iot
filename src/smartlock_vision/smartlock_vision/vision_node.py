"""Vision node for smartlock system.

This ROS 2 node manages:
- Camera capture (enabled/disabled based on IR trigger)
- Face detection and feature extraction
- Publishing face vectors to authentication system

"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from smartlock_interfaces.msg import FaceVector
from smartlock_interfaces.srv import CaptureImage, StartCamera
from smartlock_vision.camera_controller import CameraController
from smartlock_vision.face_engine import FaceEngine


class VisionNode(Node):
    """ROS 2 Node for camera and face recognition."""

    def __init__(self) -> None:
        """Initialize vision node."""
        super().__init__("smartlock_vision_node")

        # Declare parameters
        self.declare_parameter("camera_device", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("ir_timeout_sec", 10.0)

        # Get parameters with type assertions
        self.camera_device = self.get_parameter("camera_device").get_parameter_value().integer_value
        self.camera_width = self.get_parameter("camera_width").get_parameter_value().integer_value
        self.camera_height = self.get_parameter("camera_height").get_parameter_value().integer_value
        self.camera_fps = self.get_parameter("camera_fps").get_parameter_value().integer_value
        self.ir_timeout = self.get_parameter("ir_timeout_sec").get_parameter_value().double_value

        # Initialize vision components
        self.camera = CameraController(
            device=self.camera_device,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            logger=self.get_logger(),
        )

        self.face_engine = FaceEngine(
            logger=self.get_logger(),
        )

        self.callback_group = ReentrantCallbackGroup()

        # Create services
        self.capture_image_srv = self.create_service(
            CaptureImage, "/smartlock/capture_image", self._capture_image_callback, callback_group=self.callback_group
        )

        # Create publisher
        self.face_vector_pub = self.create_publisher(
            FaceVector,
            "/smartlock/face_vector",
            10,
        )

        # Create subscription
        self.start_camera_srv = self.create_service(
            StartCamera,
            "/smartlock/start_camera",
            self._start_camera_callback,
            callback_group=self.callback_group,
        )

        # Create timers
        self.camera_monitor_timer = self.create_timer(
            1.0,
            self._camera_monitor_callback,
            callback_group=self.callback_group,
        )
        self.face_detect_timer = self.create_timer(
            0.5,
            self._face_detect_callback,
            callback_group=self.callback_group,
        )
        self.face_detect_timer.cancel()

        # State management
        self.camera_enabled = False
        self.last_trigger_time = self.get_clock().now()

        self.get_logger().info("Vision node initialized")
        self.get_logger().info(f"Camera device: {self.camera_device}")
        self.get_logger().info(f"Timeout: {self.ir_timeout}s")

    def _capture_image_callback(self, request, response) -> None:
        """Handle capture image service request."""
        if not self.camera_enabled:
            if not self.camera.start():
                response.success = False
                response.message = "Failed to start camera"
                return response

            time.sleep(0.5)

        frame = self.camera.read_frame()

        if not self.camera_enabled:
            self.camera.stop()

        if frame is None:
            response.success = False
            response.message = "Failed to capture frame"
            return response

        try:
            response.image.height = frame.shape[0]
            response.image.width = frame.shape[1]
            response.image.encoding = "bgr8"
            response.image.is_bigendian = 0
            response.image.step = frame.shape[1] * 3
            response.image.data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
            response.success = True
            response.message = "Image captured successfully"
        except Exception as e:
            response.success = False
            response.message = f"Failed to convert image: {e}"

        return response

    def _start_camera_callback(self, request, response) -> None:
        """Handle start camera service request."""
        self.last_trigger_time = self.get_clock().now()

        # Enable camera if not already enabled
        if not self.camera_enabled:
            self._enable_camera()

        response.success = True
        response.message = "Camera started"
        return response

    def _enable_camera(self) -> None:
        """Enable camera capture."""
        if self.camera_enabled:
            return

        if self.camera.start():
            self.camera_enabled = True
            self.face_detect_timer.reset()
            self.get_logger().info("Camera enabled")
        else:
            self.get_logger().error("Failed to enable camera")

    def _stop_camera(self) -> None:
        """Stop camera capture to save power."""
        if not self.camera_enabled:
            return

        self.face_detect_timer.cancel()
        self.camera.stop()
        self.camera_enabled = False
        self.get_logger().info("Camera disabled (power saving)")

    def _camera_monitor_callback(self) -> None:
        """Monitor camera health and timeout."""
        current_time = self.get_clock().now()

        # Check if timeout exceeded
        if self.camera_enabled:
            if current_time - self.last_trigger_time > Duration(seconds=self.ir_timeout):
                self.get_logger().info("Timeout exceeded, disabling camera")
                self._stop_camera()

    def _face_detect_callback(self) -> None:
        """Process frame and detect faces."""
        if not self.camera_enabled:
            return

        # Read frame
        frame = self.camera.read_frame()
        if frame is None:
            return

        # Extract face vectors
        try:
            result = self.face_engine.detect_and_extract(frame)

            if result is not None:
                # Create and publish message
                face_msg = FaceVector()
                face_msg.timestamp = self.get_clock().now().to_msg()
                face_msg.vector = result.embedding.tolist()
                face_msg.confidence = result.confidence

                self.face_vector_pub.publish(face_msg)
                self.get_logger().debug(f"Published face vector")

        except Exception as e:
            self.get_logger().error(f"Error detecting face: {e}")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = VisionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_camera()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
