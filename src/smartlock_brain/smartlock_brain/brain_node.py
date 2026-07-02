"""Brain node for smartlock system.

This ROS 2 node manages:
- Authentication request processing
- Hardware command dispatch
- Reset mode and crypto provisioning coordination
- Secure storage of master keys
"""

import os
import json
import asyncio
import threading
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from smartlock_interfaces.msg import AuthenticationResult, FaceVector
from smartlock_interfaces.srv import (
    CaptureImage,
    HardwareCommand,
    HardwareEvent,
    ProvisionCrypto,
    ServerCommand,
    StartCamera,
    StoreMasterKey,
)
from smartlock_brain.auth_manager import AuthManager
from smartlock_brain.reset_channel import BleResetChannel
from smartlock_brain.brain_utils import generate_binding_key, generate_initial_pin


class BrainNode(Node):
    """ROS 2 Node for smartlock core logic."""

    def __init__(self) -> None:
        """Initialize brain node."""
        super().__init__("smartlock_brain_node")

        # Declare parameters
        self.declare_parameter("device_uuid", "001")

        # Get parameters
        self.device_uuid = self.get_parameter("device_uuid").get_parameter_value().string_value

        self.callback_group = ReentrantCallbackGroup()

        # Create services
        self.create_service(
            StoreMasterKey,
            "/smartlock/store_master_key",
            self._store_master_key_callback,
            callback_group=self.callback_group,
        )
        self.create_service(
            ServerCommand,
            "/smartlock/server_command",
            self._server_command_callback,
            callback_group=self.callback_group,
        )
        self.create_service(
            HardwareEvent,
            "/smartlock/hardware_event",
            self._hardware_event_callback,
            callback_group=self.callback_group,
        )

        # Create clients
        self.capture_image_client = self.create_client(
            CaptureImage,
            "/smartlock/capture_image",
        )
        self.provision_crypto_client = self.create_client(
            ProvisionCrypto,
            "/smartlock/provision_crypto",
        )
        self.hardware_command_client = self.create_client(
            HardwareCommand,
            "/smartlock/hardware_command",
            callback_group=self.callback_group,
        )
        self.start_camera_client = self.create_client(
            StartCamera,
            "/smartlock/start_camera",
            callback_group=self.callback_group,
        )

        # Create publishers
        self.auth_result_pub = self.create_publisher(
            AuthenticationResult,
            "/smartlock/auth_result",
            10,
        )

        # Create subscriptions
        self.face_vector_sub = self.create_subscription(
            FaceVector,
            "/smartlock/face_vector",
            self._face_vector_callback,
            10,
            callback_group=self.callback_group,
        )

        self.base_dir = Path.home() / ".smartlock"
        self.auth_db_path = self.base_dir / "auth_db.json"
        self.crypto_db_path = self.base_dir / "crypto.json"

        # Initialize managers
        self.auth_manager = AuthManager(db_path=self.auth_db_path, logger=self.get_logger())
        self.reset_channel = BleResetChannel(device_uuid=self.device_uuid, logger=self.get_logger())

        self.is_registering_face = False
        self.is_authenticating_face = False
        self.face_reg_buffer = []
        self.face_reg_timer = None
        self.face_lock = threading.Lock()

        self.binding_key = None
        self.master_key = None
        self.ble_thread = None
        self.ble_loop = None

        self.get_logger().info("Brain node initialized")
        self.get_logger().info(f"Device UUID: {self.device_uuid}")

    def _hardware_event_callback(self, request, response) -> None:
        """Process events reported by HardwareNode."""
        self.get_logger().info(f"Received HardwareEvent: {request.event_type}")

        try:
            payload_data = json.loads(request.payload) if request.payload else {}
        except Exception as e:
            response.success = False
            response.message = "Invalid JSON payload"
            self.get_logger().error(f"Invalid JSON payload: {e}")
            return response

        if request.event_type == "PIN_ENTERED":
            entered_pin = payload_data.get("pin", "")
            result = self.auth_manager.authenticate_pin(entered_pin)
            if result:
                self._send_hardware_command("UNLOCK")
                self._publish_auth_result(True, "PIN", 1.0)
                response.success = True
                response.message = "PIN Verified"
            else:
                self._publish_auth_result(False, "PIN", 0.0)
                response.success = False
                response.message = "Invalid PIN"

        elif request.event_type == "PIN_UPDATED":
            updated_pin = payload_data.get("pin", "")
            result = self.auth_manager.update_pin(updated_pin)
            if result:
                self._publish_auth_result(True, "PIN_UPDATED", 1.0)
                response.success = True
                response.message = "PIN Updated"
            else:
                self._publish_auth_result(False, "PIN_UPDATED", 0.0)
                response.success = False
                response.message = "Invalid PIN"

        elif request.event_type == "RFID_SCANNED":
            scanned_id = payload_data.get("rfid_id", "")
            result = self.auth_manager.authenticate_rfid(scanned_id)
            if result:
                self._send_hardware_command("UNLOCK")
                self._publish_auth_result(True, "RFID", 1.0)
                response.success = True
                response.message = "RFID Verified"
            else:
                self._publish_auth_result(False, "RFID", 0.0)
                response.success = False
                response.message = "Invalid RFID"

        elif request.event_type == "RFID_REGISTERED":
            scanned_id = payload_data.get("rfid_id", "")
            success, message = self.auth_manager.register_rfid_uid(scanned_id)
            if success:
                self._publish_auth_result(True, "RFID_REGISTERED", 1.0)
                response.success = True
                response.message = message
            else:
                self._publish_auth_result(False, "RFID_REGISTERED", 0.0)
                response.success = False
                response.message = message

        elif request.event_type == "FACE_REGISTERED":
            with self.face_lock:
                if self.is_registering_face:
                    response.success = False
                    response.message = "Already Active"
                    return response

                success = self._send_start_camera()
                if success:
                    self.is_registering_face = True
                    self.face_reg_buffer.clear()

                    if self.face_reg_timer:
                        self.face_reg_timer.cancel()
                    self.face_reg_timer = threading.Timer(20.0, self._face_registration_timeout)
                    self.face_reg_timer.start()

                    response.success = True
                    response.message = "Face Registering"
                else:
                    self._publish_auth_result(False, "FACE_REGISTERED", 0.0)
                    response.success = False
                    response.message = "Camera Error"

        elif request.event_type == "IR_TRIGGERED":
            success = self._send_start_camera()
            if success:
                with self.face_lock:
                    self.is_authenticating_face = True

                response.success = True
                response.message = "Camera Started"
            else:
                response.success = False
                response.message = "Camera Error"

        elif request.event_type == "RESET_TRIGGERED":
            initial_pin = self._initiate_reset_mode()
            response.success = True
            response.message = f"Init PIN: {initial_pin}"

        else:
            response.success = False
            response.message = "Unknown event type"
            self.get_logger().error(f"Unknown hardware event type: {request.event_type}")

        return response

    def _server_command_callback(self, request, response) -> None:
        """Process commands received from BridgeNode."""
        self.get_logger().info(f"Received ServerCommand: {request.command_type}")

        if request.command_type == "UNLOCK":
            success = self._send_hardware_command("UNLOCK")
            response.success = success
            response.message = "Command dispatched to hardware" if success else "Hardware unavailable"
        else:
            response.success = False
            response.message = "Unsupported command type"

        return response

    def _send_hardware_command(self, command_type: str) -> bool:
        """Send a command to the hardware node via Service."""
        if not self.hardware_command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Hardware node service not available")
            return False

        req = HardwareCommand.Request()
        req.timestamp = self.get_clock().now().to_msg()
        req.command_type = command_type
        self.hardware_command_client.call_async(req)
        return True

    def _send_start_camera(self) -> bool:
        """Start the camera."""
        if not self.start_camera_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("StartCamera service not available")
            return False

        req = StartCamera.Request()
        self.start_camera_client.call_async(req)
        return True

    def _face_vector_callback(self, msg: FaceVector) -> None:
        """Process face vector and authenticate.

        Args:
            msg: FaceVector message.
        """
        # Authenticate using face vector
        face_vector = np.array(msg.vector, dtype=np.float32)
        face_vector_list = face_vector.tolist()

        with self.face_lock:
            if self.is_registering_face:
                self._send_hardware_command("LED_Y")
                self.face_reg_buffer.append(face_vector_list)
                current_count = len(self.face_reg_buffer)

                if current_count >= 5:
                    if self.face_reg_timer:
                        self.face_reg_timer.cancel()

                    self.is_registering_face = False

                    self.auth_manager.register_face_vector(self.face_reg_buffer)
                    self.face_reg_buffer.clear()
                    self._publish_auth_result(True, "FACE_REGISTERED", 1.0)
                    self.get_logger().info("Face registration successful")

                return

            if not self.is_authenticating_face:
                return

        result = self.auth_manager.authenticate_face(face_vector)

        if result:
            is_recognized, confidence = result

            # Send unlock command to hardware
            if is_recognized:
                with self.face_lock:
                    self.is_authenticating_face = False

                self._send_hardware_command("UNLOCK")
                self._publish_auth_result(
                    authenticated=True,
                    auth_type="FACE",
                    confidence=confidence,
                )

    def _publish_auth_result(self, authenticated: bool, auth_type: str, confidence: float = 1.0) -> None:
        """Publish the result of any authentication attempt."""
        if not self.capture_image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("CaptureImage service not available")

        result_msg = AuthenticationResult()
        result_msg.timestamp = self.get_clock().now().to_msg()
        result_msg.authenticated = authenticated
        result_msg.auth_type = auth_type
        result_msg.confidence = float(confidence)

        req = CaptureImage.Request()
        future = self.capture_image_client.call_async(req)

        def _image_captured_done_callback(future):
            try:
                response = future.result()
                if response and response.success:
                    result_msg.image = response.image

            except Exception as e:
                self.get_logger().error(f"Failed to capture image for auth result: {e}")

            self.auth_result_pub.publish(result_msg)
            self.get_logger().info(f"{auth_type} authentication {'successful' if authenticated else 'failed'}")

        future.add_done_callback(_image_captured_done_callback)

    def _face_registration_timeout(self):
        """Handle face registration timeout."""
        with self.face_lock:
            if self.is_registering_face:
                self.is_registering_face = False
                self.face_reg_buffer.clear()
                self.get_logger().info("Face registration timed out")

    def _initiate_reset_mode(self) -> str:
        """Clears state, generates new binding key, and notifies BridgeNode."""
        self.master_key = None
        self._save_crypto_keys()
        self.auth_manager.reset_database()
        initial_pin = generate_initial_pin()
        self.auth_manager.update_pin(initial_pin)
        self.binding_key = generate_binding_key()
        self.ble_thread = threading.Thread(
            target=self._run_ble_task,
            args=(
                self.binding_key,
                initial_pin,
            ),
            daemon=True,
        )
        self.ble_thread.start()

        req = ProvisionCrypto.Request()
        req.binding_key = list(self.binding_key)
        req.master_key = []
        self.provision_crypto_client.call_async(req)
        self.get_logger().info("System Reset initiated: Binding Key generated.")
        return initial_pin

    def _store_master_key_callback(
        self, request: StoreMasterKey.Request, response: StoreMasterKey.Response
    ) -> StoreMasterKey.Response:
        """Receive Master Key from BridgeNode after successful cloud binding."""
        self.master_key = bytes(request.master_key)
        self._save_crypto_keys()

        if self.ble_loop and self.ble_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.reset_channel.stop_ble_reset_advertisement(),
                self.ble_loop,
            )
        response.success = True
        response.message = "Master Key stored successfully"
        self.get_logger().info("Device securely bound. Master Key stored.")
        return response

    def _load_crypto_keys(self):
        if self.crypto_db_path.exists():
            with open(self.crypto_db_path, "r") as f:
                data = json.load(f)
                self.binding_key = bytes.fromhex(data.get("binding_key", "")) if "binding_key" in data else None
                self.master_key = bytes.fromhex(data.get("master_key", "")) if "master_key" in data else None

    def _save_crypto_keys(self):
        os.makedirs(self.crypto_db_path.parent, exist_ok=True)

        data = {}
        if self.binding_key:
            data["binding_key"] = self.binding_key.hex()
        if self.master_key:
            data["master_key"] = self.master_key.hex()

        with open(self.crypto_db_path, "w") as f:
            json.dump(data, f)

    def _run_ble_task(self, binding_key, initial_pin):
        """Run BLE advertisement and GATT service in a separate thread."""
        self.ble_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ble_loop)
        try:
            self.ble_loop.run_until_complete(
                self.reset_channel.start_ble_reset_advertisement(
                    binding_key,
                    initial_pin,
                )
            )
        finally:
            if self.ble_loop is not None:
                try:
                    self.ble_loop.close()
                    self.ble_loop = None
                except Exception as e:
                    self.get_logger().error(f"Error occurred while closing BLE loop: {e}")

    def on_startup(self) -> None:
        """Called when node starts up."""
        self._load_crypto_keys()
        if self.binding_key and self.master_key:
            req = ProvisionCrypto.Request()
            req.binding_key = list(self.binding_key)
            req.master_key = list(self.master_key)
            self.provision_crypto_client.call_async(req)

    def on_shutdown(self) -> None:
        """Called when node shuts down."""
        if self.ble_loop and self.ble_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.reset_channel.stop_ble_reset_advertisement(),
                self.ble_loop,
            )
            if self.ble_thread:
                self.ble_thread.join(timeout=2.0)
        self.get_logger().info("Brain node shut down")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = BrainNode()
    node.on_startup()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
