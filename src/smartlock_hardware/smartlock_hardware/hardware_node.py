"""Hardware node for smartlock system.

This ROS 2 node manages:
- Serial communication with STM32 microcontroller
- Listening to hardware commands from brain node
"""

import os
import json
from functools import partial
from typing import Optional
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory

from smartlock_hardware.protos import smart_lock_pb2
from smartlock_interfaces.srv import HardwareCommand, HardwareEvent
from smartlock_hardware.stm32_comm import STM32Communication


class HardwareNode(Node):
    """ROS 2 Node for hardware control and monitoring."""

    def __init__(self) -> None:
        """Initialize hardware node."""
        super().__init__("smartlock_hardware_node")

        # Declare parameters
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("serial_baudrate", 115200)

        # Get parameters with type assertions
        self.serial_port = self.get_parameter("serial_port").get_parameter_value().string_value
        self.serial_baudrate = self.get_parameter("serial_baudrate").get_parameter_value().integer_value

        # Initialize hardware components
        self.stm32_comm = STM32Communication(
            port=self.serial_port,
            baudrate=self.serial_baudrate,
            aes_key=self._load_aes_key(),
            on_message_received=self._on_stm32_message_received,
            logger=self.get_logger(),
        )

        self.callback_group = ReentrantCallbackGroup()

        # Create Service
        self.hardware_command_srv = self.create_service(
            HardwareCommand,
            "/smartlock/hardware_command",
            self._hardware_command_callback,
            callback_group=self.callback_group,
        )
        self.hardware_event_client = self.create_client(
            HardwareEvent,
            "/smartlock/hardware_event",
            callback_group=self.callback_group,
        )

        self.get_logger().info("Hardware node initialized")
        self.get_logger().info(f"Serial port: {self.serial_port}")

    def _load_aes_key(self) -> Optional[bytes]:
        """Load AES key from environment variable."""
        package_share_dir = get_package_share_directory("smartlock_hardware")
        env_path = os.path.join(package_share_dir, ".env")

        if os.path.exists(env_path):
            load_dotenv(env_path)
            self.get_logger().info(f"Loaded environment variables from {env_path}")
        else:
            self.get_logger().warning(f"No .env file found at {env_path}")

        # aes_key_str = os.getenv("AES_KEY")
        aes_key_str = None
        if aes_key_str:
            try:
                aes_key = bytes.fromhex(aes_key_str)
                if len(aes_key) != 32:
                    self.get_logger().error("AES key must be 32 bytes (64 hex characters)")
                    return None
                return aes_key
            except Exception as e:
                self.get_logger().error(f"Invalid AES key format in environment variable: {e}")
                return None
        else:
            self.get_logger().warning("AES_KEY not found in environment variables")
            return None

    def _hardware_command_callback(self, request, response) -> None:
        """Handle hardware command requests from brain node."""
        self.get_logger().info(f"Received Command: {request.command_type}")

        pi_msg = smart_lock_pb2.PiMessage()

        if request.command_type == "UNLOCK":
            pi_msg.unlock.SetInParent()
        elif request.command_type == "LED_Y":
            pi_msg.control_rgb_led.red = True
            pi_msg.control_rgb_led.green = True
            pi_msg.control_rgb_led.blue = False
        else:
            self.get_logger().error(f"Unknown command: {request.command_type}")
            response.success = False
            return response

        success = self.stm32_comm.send_pi_message(pi_msg)
        response.success = success
        return response

    def _on_stm32_message_received(self, stm32_msg: smart_lock_pb2.Stm32Message, seq: int) -> None:
        """Background callback triggered when STM32 sends a Protobuf message."""
        event_type = stm32_msg.WhichOneof("body")

        # Prepare Event Request for Brain Node
        req = HardwareEvent.Request()
        payload = {}

        if event_type == "pin_input":
            req.event_type = "PIN_ENTERED"
            payload["pin"] = stm32_msg.pin_input.pin
            self.get_logger().info(f"Pin received: {payload['pin']}")

        elif event_type == "rfid_scanned":
            req.event_type = "RFID_SCANNED"
            payload["rfid_id"] = stm32_msg.rfid_scanned.uid.hex()
            self.get_logger().info(f"RFID scanned: {payload['rfid_id']}")

        elif event_type == "rfid_registration":
            req.event_type = "RFID_REGISTERED"
            payload["rfid_id"] = stm32_msg.rfid_registration.uid.hex()
            self.get_logger().info(f"RFID registered event received")

        elif event_type == "face_registration":
            req.event_type = "FACE_REGISTERED"
            self.get_logger().info(f"Face registered event received")

        elif event_type == "ir_triggered":
            req.event_type = "IR_TRIGGERED"
            self.get_logger().info("IR Event Triggered")

        elif event_type == "system_reset":
            req.event_type = "RESET_TRIGGERED"
            self.get_logger().info("System reset event received")

        elif event_type == "status_response":
            self.get_logger().info(f"STM32 Status Response: {stm32_msg.status_response.message}")
            return
        else:
            self.get_logger().warning(f"Unhandled STM32 event type: {event_type}")
            return

        req.payload = json.dumps(payload)

        if self.hardware_event_client.wait_for_service(timeout_sec=1.0):
            future = self.hardware_event_client.call_async(req)
            future.add_done_callback(partial(self._hardware_event_done_callback, stm32_seq=seq))
        else:
            status = smart_lock_pb2.STATUS_TYPE_FAILED
            message = "Service Unavailable"
            self.stm32_comm.send_pi_status(seq, status, message)
            self.get_logger().error("Brain node HardwareEvent service unavailable!")

    def _hardware_event_done_callback(self, future, stm32_seq: int) -> None:
        """Handle response from brain node and optionally notify STM32."""
        try:
            response = future.result()

            # Send an ACK back to STM32
            if response.success:
                status = smart_lock_pb2.STATUS_TYPE_OK
            else:
                status = smart_lock_pb2.STATUS_TYPE_FAILED

            self.stm32_comm.send_pi_status(stm32_seq, status, response.message)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def on_startup(self) -> None:
        if not self.stm32_comm.connect():
            self.get_logger().error("Failed to connect to STM32")
            return
        self.get_logger().info("Connected to STM32 successfully")

    def on_shutdown(self) -> None:
        self.stm32_comm.disconnect()
        self.get_logger().info("Disconnected from STM32")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = HardwareNode()
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
