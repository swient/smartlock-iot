"""Bridge node for smartlock cloud communication.

This ROS 2 node manages:
- Communication with cloud server
- ECDH key exchange and AES encryption
- Device binding and pairing
- Authentication log synchronization
"""

import os
import cv2
import json
import queue
import base64
import threading
import numpy as np
import paho.mqtt.client as mqtt
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from smartlock_interfaces.msg import AuthenticationResult
from smartlock_interfaces.srv import ProvisionCrypto, ServerCommand, StoreMasterKey
from smartlock_bridge.crypto_manager import CryptoManager


class BridgeNode(Node):
    """ROS 2 Node for cloud communication bridge."""

    def __init__(self) -> None:
        """Initialize bridge node."""
        super().__init__("smartlock_bridge_node")

        # Declare parameters
        self.declare_parameter("mqtt_broker", "140.125.33.181")
        self.declare_parameter("mqtt_port", 1883)
        self.declare_parameter("device_uuid", "001")

        # Get parameters
        self.mqtt_broker = self.get_parameter("mqtt_broker").get_parameter_value().string_value
        self.mqtt_port = self.get_parameter("mqtt_port").get_parameter_value().integer_value
        self.device_uuid = self.get_parameter("device_uuid").get_parameter_value().string_value

        self.callback_group = ReentrantCallbackGroup()

        # Create services
        self.provision_srv = self.create_service(
            ProvisionCrypto,
            "/smartlock/provision_crypto",
            self._provision_crypto_callback,
            callback_group=self.callback_group,
        )

        # Create subscription
        self.auth_result_sub = self.create_subscription(
            AuthenticationResult,
            "/smartlock/auth_result",
            self._auth_result_callback,
            10,
            callback_group=self.callback_group,
        )

        # Create timers
        self.sync_logs_timer = self.create_timer(
            10.0,
            self._sync_logs_timer_callback,
            callback_group=self.callback_group,
        )

        self.mqtt_queue_timer = self.create_timer(
            0.05,
            self._process_mqtt_queue_callback,
            callback_group=self.callback_group,
        )

        self.server_command_client = self.create_client(ServerCommand, "/smartlock/server_command")
        self.store_master_key_client = self.create_client(StoreMasterKey, "/smartlock/store_master_key")

        # Initialize crypto manager
        self.crypto_manager: Optional[CryptoManager] = None
        self._crypto_lock = threading.Lock()

        # Device binding state
        self.is_bound = False
        self._current_transaction_id = None
        self._temp_master_key = None

        # Authentication log
        self.auth_logs: list = []
        self._auth_logs_lock = threading.Lock()
        self._pending_logs: dict = {}
        self._pending_logs_lock = threading.Lock()

        # MQTT client
        self.base_topic = f"device/smartlock/{self.device_uuid}"
        self.mqtt_client = mqtt.Client(client_id=f"lock-bridge-{self.device_uuid}")
        self.mqtt_client.username_pw_set(username="group01", password="1234")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_publish = self._on_mqtt_publish
        self._mqtt_queue: queue.Queue = queue.Queue()

        self.get_logger().info("Bridge node initialized")
        self.get_logger().info(f"Device UUID: {self.device_uuid}")
        self.get_logger().info(f"MQTT Broker: {self.mqtt_broker}:{self.mqtt_port}")

    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            topic = f"{self.base_topic}/downlink/+"
            client.subscribe(topic)
            self.get_logger().info(f"Subscribed to MQTT topic: {topic}")
        else:
            self.get_logger().error(f"Failed to connect to MQTT broker, return code {rc}")

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        try:
            topic_layers = msg.topic.split("/")
            action_type = topic_layers[-1] if topic_layers else ""
            payload_data = json.loads(msg.payload.decode("utf-8"))
            self._mqtt_queue.put((action_type, payload_data))

        except Exception as e:
            self.get_logger().error(f"Error buffering MQTT message: {e}")

    def _on_mqtt_publish(self, client, userdata, mid) -> None:
        with self._pending_logs_lock:
            if mid in self._pending_logs:
                sent_log_ids = self._pending_logs.pop(mid)

                with self._auth_logs_lock:
                    self.auth_logs = [log for log in self.auth_logs if log["timestamp_ms"] not in sent_log_ids]

                self.get_logger().debug(f"Log sync confirmed for message ID {mid}")

    def _process_mqtt_queue_callback(self) -> None:
        """Process buffered MQTT messages for binding and control commands."""
        while not self._mqtt_queue.empty():
            try:
                action_type, payload_data = self._mqtt_queue.get_nowait()

                if action_type == "bind_req":
                    if self._process_device_binding(payload_data):
                        self.get_logger().info("Device binding processed successfully")
                    else:
                        self.get_logger().error("Failed to process device binding")

                elif action_type == "bind_ack":
                    self._process_bind_ack(payload_data)

                elif action_type == "control":
                    if self._process_control_command(payload_data):
                        self.get_logger().info("Control command processed successfully")
                    else:
                        self.get_logger().error("Failed to process control command")

                else:
                    self.get_logger().warn(f"Received MQTT message with unknown action type: {action_type}")

            except queue.Empty:
                break
            except Exception as e:
                self.get_logger().error(f"Error processing MQTT message: {e}")

    def _process_device_binding(self, payload_data: dict) -> bool:
        """Process device binding with cloud server.

        Args:
            payload_data: Data received from the cloud server.

        Returns:
            True if binding successful, False otherwise.
        """
        if self.is_bound or self.crypto_manager is None:
            self.get_logger().error("Device is already bound or crypto manager is not provisioned.")
            return False

        received_tx_id = payload_data.get("transaction_id", "")
        if not received_tx_id:
            self.get_logger().error("Transaction ID missing in binding response.")
            return False

        try:
            server_pub_bytes = base64.b64decode(payload_data["public_key"])
            server_hmac_bytes = base64.b64decode(payload_data["hmac"])

            with self._crypto_lock:
                public_bytes, hmac_bytes = self.crypto_manager.generate_ecdh_keypair()
                master_key = self.crypto_manager.derive_master_key(server_pub_bytes, server_hmac_bytes)

            if master_key:
                self._temp_master_key = master_key
                self._current_transaction_id = received_tx_id
                res_topic = f"{self.base_topic}/uplink/bind_res"
                res_payload = {
                    "transaction_id": received_tx_id,
                    "public_key": base64.b64encode(public_bytes).decode("utf-8"),
                    "hmac": base64.b64encode(hmac_bytes).decode("utf-8"),
                }
                self.mqtt_client.publish(res_topic, json.dumps(res_payload), qos=1)
                return True
            else:
                self.is_bound = False
                self.get_logger().error("Failed to derive master key from cloud request.")
                return False

        except Exception as e:
            self.is_bound = False
            self.get_logger().error(f"Failed to process device binding: {e}")
            return False

    def _process_control_command(self, payload_data: dict) -> bool:
        """Process control command from cloud server.

        Args:
            payload_data: Data received from the cloud server.

        Returns:
            True if command processed successfully, False otherwise.
        """
        if not self.server_command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Server command service is not available.")
            return False

        if not self.is_bound or self.crypto_manager is None:
            self.get_logger().error("Device is not bound or crypto manager is not provisioned.")
            return False

        try:
            session_salt = bytes.fromhex(payload_data["session_salt"])
            ciphertext = base64.b64decode(payload_data["ciphertext"])

            with self._crypto_lock:
                session_key = self.crypto_manager.derive_session_key(session_salt)

            if not session_key:
                self.get_logger().error("Failed to derive session key for control command.")
                return False

            plaintext = self.crypto_manager.decrypt_data(ciphertext, session_key)
            if plaintext:
                command_dict = json.loads(plaintext.decode("utf-8"))

                req = ServerCommand.Request()
                req.command_type = command_dict.get("command_type", "")
                req.payload = json.dumps(command_dict.get("command_payload", {}))
                future = self.server_command_client.call_async(req)
                future.add_done_callback(self._server_command_response_callback)
                return True
            else:
                self.get_logger().error("Failed to decrypt control command payload.")
                return False

        except Exception as e:
            self.get_logger().error(f"Failed to process control command: {e}")
            return False

    def _process_bind_ack(self, payload_data: dict) -> None:
        """Process bind_ack from cloud server."""
        if not self.store_master_key_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Store master key service is not available.")
            return

        received_tx_id = payload_data.get("transaction_id", "")
        status = payload_data.get("status", "")

        if received_tx_id != self._current_transaction_id or not self._temp_master_key:
            self.get_logger().warn("Ignored bind_ack: Transaction ID mismatch or no temp key.")
            return

        if status == "success":
            req = StoreMasterKey.Request()
            req.master_key = list(self._temp_master_key)
            future = self.store_master_key_client.call_async(req)
            future.add_done_callback(self._store_master_key_done_callback)
            self.get_logger().info("Sent master key to brain node for storage.")

    def _provision_crypto_callback(self, request, response) -> None:
        try:
            binding_key = bytes(request.binding_key)
            if len(binding_key) != 32:
                response.success = False
                response.message = "Invalid binding key length"
                self.get_logger().error(response.message)
                return response

            if len(request.master_key) == 0:  # reset
                self.is_bound = False
                with self._auth_logs_lock:
                    self.auth_logs.clear()

                self.crypto_manager = CryptoManager(binding_key=binding_key, logger=self.get_logger())

                response.success = True
                response.message = "CryptoManager initialized for Reset."
                self.get_logger().info("CryptoManager initialized for Reset.")
            elif len(request.master_key) == 32:  # provision
                master_key = bytes(request.master_key)
                self.crypto_manager = CryptoManager(
                    binding_key=binding_key, master_key=master_key, logger=self.get_logger()
                )
                self.is_bound = True
                response.success = True
                response.message = "Crypto manager provisioned successfully."
                self.get_logger().info("Crypto manager provisioned successfully")
            else:
                response.success = False
                response.message = "Invalid master key length"
                self.get_logger().error(response.message)
                return response

        except Exception as e:
            response.success = False
            response.message = f"Failed to provision crypto manager: {e}"
            self.get_logger().error(response.message)

        return response

    def _auth_result_callback(self, msg: AuthenticationResult) -> None:
        """Handle authentication result from brain node.

        Log authentication events for cloud sync.

        Args:
            msg: AuthenticationResult message.
        """
        img_data = msg.image.data
        height = msg.image.height
        width = msg.image.width
        cv_img_raw = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
        cv_img_resized = cv2.resize(cv_img_raw, (int(width / 1.5), int(height / 1.5)))
        success, image = cv2.imencode(".jpg", cv_img_resized)
        image_b64 = base64.b64encode(image).decode("utf-8")

        timestamp_ms = (msg.timestamp.sec * 1000) + msg.timestamp.nanosec // 1_000_000

        # Add to authentication log
        log_entry = {
            "timestamp_ms": timestamp_ms,
            "image": image_b64,
            "authenticated": msg.authenticated,
            "auth_type": msg.auth_type,
            "confidence": msg.confidence,
        }

        with self._auth_logs_lock:
            self.auth_logs.append(log_entry)
            if len(self.auth_logs) > 100:
                self.auth_logs.pop(0)

        self.get_logger().info(f"Authentication result logger")

    def _sync_logs_timer_callback(self) -> None:
        """Periodically sync authentication logs with cloud server."""
        if not self.is_bound or self.crypto_manager is None:
            self.get_logger().debug("Device not ready or not paired, skipping sync")
            return

        with self._auth_logs_lock:
            if not self.auth_logs:
                return
            logs_to_send = list(self.auth_logs)

        self.get_logger().info(f"Syncing {len(logs_to_send)} authentication records to cloud")

        try:
            session_salt = os.urandom(16)
            with self._crypto_lock:
                session_key = self.crypto_manager.derive_session_key(session_salt)

            if not session_key:
                self.get_logger().error("Failed to derive session key, aborting log sync")
                return

            plaintext = json.dumps(logs_to_send).encode("utf-8")
            ciphertext = self.crypto_manager.encrypt_data(plaintext, session_key)

            if ciphertext:
                upload_payload = {
                    "session_salt": session_salt.hex(),
                    "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
                }

                logs_topic = f"{self.base_topic}/uplink/logs"
                msg_info = self.mqtt_client.publish(logs_topic, json.dumps(upload_payload), qos=1)
                if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
                    with self._pending_logs_lock:
                        self._pending_logs[msg_info.mid] = {log["timestamp_ms"] for log in logs_to_send}

                self.get_logger().info(f"Logs encrypted successfully ({len(plaintext)} bytes)")
            else:
                self.get_logger().error("Failed to encrypt logs")

        except Exception as e:
            self.get_logger().error(f"Failed to sync logs: {e}")

    def _server_command_response_callback(self, future) -> None:
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Command executed successfully: {response.message}")
            else:
                self.get_logger().error(f"Command execution failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Error in execute command response callback: {e}")

    def _store_master_key_done_callback(self, future) -> None:
        try:
            response = future.result()
            if response.success:
                self.is_bound = True
                self.get_logger().info("Master key stored successfully in brain node")
            else:
                self.is_bound = False
                self.get_logger().error(f"Failed to store master key in brain node: {response.message}")
        except Exception as e:
            self.is_bound = False
            self.get_logger().error(f"Error in store master key response callback: {e}")

    def on_startup(self) -> None:
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_start()
            self.get_logger().info("MQTT network loop started in background")
        except Exception as e:
            self.get_logger().error(f"MQTT connection failed: {e}")

    def on_shutdown(self) -> None:
        """Called when node shuts down."""
        try:
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()
            self.get_logger().info("MQTT client disconnected successfully")
        except Exception as e:
            self.get_logger().error(f"Error disconnecting MQTT client: {e}")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = BridgeNode()
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
