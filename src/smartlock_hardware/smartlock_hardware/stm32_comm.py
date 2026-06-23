import os
import time
import serial
import threading
from typing import Any, Callable, Optional, cast
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pySerialTransfer import pySerialTransfer

from smartlock_hardware.protos import smart_lock_pb2
from smartlock_hardware.hardware_utils import get_default_logger


class STM32Communication:
    """Handles serial communication with STM32 microcontroller using Protobuf."""

    # AES parameters
    AES_KEY_SIZE = 32  # 256 bits
    GCM_NONCE_SIZE = 12  # 96 bits

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        aes_key: Optional[bytes] = None,
        on_message_received: Optional[Callable[[smart_lock_pb2.Stm32Message, int], None]] = None,
        logger: Optional[Any] = None,
    ):
        self.link = pySerialTransfer.SerialTransfer(port, baudrate)
        self.aes_key = aes_key
        self.on_message_received = on_message_received
        self.logger = logger or get_default_logger(__name__)

        self.sequence_number = 0
        self._running = False
        self._listen_thread: Optional[threading.Thread] = None
        self._tx_lock = threading.Lock()

        if self.aes_key is None:
            self.logger.warning("No AES key provided, communication will be unencrypted")
        elif len(self.aes_key) != self.AES_KEY_SIZE:
            self.logger.error(f"AES key must be {self.AES_KEY_SIZE} bytes long")
            raise ValueError(f"AES key must be {self.AES_KEY_SIZE} bytes long")
        else:
            self._aesgcm = AESGCM(self.aes_key)

    def connect(self) -> bool:
        """Establish serial connection and start background listening thread."""
        try:
            self.link.open()

            # Start background listener
            self._running = True
            self._listen_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._listen_thread.start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to STM32: {e}")
            return False

    def disconnect(self) -> None:
        """Close serial connection and stop listener."""
        self._running = False
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)

        self.link.close()
        self.logger.info("Disconnected from STM32")

    def send_pi_message(self, pi_msg: smart_lock_pb2.PiMessage) -> bool:
        """Wrap PiMessage in SmartLockPacket, encrypt, and send to STM32."""
        self.sequence_number += 1

        # Build Protobuf Packet
        packet = smart_lock_pb2.SmartLockPacket()
        packet.sequence = self.sequence_number
        packet.pi_message.CopyFrom(pi_msg)

        if not self._send_packet(packet):
            self.logger.error("Failed to send PiMessage to STM32")
            return False

        return True

    def send_pi_status(self, seq: int, status: smart_lock_pb2.StatusType.ValueType, message: str) -> bool:
        """Send status response back to STM32."""
        pi_msg = smart_lock_pb2.PiMessage()
        pi_msg.status_response.status = status
        pi_msg.status_response.message = message

        # Build Protobuf Packet
        packet = smart_lock_pb2.SmartLockPacket()
        packet.sequence = seq
        packet.pi_message.CopyFrom(pi_msg)

        if not self._send_packet(packet):
            self.logger.error("Failed to send status response to STM32")
            return False

        return True

    def _send_packet(self, packet: smart_lock_pb2.SmartLockPacket) -> bool:
        if not self.link.connection or not self.link.connection.is_open:
            self.logger.error("Serial port not connected")
            return False

        # Serialize
        tx_payload = packet.SerializeToString()

        # Encrypt
        if self.aes_key:
            encrypt_payload = self._encrypt_payload(tx_payload)
            if encrypt_payload is None:
                self.logger.error("Failed to encrypt payload, not sending")
                return False
        else:
            encrypt_payload = tx_payload

        try:
            with self._tx_lock:
                payload_size = len(encrypt_payload)

                self.link.tx_buff = [0] * len(self.link.tx_buff)  # type: ignore
                for i, b in enumerate(encrypt_payload):
                    self.link.tx_buff[i] = b  # type: ignore

                self.link.send(payload_size)
            self.logger.debug(f"Sent Protobuf PiMessage (Seq: {packet.sequence}, {payload_size} bytes)")
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            return False

        return True

    def _read_loop(self) -> None:
        """Background thread to continuously read and parse incoming frames."""
        while self._running:
            try:
                if not self.link.connection or not self.link.connection.is_open:
                    break

                if self.link.available():
                    rx_payload = bytes(cast(list[int], self.link.rx_buff[: self.link.bytes_read]))
                    if rx_payload:
                        self._process_incoming_payload(rx_payload)

                elif (
                    self.link.status != pySerialTransfer.Status.NO_DATA
                    and self.link.status != pySerialTransfer.Status.CONTINUE
                ):
                    if self.link.status == pySerialTransfer.Status.CRC_ERROR:
                        self.logger.error("CRC_ERROR: Received corrupted data")
                    elif self.link.status == pySerialTransfer.Status.PAYLOAD_ERROR:
                        self.logger.error("PAYLOAD_ERROR: Received corrupted data")
                    elif self.link.status == pySerialTransfer.Status.STOP_BYTE_ERROR:
                        self.logger.error("STOP_BYTE_ERROR: Received corrupted data")
                    else:
                        self.logger.error(f"Unknown error: {self.link.status}")

                time.sleep(0.001)

            except Exception as e:
                self.logger.error(f"Unexpected error in read loop: {e}")
                time.sleep(1.0)

    def _process_incoming_payload(self, rx_payload: bytes) -> None:
        """Decrypt payload and invoke callback with Protobuf message."""
        if self.aes_key:
            decrypted_payload = self._decrypt_payload(rx_payload)
            if decrypted_payload is None:
                self.logger.error("Failed to decrypt incoming payload")
                return
        else:
            decrypted_payload = rx_payload

        packet = smart_lock_pb2.SmartLockPacket()
        try:
            packet.ParseFromString(decrypted_payload)
            if packet.HasField("stm32_message"):
                if self.on_message_received:
                    self.on_message_received(packet.stm32_message, packet.sequence)
            else:
                self.logger.warning("Received SmartLockPacket but no stm32_message field.")
        except Exception as e:
            self.logger.error(f"Protobuf Parse Error: {e}")

    def _encrypt_payload(self, payload: bytes) -> Optional[bytes]:
        try:
            if not self.aes_key:
                return payload

            nonce = os.urandom(self.GCM_NONCE_SIZE)
            ciphertext = self._aesgcm.encrypt(nonce, payload, associated_data=None)
            encrypted = nonce + ciphertext

        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None

        return encrypted

    def _decrypt_payload(self, payload: bytes) -> Optional[bytes]:
        try:
            if not self.aes_key:
                return payload

            if len(payload) < self.GCM_NONCE_SIZE + 16:
                self.logger.error("Decrypted data too short")
                return None

            nonce = payload[: self.GCM_NONCE_SIZE]
            ciphertext = payload[self.GCM_NONCE_SIZE :]
            decrypt = self._aesgcm.decrypt(nonce, ciphertext, associated_data=None)

        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None

        return decrypt
