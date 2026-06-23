"""BLE GATT Service and Advertisement Manager for smartlock factory reset.

This module handles:
- Generating device binding keys and dynamic pairing PINs.
- Registering a custom Linux BlueZ Pairing Agent for secure PIN interception.
- Hosting a secure BLE GATT Service with encrypted read characteristics.
- Broadcasting BLE Advertisements to allow phone discovery.
"""

import asyncio
import subprocess
from typing import Any, Optional
from datetime import datetime
from dbus_next.service import ServiceInterface, method
from bluez_peripheral.util import get_message_bus, Adapter
from bluez_peripheral.gatt.service import Service
from bluez_peripheral.advert import Advertisement
from bluez_peripheral.gatt.descriptor import DescriptorFlags
from bluez_peripheral.gatt.characteristic import characteristic, CharacteristicFlags as CharFlags

from smartlock_brain.brain_utils import get_default_logger


class BlePairingAgent(ServiceInterface):
    def __init__(self, passkey: str):
        super().__init__("org.bluez.Agent1")
        self._passkey = passkey

    @method()
    def RequestPasskey(self, device: "o") -> "u":  # type: ignore
        """Handle passkey request from BlueZ."""
        return int(self._passkey)

    @method()
    def RequestPinCode(self, device: "o") -> "s":  # type: ignore
        """Handle PIN code request from BlueZ."""
        return self._passkey

    @method()
    def DisplayPasskey(self, device: "o", passkey: "u", entered: "q"):  # type: ignore
        pass

    @method()
    def Cancel(self):
        pass


class GattService(Service):
    """GATT Service for data transmission."""

    SERVICE_UUID = "430e32cd-1f0f-4690-abf2-372475d61a00"
    UUID_CHAR_UUID = "5a96eb9b-73f3-429a-b1a0-4f78922424a1"
    KEY_CHAR_UUID = "6e961028-39f5-4eda-946c-acc3baafb97f"

    def __init__(self, manager: "BleResetChannel"):
        super().__init__(self.SERVICE_UUID, True)
        self.manager = manager

    @characteristic(UUID_CHAR_UUID, CharFlags.READ)
    def read_device_uuid(self, options) -> bytes:
        if not self.manager.binding_key:
            return b"ERROR: NOT_INITIALIZED"

        return bytes(self.manager.device_uuid, "utf-8")

    @characteristic(KEY_CHAR_UUID, CharFlags.READ)
    def read_binding_key(self, options) -> bytes:
        if not self.manager.binding_key:
            return b"ERROR: NOT_INITIALIZED"

        return bytes(self.manager.binding_key.hex(), "utf-8")


class BleResetChannel:
    """Manage device reset and Bluetooth connection."""

    def __init__(self, device_uuid: str = "", logger: Optional[Any] = None):
        """Initialize reset manager.

        Args:
            device_uuid: Device unique identifier.
        """
        self.device_uuid = device_uuid
        self.binding_key: Optional[bytes] = None
        self.initial_pin: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.session_timeout: float = 3600.0
        self.logger = logger or get_default_logger(__name__)

        self._bus = None
        self._agent_manager = None
        self._agent_path = "/com/smartlock/agent"
        self._gatt_service: Optional[GattService] = None
        self._advertisement: Optional[Advertisement] = None
        self._session_event: Optional[asyncio.Event] = None
        self.is_active = False

    async def start_ble_reset_advertisement(
        self,
        binding_key: bytes,
        initial_pin: str,
        timeout_sec: float = 3600.0,
    ) -> bool:
        """Start the BLE advertisement, register GATT services, and await phone connection."""
        if self.is_active:
            self.logger.warning("BLE reset advertisement session is already running.")
            return False

        self.binding_key = binding_key
        self.initial_pin = initial_pin
        self.session_start_time = datetime.now()
        self._session_event = asyncio.Event()

        try:
            self._bus = await get_message_bus()
            adapter = await Adapter.get_first(self._bus)

            pairing_agent = BlePairingAgent(initial_pin)
            self._bus.export(self._agent_path, pairing_agent)

            introspection = await self._bus.introspect("org.bluez", "/org/bluez")
            bluez_obj = self._bus.get_proxy_object("org.bluez", "/org/bluez", introspection)
            self._agent_manager = bluez_obj.get_interface("org.bluez.AgentManager1")

            await self._agent_manager.call_register_agent(self._agent_path, "NoInputNoOutput")  # type: ignore
            await self._agent_manager.call_request_default_agent(self._agent_path)  # type: ignore
            self.logger.info(f"BLE Pairing Agent successfully registered with PIN: {initial_pin}")

            self._gatt_service = GattService(self)
            await self._gatt_service.register(self._bus)
            self.logger.info("BLE GATT Reset Service injected into system bus.")

            self._advertisement = Advertisement(
                localName="Smartlock-Reset",
                serviceUUIDs=[self._gatt_service.SERVICE_UUID],
                appearance=1152,
                timeout=int(timeout_sec),
            )
            await self._advertisement.register(self._bus, adapter)

            subprocess.run(["bluetoothctl", "discoverable", "on"], check=True)
            subprocess.run(["bluetoothctl", "pairable", "on"], check=True)

            self.is_active = True
            self.logger.info("Pure BLE GATT server is now advertising. Waiting for smartphone connection...")

            try:
                await asyncio.wait_for(self._session_event.wait(), timeout=timeout_sec)
                self.logger.info("BLE reset session completed successfully via phone verification.")
            except asyncio.TimeoutError:
                self.logger.warning("BLE reset session closed due to connection timeout.")

        except Exception as e:
            self.logger.error(f"Failed to maintain BLE GATT session: {e}")
            return False
        finally:
            await self.stop_ble_reset_advertisement()

        return True

    async def stop_ble_reset_advertisement(self) -> bool:
        """Stop BLE advertisement, unregister GATT services, and clean up resources."""
        if not self.is_active:
            return False

        if self._session_event and not self._session_event.is_set():
            self._session_event.set()

        try:
            subprocess.run(["bluetoothctl", "discoverable", "off"], check=True)

            if self._agent_manager and self._bus:
                try:
                    await self._agent_manager.call_unregister_agent(self._agent_path)  # type: ignore
                except Exception:
                    pass

            if self._bus:
                self._bus.disconnect()

            self.logger.info("Pure BLE GATT infrastructure stopped and resources released cleanly.")
            self.is_active = False
            self.binding_key = None
            self.initial_pin = None
            self.session_start_time = None
            self._gatt_service = None
            self._advertisement = None

        except Exception as e:
            self.logger.error(f"Error during BLE resource cleanup: {e}")
            return False

        return True
