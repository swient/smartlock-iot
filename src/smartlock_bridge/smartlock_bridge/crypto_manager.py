"""Encryption and key management module for smartlock bridge.

This module handles:
- ECDH key agreement for Master Key establishment
- Session Key derivation using HKDF
- AES-256-GCM encryption/decryption using Session Keys
"""

import os
import hmac
import hashlib
from typing import Any, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from smartlock_bridge.bridge_utils import get_default_logger


class CryptoManager:
    """Manage cryptographic operations for cloud communication.

    Uses ECDH for key exchange and AES-256-GCM for symmetric encryption.
    """

    # AES parameters
    AES_KEY_SIZE = 32  # 256 bits
    GCM_NONCE_SIZE = 12  # 96 bits

    def __init__(
        self,
        binding_key: bytes,
        master_key: Optional[bytes] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize crypto manager."""
        self.binding_key = binding_key
        self.master_key = master_key
        self.private_key = None
        self.logger = logger or get_default_logger(__name__)

    def generate_ecdh_keypair(self) -> tuple[bytes, bytes]:
        """Generate ECDH key pair for this device.

        Returns:
            A tuple containing the public key and its HMAC.
        """
        try:
            self.private_key = ec.generate_private_key(ec.SECP256R1())
            public_key = self.private_key.public_key()

            # Serialize public key to uncompressed point format
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
            hmac_bytes = hmac.new(self.binding_key, public_bytes, hashlib.sha256).digest()
            self.logger.info("Generated ECDH key pair")

        except Exception as e:
            self.logger.error(f"Failed to generate ECDH keypair: {e}")
            raise

        return public_bytes, hmac_bytes

    def derive_master_key(
        self,
        server_public_bytes: bytes,
        server_hmac_bytes: bytes,
    ) -> bytes:
        """Perform ECDH key agreement with server public key.

        Args:
            server_public_bytes: Raw uncompressed bytes of server public key.
            server_hmac_bytes: HMAC of the server public key.

        Returns:
            The derived master key if agreement succeeds.
        """
        try:
            if self.private_key is None:
                self.logger.error("Device private key not initialized or already used")
                raise RuntimeError("Device private key not initialized or already used")

            expected_hmac = hmac.new(self.binding_key, server_public_bytes, hashlib.sha256).digest()

            if not hmac.compare_digest(expected_hmac, server_hmac_bytes):
                self.logger.error("Server HMAC verification failed.")
                raise ValueError("Server HMAC verification failed.")

            # Decode server public key
            server_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(),
                server_public_bytes,
            )

            # Perform ECDH
            shared_secret = self.private_key.exchange(
                ec.ECDH(),
                server_public_key,
            )

            # Derive master key using HKDF
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=self.AES_KEY_SIZE,
                salt=None,
                info=b"smartlock-master-key",
            )

            self.master_key = hkdf.derive(shared_secret)
            self.logger.info("ECDH key agreement completed")

        except Exception as e:
            self.master_key = None
            self.logger.error(f"ECDH key agreement failed: {e}")
            raise
        finally:
            self.private_key = None

        return self.master_key

    def derive_session_key(self, session_salt: bytes) -> Optional[bytes]:
        """Derive session key from master key and session salt.

        Args:
            session_salt: Unique salt for this session.

        Returns:
            The derived session key if successful, None otherwise.
        """
        try:
            if self.master_key is None:
                self.logger.error("AES master key not set")
                return None

            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=self.AES_KEY_SIZE,
                salt=session_salt,
                info=b"smartlock-session-key",
            )

            session_key = hkdf.derive(self.master_key)
            self.logger.info("Session key derived successfully")

        except Exception as e:
            self.logger.error(f"Failed to derive session key: {e}")
            return None

        return session_key

    def encrypt_data(self, plaintext: bytes, session_key: bytes) -> Optional[bytes]:
        """Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt.
            session_key: The session key to use for encryption.

        Returns:
            Encrypted data as bytes.
        """
        try:
            if session_key is None or len(session_key) != self.AES_KEY_SIZE:
                self.logger.error("Invalid AES session key")
                return None

            # Encrypt
            nonce = os.urandom(self.GCM_NONCE_SIZE)
            aesgcm = AESGCM(session_key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None

        return nonce + ciphertext

    def decrypt_data(self, ciphertext: bytes, session_key: bytes) -> Optional[bytes]:
        """Decrypt data using AES-256-GCM.

        Args:
            ciphertext: Data to decrypt.
            session_key: The session key to use for decryption.

        Returns:
            Decrypted plaintext bytes.
        """
        try:
            if session_key is None or len(session_key) != self.AES_KEY_SIZE:
                self.logger.error("Invalid AES session key")
                return None

            if len(ciphertext) < self.GCM_NONCE_SIZE + 16:
                self.logger.error("Ciphertext too short")
                return None

            # Decrypt
            nonce = ciphertext[: self.GCM_NONCE_SIZE]
            ciphertext = ciphertext[self.GCM_NONCE_SIZE :]
            aesgcm = AESGCM(session_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None

        return plaintext
