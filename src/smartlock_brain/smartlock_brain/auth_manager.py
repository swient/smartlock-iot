"""Authentication manager for smartlock brain.

This module handles:
- RFID uid database management
- Face vector comparison and matching
- PIN code verification
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Tuple, Optional, Set, List

from smartlock_brain.brain_utils import get_default_logger, compute_similarity


class AuthManager:
    """Manage authentication and credentials.

    Attributes:
        db_path (Path): Path to authentication database file.
    """

    # Face similarity threshold (0.0 to 1.0)
    FACE_SIMILARITY_THRESHOLD = 0.6

    def __init__(self, db_path: Path, logger: Optional[Any] = None):
        """Initialize authentication manager.

        Args:
            db_path: Path to authentication database JSON file.
        """
        self.db_path = db_path
        self.pin_hash: Optional[str] = None  # SHA256 hash of PIN
        self.rfid_uids: Set[str] = set()
        self.face_vectors: List[np.ndarray] = []
        self.logger = logger or get_default_logger(__name__)

        if not self._load_database():
            self.logger.info("Initialized new authentication database")

    def authenticate_rfid(self, rfid_uid: str) -> bool:
        """Authenticate by RFID uid.

        Args:
            rfid_uid: RFID uid value.

        Returns:
            True if authenticated, False otherwise.
        """
        if rfid_uid in self.rfid_uids:
            return True
        else:
            self.logger.warning(f"Unknown RFID uid: {rfid_uid}")
            return False

    def authenticate_pin(self, pin_code: str) -> bool:
        """Authenticate by PIN code.

        Args:
            pin_code: PIN code to verify.

        Returns:
            True if authenticated, False otherwise.
        """
        if not self.pin_hash:
            self.logger.warning("No PIN hash stored in database")
            return False

        computed_hash = hashlib.sha256(pin_code.encode()).hexdigest()

        if computed_hash == self.pin_hash:
            return True

        self.logger.warning("PIN authentication failed")
        return False

    def authenticate_face(self, face_vector: np.ndarray) -> Tuple[bool, float]:
        """Authenticate by face vector.

        Args:
            face_vector: 128-dimensional face feature vector.

        Returns:
            Tuple of (authenticated: bool, confidence: float).
        """
        best_similarity = 0.0

        for stored_vector in self.face_vectors:
            similarity = compute_similarity(
                face_vector,
                np.array(stored_vector, dtype=np.float32),
            )

            if similarity > best_similarity:
                best_similarity = similarity

        self.logger.info(f"Best face similarity: {best_similarity:.2f}")
        if best_similarity >= self.FACE_SIMILARITY_THRESHOLD:
            return True, best_similarity

        return False, best_similarity

    def update_pin(self, new_pin: str) -> bool:
        """Update the stored PIN code.

        Args:
            new_pin: New PIN code to set.

        Returns:
            True if updated successfully, False otherwise.
        """
        try:
            self.pin_hash = hashlib.sha256(new_pin.encode()).hexdigest()
            self._save_database()
            self.logger.info("PIN code updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update PIN code: {e}")
            return False

        return True

    def register_rfid_uid(self, rfid_uid: str) -> Tuple[bool, str]:
        """Register new RFID UID to database.

        Args:
            rfid_uid: RFID UID value to register.

        Returns:
            Tuple of (success: bool, message: str).
        """
        if rfid_uid in self.rfid_uids:
            self.logger.warning(f"RFID UID {rfid_uid} already exists")
            return False, "Duplicate RFID"

        self.rfid_uids.add(rfid_uid)
        self._save_database()
        self.logger.info(f"Registered RFID UID: {rfid_uid}")
        return True, "RFID Registered"

    def delete_rfid_uid(self, rfid_uid: str) -> bool:
        """Delete RFID UID from database.

        Args:
            rfid_uid: RFID UID value to remove.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if rfid_uid in self.rfid_uids:
            self.rfid_uids.remove(rfid_uid)
            self._save_database()
            self.logger.info(f"Deleted RFID UID: {rfid_uid}")
        else:
            self.logger.error(f"RFID UID {rfid_uid} not found")
            return False

        return True

    def register_face_vector(self, face_vectors: List[np.ndarray]) -> bool:
        """Register new face vectors to database.

        Args:
            face_vectors: List of face feature vectors.

        Returns:
            True if registered successfully.
        """
        face_vectors_mean = np.mean(face_vectors, axis=0)
        self.face_vectors.append(face_vectors_mean)
        self._save_database()
        self.logger.info("Registered new face vectors")
        return True

    def clear_face_vectors(self) -> bool:
        """Clear all face vectors from database.

        Returns:
            True indicating success.
        """
        self.face_vectors.clear()
        self._save_database()
        self.logger.info("Cleared all face vectors")
        return True

    def reset_database(self) -> bool:
        """Reset the entire authentication database.

        Returns:
            True if reset successfully, False otherwise.
        """
        try:
            self.pin_hash = None
            self.rfid_uids.clear()
            self.face_vectors.clear()
            self._save_database()
            self.logger.info("Authentication database reset successfully")

        except Exception as e:
            self.logger.error(f"Failed to reset database: {e}")
            return False

        return True

    def _load_database(self) -> bool:
        """Load authentication database from file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.db_path.exists():
            self.logger.warning(f"Database file not found: {self.db_path}")
            return False

        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)

            self.pin_hash = data.get("pin_hash")
            self.rfid_uids = set(data.get("rfid_uids", []))
            serialized_vectors = data.get("face_vectors", [])
            self.face_vectors = [np.array(vector) for vector in serialized_vectors]
            self.logger.info(f"Loaded RFID UIDs and face vectors from database")

        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            return False

        return True

    def _save_database(self) -> bool:
        """Save authentication database to file.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)

            serialized_vectors = [vector.tolist() for vector in self.face_vectors]
            data = {
                "pin_hash": self.pin_hash,
                "rfid_uids": list(self.rfid_uids),
                "face_vectors": serialized_vectors,
            }
            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Saved database with RFID UIDs and face vectors")

        except Exception as e:
            self.logger.error(f"Failed to save database: {e}")
            return False

        return True
