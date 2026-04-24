"""User authentication, registration codes, and brute-force protection.

Manages the user whitelist, registration code lifecycle, and rate limiting
for unknown numbers. All database operations use aiosqlite.

Security invariants:
- Unknown numbers receive NO response (silent drop)
- Registration codes are single-use, time-limited, cryptographically random
- Failed attempts are rate-limited with temp/permanent blocking
- The first admin is bootstrapped via a code on the physical screen

Usage:
    from boxbot.communication.auth import AuthManager

    auth = AuthManager()
    await auth.init_db()

    if await auth.is_authorized("+15551234567"):
        user = await auth.get_user("+15551234567")
"""

from __future__ import annotations

import logging
import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path("data/auth/users.db")

# Registration defaults (overridden by whatsapp.yaml if loaded)
DEFAULT_CODE_LENGTH = 6
DEFAULT_CODE_EXPIRY_SECONDS = 600  # 10 minutes
DEFAULT_MAX_CODES_PER_HOUR = 3
DEFAULT_MAX_ATTEMPTS_PER_WINDOW = 5
DEFAULT_ATTEMPT_WINDOW_SECONDS = 600  # 10 minutes
DEFAULT_TEMP_BLOCK_DURATION_SECONDS = 3600  # 1 hour
DEFAULT_MAX_TEMP_BLOCKS = 3


# Singleton accessor — set by main during startup, read by the output dispatcher
# so the agent loop can resolve person-name → phone-number without being passed
# a reference. Mirrors the voice session singleton pattern.
_auth_manager: "AuthManager | None" = None


def get_auth_manager() -> "AuthManager | None":
    """Return the process-wide AuthManager instance, or None if unset."""
    return _auth_manager


def set_auth_manager(manager: "AuthManager | None") -> None:
    """Register the process-wide AuthManager instance."""
    global _auth_manager
    _auth_manager = manager


@dataclass(frozen=True)
class User:
    """A registered boxBot user."""

    phone: str
    name: str
    role: str  # "admin" or "user"
    registered_at: str
    last_seen: str | None = None


@dataclass(frozen=True)
class RegisterResult:
    """Result of a registration attempt."""

    success: bool
    error: str | None = None
    user: User | None = None


class AuthManager:
    """Manages user authentication, registration codes, and rate limiting.

    Args:
        db_path: Path to the SQLite database. Defaults to data/auth/users.db.
        code_length: Length of generated registration codes.
        code_expiry: Code expiry in seconds.
        max_codes_per_hour: Maximum code generations per admin per hour.
        max_attempts_per_window: Failed attempts before temp block.
        attempt_window: Rate-limit window in seconds.
        temp_block_duration: Temp block duration in seconds.
        max_temp_blocks: Temp blocks before permanent block.
    """

    def __init__(
        self,
        db_path: Path | str = DB_PATH,
        *,
        code_length: int = DEFAULT_CODE_LENGTH,
        code_expiry: int = DEFAULT_CODE_EXPIRY_SECONDS,
        max_codes_per_hour: int = DEFAULT_MAX_CODES_PER_HOUR,
        max_attempts_per_window: int = DEFAULT_MAX_ATTEMPTS_PER_WINDOW,
        attempt_window: int = DEFAULT_ATTEMPT_WINDOW_SECONDS,
        temp_block_duration: int = DEFAULT_TEMP_BLOCK_DURATION_SECONDS,
        max_temp_blocks: int = DEFAULT_MAX_TEMP_BLOCKS,
    ) -> None:
        self._db_path = Path(db_path)
        self._code_length = code_length
        self._code_expiry = code_expiry
        self._max_codes_per_hour = max_codes_per_hour
        self._max_attempts_per_window = max_attempts_per_window
        self._attempt_window = attempt_window
        self._temp_block_duration = temp_block_duration
        self._max_temp_blocks = max_temp_blocks

    async def init_db(self) -> None:
        """Create database tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    phone TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    registered_at TEXT NOT NULL,
                    last_seen TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS registration_codes (
                    code TEXT PRIMARY KEY,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_by TEXT,
                    used_at TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS failed_attempts (
                    phone TEXT NOT NULL,
                    attempted_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    phone TEXT PRIMARY KEY,
                    blocked_at TEXT NOT NULL,
                    permanent INTEGER NOT NULL DEFAULT 0,
                    temp_block_count INTEGER NOT NULL DEFAULT 1,
                    expires_at TEXT
                )
            """)
            await db.commit()
            logger.info("Auth database initialized at %s", self._db_path)

    def _get_db(self) -> aiosqlite.Connection:
        """Open a database connection.

        Returns an un-awaited ``aiosqlite.Connection`` that callers use as
        an async context manager:  ``async with self._get_db() as db:``
        """
        return aiosqlite.connect(self._db_path)

    # -----------------------------------------------------------------
    # User management
    # -----------------------------------------------------------------

    async def is_authorized(self, phone: str) -> bool:
        """Check if a phone number belongs to a registered user."""
        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT 1 FROM users WHERE phone = ?", (phone,)
            )
            row = await cursor.fetchone()
            return row is not None

    async def get_user(self, phone: str) -> User | None:
        """Get a user by phone number, or None if not registered."""
        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT phone, name, role, registered_at, last_seen "
                "FROM users WHERE phone = ?",
                (phone,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return User(
                phone=row["phone"],
                name=row["name"],
                role=row["role"],
                registered_at=row["registered_at"],
                last_seen=row["last_seen"],
            )

    async def list_users(self) -> list[User]:
        """Return all registered users."""
        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT phone, name, role, registered_at, last_seen "
                "FROM users ORDER BY registered_at"
            )
            rows = await cursor.fetchall()
            return [
                User(
                    phone=row["phone"],
                    name=row["name"],
                    role=row["role"],
                    registered_at=row["registered_at"],
                    last_seen=row["last_seen"],
                )
                for row in rows
            ]

    async def update_last_seen(self, phone: str) -> None:
        """Update the last_seen timestamp for a user."""
        now = datetime.now().isoformat()
        async with self._get_db() as db:
            await db.execute(
                "UPDATE users SET last_seen = ? WHERE phone = ?", (now, phone)
            )
            await db.commit()

    async def remove_user(self, phone: str) -> bool:
        """Remove a user from the whitelist. Returns True if removed."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "DELETE FROM users WHERE phone = ?", (phone,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def update_role(self, phone: str, role: str) -> bool:
        """Update a user's role. Returns True if updated."""
        if role not in ("admin", "user"):
            return False
        async with self._get_db() as db:
            cursor = await db.execute(
                "UPDATE users SET role = ? WHERE phone = ?", (role, phone)
            )
            await db.commit()
            return cursor.rowcount > 0

    # -----------------------------------------------------------------
    # Registration codes
    # -----------------------------------------------------------------

    async def generate_registration_code(
        self,
        created_by: str,
        expires_hours: float | None = None,
    ) -> str:
        """Generate a single-use registration code.

        Args:
            created_by: Phone number of the admin generating the code.
            expires_hours: Override expiry in hours. Defaults to code_expiry
                          config (10 minutes = 1/6 hour).

        Returns:
            The generated 6-character alphanumeric code.

        Raises:
            PermissionError: If the caller is not an admin.
            RuntimeError: If the admin has exceeded the hourly code limit.
        """
        # Verify creator is admin
        user = await self.get_user(created_by)
        if user is None or user.role != "admin":
            raise PermissionError("Only admins can generate registration codes")

        # Rate limit: max codes per hour per admin
        async with self._get_db() as db:
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM registration_codes "
                "WHERE created_by = ? AND created_at > ?",
                (created_by, one_hour_ago),
            )
            row = await cursor.fetchone()
            if row and row[0] >= self._max_codes_per_hour:
                raise RuntimeError(
                    f"Rate limit: max {self._max_codes_per_hour} codes per hour"
                )

            # Generate code
            alphabet = string.digits
            code = "".join(secrets.choice(alphabet) for _ in range(self._code_length))

            now = datetime.now()
            if expires_hours is not None:
                expires_at = now + timedelta(hours=expires_hours)
            else:
                expires_at = now + timedelta(seconds=self._code_expiry)

            await db.execute(
                "INSERT INTO registration_codes "
                "(code, created_by, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (code, created_by, now.isoformat(), expires_at.isoformat()),
            )
            await db.commit()

        logger.info("Registration code generated by %s", created_by)
        return code

    async def generate_bootstrap_code(self) -> str:
        """Generate a bootstrap registration code for first admin setup.

        This does not require an existing admin. Can only be used when no
        admins exist yet.

        Returns:
            The generated 6-character code.

        Raises:
            RuntimeError: If an admin already exists.
        """
        users = await self.list_users()
        if any(u.role == "admin" for u in users):
            raise RuntimeError(
                "Bootstrap disabled: an admin already exists"
            )

        alphabet = string.digits
        code = "".join(secrets.choice(alphabet) for _ in range(self._code_length))

        now = datetime.now()
        expires_at = now + timedelta(seconds=self._code_expiry)

        async with self._get_db() as db:
            await db.execute(
                "INSERT INTO registration_codes "
                "(code, created_by, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (code, "bootstrap", now.isoformat(), expires_at.isoformat()),
            )
            await db.commit()

        logger.info("Bootstrap registration code generated")
        return code

    async def validate_code(self, code: str) -> bool:
        """Check if a registration code is valid (exists, not expired, not used)."""
        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT code, expires_at, used_by FROM registration_codes "
                "WHERE code = ?",
                (code,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            if row["used_by"] is not None:
                return False
            expires_at = datetime.fromisoformat(row["expires_at"])
            return datetime.now() < expires_at

    async def register_user(
        self,
        phone: str,
        name: str,
        code: str,
    ) -> RegisterResult:
        """Register a new user with a valid registration code.

        Args:
            phone: Phone number in E.164 format.
            name: Display name.
            code: The registration code to consume.

        Returns:
            RegisterResult with success status.
        """
        # Check if already registered
        if await self.is_authorized(phone):
            return RegisterResult(success=False, error="Phone already registered")

        # Validate code
        if not await self.validate_code(code):
            return RegisterResult(success=False, error="Invalid or expired code")

        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row

            # Determine role: admin if bootstrap code, else user
            cursor = await db.execute(
                "SELECT created_by FROM registration_codes WHERE code = ?",
                (code,),
            )
            code_row = await cursor.fetchone()
            is_bootstrap = code_row and code_row["created_by"] == "bootstrap"
            role = "admin" if is_bootstrap else "user"

            now = datetime.now().isoformat()

            # Mark code as used
            await db.execute(
                "UPDATE registration_codes SET used_by = ?, used_at = ? "
                "WHERE code = ?",
                (phone, now, code),
            )

            # Create user
            await db.execute(
                "INSERT INTO users (phone, name, role, registered_at) "
                "VALUES (?, ?, ?, ?)",
                (phone, name, role, now),
            )
            await db.commit()

        user = User(phone=phone, name=name, role=role, registered_at=now)
        logger.info(
            "User registered: %s (%s) as %s", name, phone, role
        )
        return RegisterResult(success=True, user=user)

    # -----------------------------------------------------------------
    # Brute-force protection
    # -----------------------------------------------------------------

    async def is_blocked(self, phone: str) -> bool:
        """Check if a phone number is currently blocked."""
        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT permanent, expires_at FROM blocks WHERE phone = ?",
                (phone,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            if row["permanent"]:
                return True
            if row["expires_at"]:
                return datetime.now() < datetime.fromisoformat(row["expires_at"])
            return False

    async def record_failed_attempt(self, phone: str) -> None:
        """Record a failed registration attempt and apply rate limiting.

        This handles the full rate-limiting lifecycle:
        1. Record the attempt
        2. Check if attempts exceed the window threshold
        3. Apply temp block if needed
        4. Escalate to permanent block after max temp blocks
        """
        now = datetime.now()

        async with self._get_db() as db:
            db.row_factory = aiosqlite.Row

            # Record the attempt
            await db.execute(
                "INSERT INTO failed_attempts (phone, attempted_at) VALUES (?, ?)",
                (phone, now.isoformat()),
            )

            # Count recent attempts in the window
            window_start = (
                now - timedelta(seconds=self._attempt_window)
            ).isoformat()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM failed_attempts "
                "WHERE phone = ? AND attempted_at > ?",
                (phone, window_start),
            )
            row = await cursor.fetchone()
            attempt_count = row[0] if row else 0

            if attempt_count >= self._max_attempts_per_window:
                # Check existing block record
                cursor = await db.execute(
                    "SELECT temp_block_count, permanent FROM blocks "
                    "WHERE phone = ?",
                    (phone,),
                )
                block_row = await cursor.fetchone()

                if block_row and block_row["permanent"]:
                    # Already permanently blocked
                    await db.commit()
                    return

                if block_row:
                    new_count = block_row["temp_block_count"] + 1
                    if new_count >= self._max_temp_blocks:
                        # Permanent block
                        await db.execute(
                            "UPDATE blocks SET permanent = 1, "
                            "blocked_at = ?, expires_at = NULL, "
                            "temp_block_count = ? WHERE phone = ?",
                            (now.isoformat(), new_count, phone),
                        )
                        logger.warning(
                            "Phone %s permanently blocked after %d temp blocks",
                            phone,
                            new_count,
                        )
                    else:
                        # Update temp block
                        expires_at = now + timedelta(
                            seconds=self._temp_block_duration
                        )
                        await db.execute(
                            "UPDATE blocks SET blocked_at = ?, "
                            "expires_at = ?, temp_block_count = ? "
                            "WHERE phone = ?",
                            (
                                now.isoformat(),
                                expires_at.isoformat(),
                                new_count,
                                phone,
                            ),
                        )
                        logger.warning(
                            "Phone %s temp blocked (%d/%d)",
                            phone,
                            new_count,
                            self._max_temp_blocks,
                        )
                else:
                    # First temp block
                    expires_at = now + timedelta(
                        seconds=self._temp_block_duration
                    )
                    await db.execute(
                        "INSERT INTO blocks "
                        "(phone, blocked_at, permanent, temp_block_count, expires_at) "
                        "VALUES (?, ?, 0, 1, ?)",
                        (phone, now.isoformat(), expires_at.isoformat()),
                    )
                    logger.warning(
                        "Phone %s temp blocked (1/%d)",
                        phone,
                        self._max_temp_blocks,
                    )

            await db.commit()

    async def unblock(self, phone: str) -> bool:
        """Remove a block on a phone number (admin action). Returns True if unblocked."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "DELETE FROM blocks WHERE phone = ?", (phone,)
            )
            # Also clear failed attempts
            await db.execute(
                "DELETE FROM failed_attempts WHERE phone = ?", (phone,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def run_maintenance(self) -> dict[str, int]:
        """Purge stale auth records: expired codes, old attempts, expired blocks.

        Returns a dict with the number of rows deleted per table.
        """
        now = datetime.now()
        expired_code_cutoff = (now - timedelta(hours=24)).isoformat()
        old_attempt_cutoff = (now - timedelta(days=30)).isoformat()

        deleted: dict[str, int] = {}

        async with self._get_db() as db:
            # Expired or used registration codes older than 24h
            cursor = await db.execute(
                "DELETE FROM registration_codes "
                "WHERE expires_at < ? OR (used_at IS NOT NULL AND used_at < ?)",
                (expired_code_cutoff, expired_code_cutoff),
            )
            deleted["registration_codes"] = cursor.rowcount

            # Failed attempts older than 30 days
            cursor = await db.execute(
                "DELETE FROM failed_attempts WHERE attempted_at < ?",
                (old_attempt_cutoff,),
            )
            deleted["failed_attempts"] = cursor.rowcount

            # Expired temporary blocks (non-permanent, past expiry)
            cursor = await db.execute(
                "DELETE FROM blocks "
                "WHERE permanent = 0 AND expires_at IS NOT NULL AND expires_at < ?",
                (now.isoformat(),),
            )
            deleted["blocks"] = cursor.rowcount

            await db.commit()

        total = sum(deleted.values())
        if total:
            logger.info("Auth maintenance: purged %s", deleted)
        return deleted

    async def has_admins(self) -> bool:
        """Check if any admin users exist."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1"
            )
            row = await cursor.fetchone()
            return row is not None
