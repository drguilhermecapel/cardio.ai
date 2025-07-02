import os
import sqlite3
import hashlib
from pathlib import Path


def create_admin_user():
    """Create admin user directly in SQLite database"""
    try:
        os.environ["STANDALONE_MODE"] = "true"

        db_path = Path(__file__).parent / "cardioai.db"

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                phone VARCHAR(20),
                role VARCHAR(20) DEFAULT 'PHYSICIAN',
                license_number VARCHAR(50),
                specialty VARCHAR(100),
                institution VARCHAR(200),
                experience_years INTEGER,
                is_active BOOLEAN DEFAULT 1,
                is_verified BOOLEAN DEFAULT 1,
                is_superuser BOOLEAN DEFAULT 0,
                last_login TIMESTAMP,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                digital_signature_key TEXT,
                signature_created_at TIMESTAMP,
                notification_preferences TEXT DEFAULT '{}',
                ui_preferences TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if cursor.fetchone():
            print("Admin user already exists")
            conn.close()
            return True

        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from app.core.security import get_password_hash

        password = "admin"
        hashed_password = get_password_hash(password)

        cursor.execute(
            """
            INSERT INTO users (
                username, email, hashed_password, first_name, last_name,
                is_active, is_verified, is_superuser, role
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "admin",
                "admin@cardioai.pro",
                hashed_password,
                "Administrator",
                "System",
                1,
                1,
                1,
                "ADMIN",
            ),
        )

        conn.commit()
        conn.close()

        print("✅ Admin user created successfully")
        return True

    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False


if __name__ == "__main__":
    success = create_admin_user()
    exit(0 if success else 1)
