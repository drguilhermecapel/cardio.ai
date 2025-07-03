import os
import sys
import asyncio
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

os.environ["STANDALONE_MODE"] = "true"


async def get_admin_password():
    """Get the admin password from database or create with known password"""
    try:
        from app.core.config import settings
        from app.core.security import get_password_hash, verify_password
        import sqlite3

        print(f"Config password setting: {settings.FIRST_SUPERUSER_PASSWORD}")

        db_path = Path(__file__).parent / "cardioai.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                "SELECT hashed_password FROM users WHERE username = ?", ("admin",)
            )
            result = cursor.fetchone()

            if result:
                hashed_password = result[0]
                print("Admin user found in database")

                test_passwords = [
                    "admin",
                    "admin123",
                    "CHANGE_ME_SECURE_PASSWORD_REQUIRED",
                ]
                for pwd in test_passwords:
                    if verify_password(pwd, hashed_password):
                        print(f"✅ Admin password is: {pwd}")
                        conn.close()
                        return pwd

                print("❌ Could not verify admin password with common passwords")

                new_password = "admin123"
                new_hash = get_password_hash(new_password)
                cursor.execute(
                    "UPDATE users SET hashed_password = ? WHERE username = ?",
                    (new_hash, "admin"),
                )
                conn.commit()
                print(f"✅ Updated admin password to: {new_password}")
                conn.close()
                return new_password
            else:
                print("❌ Admin user not found in database")
                conn.close()
                return None
        else:
            print("❌ Database file does not exist")
            return None

    except Exception as e:
        print(f"❌ Error getting admin password: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    password = asyncio.run(get_admin_password())
    if password:
        print(f"Use password: {password}")
    else:
        print("Failed to get admin password")
