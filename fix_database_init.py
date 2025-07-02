import os
import sys
import asyncio
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

os.environ["STANDALONE_MODE"] = "true"


async def fix_database_initialization():
    """Fix database initialization with proper async handling"""
    try:
        from app.db.session import get_engine
        from app.db.base import Base

        print("Creating database tables...")
        engine = get_engine()

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("✅ Database tables created successfully")

        from app.core.security import get_password_hash
        import sqlite3

        db_path = Path(__file__).parent / "cardioai.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if not cursor.fetchone():
            hashed_password = get_password_hash("admin")

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
            print("✅ Admin user created successfully")
        else:
            print("Admin user already exists")

        conn.close()

        await engine.dispose()
        return True

    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(fix_database_initialization())
    sys.exit(0 if success else 1)
