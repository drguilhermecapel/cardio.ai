import os
import sys
import asyncio
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

os.environ["STANDALONE_MODE"] = "true"


async def init_database_tables():
    """Initialize database tables and create admin user"""
    try:
        from app.db.session import engine
        from app.db.base import Base
        from app.db.init_db import init_db

        print("Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("Initializing database with default data...")
        await init_db()

        print("✅ Database initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(init_database_tables())
    sys.exit(0 if success else 1)
