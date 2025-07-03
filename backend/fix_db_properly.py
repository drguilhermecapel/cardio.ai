import os
import sys
import asyncio
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

os.environ["STANDALONE_MODE"] = "true"


async def fix_database_properly():
    """Fix database initialization with correct imports"""
    try:
        from app.models.base import Base
        from app.db.session import get_engine
        from app.db.init_db import create_admin_user, get_session_factory

        print("Creating database tables...")
        engine = get_engine()

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("✅ Database tables created successfully")

        print("Creating admin user...")
        session_factory = get_session_factory()
        async with session_factory() as session:
            admin_user = await create_admin_user(session)
            if admin_user:
                print("✅ Admin user created/verified successfully")
            else:
                print("⚠️ Admin user creation had issues")

        await engine.dispose()
        return True

    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(fix_database_properly())
    sys.exit(0 if success else 1)
