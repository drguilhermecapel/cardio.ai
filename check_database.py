import os
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

os.environ["STANDALONE_MODE"] = "true"

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_password_hash
from sqlalchemy.orm import Session


def check_and_create_admin():
    """Check if admin user exists and create if not"""
    try:
        db = next(get_db())

        users = db.query(User).all()
        print(f"Total users in database: {len(users)}")

        for user in users:
            print(
                f"User: {user.username}, Email: {user.email}, Active: {user.is_active}"
            )

        admin_user = db.query(User).filter(User.username == "admin").first()

        if not admin_user:
            print("Creating default admin user...")

            hashed_password = get_password_hash("admin")
            admin_user = User(
                username="admin",
                email="admin@cardioai.pro",
                hashed_password=hashed_password,
                is_active=True,
                is_superuser=True,
                full_name="Administrator",
            )

            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)

            print(f"Admin user created successfully: {admin_user.username}")
        else:
            print(f"Admin user already exists: {admin_user.username}")

        db.close()
        return True

    except Exception as e:
        print(f"Error checking/creating admin user: {e}")
        return False


if __name__ == "__main__":
    success = check_and_create_admin()
    sys.exit(0 if success else 1)
