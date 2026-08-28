import os

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models.database import User
from app.services.auth import AuthService, SYSTEM_TENANT_ID, hash_password


DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "123456"
DEFAULT_ADMIN_DISPLAY_NAME = "管理员"


def ensure_default_admin(
    session: Session,
    *,
    username: str,
    password: str,
    display_name: str,
) -> tuple[User, bool]:
    normalized_username = username.strip().casefold()
    if not normalized_username or not password:
        raise ValueError("默认管理员账号和密码不能为空")

    # 先保证固定租户存在，新数据库和已有数据库都走同一条初始化路径。
    AuthService(session, settings).ensure_system_user()
    existing = session.scalar(
        select(User).where(
            User.tenant_id == SYSTEM_TENANT_ID,
            func.lower(User.email) == normalized_username,
        )
    )
    if existing is not None:
        # Seed 重复运行时不覆盖密码、角色或状态，避免重启后撤销人工维护结果。
        return existing, False

    admin = User(
        tenant_id=SYSTEM_TENANT_ID,
        email=normalized_username,
        display_name=display_name.strip() or DEFAULT_ADMIN_DISPLAY_NAME,
        password_hash=hash_password(password),
        role="ADMIN",
        status="ACTIVE",
    )
    session.add(admin)
    session.commit()
    session.refresh(admin)
    return admin, True


def main() -> None:
    username = os.getenv("DEFAULT_ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)
    password = os.getenv("DEFAULT_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
    display_name = os.getenv("DEFAULT_ADMIN_DISPLAY_NAME", DEFAULT_ADMIN_DISPLAY_NAME)
    with SessionLocal() as session:
        admin, created = ensure_default_admin(
            session,
            username=username,
            password=password,
            display_name=display_name,
        )
    action = "created" if created else "already exists"
    print(f"Default administrator {admin.email}: {action}")


if __name__ == "__main__":
    main()
