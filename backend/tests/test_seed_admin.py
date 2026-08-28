from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.database import Base
from app.services.auth import verify_password
from seed.seed_admin import ensure_default_admin


def test_default_admin_seed_is_idempotent_and_does_not_reset_password() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as session:
        admin, created = ensure_default_admin(
            session,
            username="admin",
            password="123456",
            display_name="管理员",
        )
        assert created is True
        assert admin.email == "admin"
        assert admin.role == "ADMIN"
        assert admin.status == "ACTIVE"
        assert verify_password("123456", admin.password_hash)

        same_admin, created_again = ensure_default_admin(
            session,
            username="ADMIN",
            password="a-different-password",
            display_name="其他名称",
        )
        assert created_again is False
        assert same_admin.id == admin.id
        assert same_admin.display_name == "管理员"
        assert verify_password("123456", same_admin.password_hash)
        assert not verify_password("a-different-password", same_admin.password_hash)
