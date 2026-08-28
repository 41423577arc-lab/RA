from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=10, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=200)


class CurrentUserResponse(BaseModel):
    auth_enabled: bool
    registration_enabled: bool
    agent_admin_enabled: bool
    user_id: str | None = None
    email: str | None = None
    display_name: str | None = None
    role: str | None = None
