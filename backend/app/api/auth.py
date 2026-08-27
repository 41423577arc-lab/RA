from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_session
from app.schemas.auth import CurrentUserResponse, LoginRequest, RegisterRequest
from app.services.auth import (
    AUTH_COOKIE_NAME,
    AuthService,
    Principal,
    optional_principal,
)


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.get("/me", response_model=CurrentUserResponse)
def current_user(
    principal: Principal | None = Depends(optional_principal),
) -> CurrentUserResponse:
    if principal is None:
        return CurrentUserResponse(
            auth_enabled=True,
            registration_enabled=settings.auth_allow_registration,
        )
    return _user_response(principal)


@router.post("/register", response_model=CurrentUserResponse)
def register(
    payload: RegisterRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> CurrentUserResponse:
    if not settings.auth_enabled:
        raise HTTPException(status_code=409, detail="Authentication is disabled")
    service = AuthService(session, settings)
    try:
        user = service.register(**payload.model_dump())
        token, _ = service.create_session(user)
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _set_session_cookie(response, token)
    return CurrentUserResponse(
        auth_enabled=True,
        registration_enabled=settings.auth_allow_registration,
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
    )


@router.post("/login", response_model=CurrentUserResponse)
def login(
    payload: LoginRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> CurrentUserResponse:
    if not settings.auth_enabled:
        raise HTTPException(status_code=409, detail="Authentication is disabled")
    service = AuthService(session, settings)
    try:
        user = service.authenticate(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    token, _ = service.create_session(user)
    _set_session_cookie(response, token)
    return CurrentUserResponse(
        auth_enabled=True,
        registration_enabled=settings.auth_allow_registration,
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
    )


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
) -> Response:
    AuthService(session, settings).revoke(request.cookies.get(AUTH_COOKIE_NAME))
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    response.status_code = 204
    return response


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        max_age=settings.auth_session_days * 24 * 60 * 60,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )


def _user_response(principal: Principal) -> CurrentUserResponse:
    return CurrentUserResponse(
        auth_enabled=principal.auth_enabled,
        registration_enabled=settings.auth_allow_registration,
        user_id=principal.user_id,
        email=principal.email,
        display_name=principal.display_name,
        role=principal.role,
    )
