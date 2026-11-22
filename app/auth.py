import hmac
from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class Actor:
    actor_id: str
    roles: tuple[str, ...]
    scopes: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


def resolve_token(token: str, settings: Settings) -> Actor | None:
    """Demo auth: bearer token -> server-side identity. Replace with SSO/OIDC in production."""
    for known, user in settings.demo_users.items():
        if hmac.compare_digest(token.encode(), known.encode()):
            return Actor(user.actor_id, tuple(user.roles), tuple(user.scopes))
    return None
