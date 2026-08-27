import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from openai import OpenAI


class ModelProviderConfigurationError(ValueError):
    pass


def validate_model_base_url(
    base_url: str,
    *,
    allow_private: bool,
    resolve_dns: bool = False,
    trusted_hosts: str | tuple[str, ...] | list[str] | set[str] = (),
) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in ({"http", "https"} if allow_private else {"https"}):
        raise ModelProviderConfigurationError("Model Base URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ModelProviderConfigurationError("Model Base URL has an invalid authority or suffix")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not allow_private and not address.is_global:
        raise ModelProviderConfigurationError("Private or non-global model addresses are disabled")
    hostname = parsed.hostname.lower().rstrip(".")
    if (hostname == "localhost" or hostname.endswith(".localhost")) and not allow_private:
        raise ModelProviderConfigurationError("Private or non-global model addresses are disabled")
    trusted_hostnames = _normalize_trusted_hosts(trusted_hosts)
    hostname_is_trusted = address is None and hostname in trusted_hostnames
    if resolve_dns and not allow_private:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as exc:
            raise ModelProviderConfigurationError("Model Base URL hostname cannot be resolved") from exc
        if (
            not addresses
            or any(not ipaddress.ip_address(item).is_global for item in addresses)
        ) and not hostname_is_trusted:
            raise ModelProviderConfigurationError("Model Base URL resolves to a non-global address")
    return value


def _normalize_trusted_hosts(
    trusted_hosts: str | tuple[str, ...] | list[str] | set[str],
) -> frozenset[str]:
    values = trusted_hosts.split(",") if isinstance(trusted_hosts, str) else trusted_hosts
    normalized: set[str] = set()
    for value in values:
        hostname = str(value).strip().lower().rstrip(".")
        if not hostname:
            continue
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            continue
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or any(character in hostname for character in "/:@?#")
        ):
            continue
        normalized.add(hostname)
    return frozenset(normalized)


@dataclass(frozen=True)
class OpenAIProviderAdapter:
    provider_key: str

    def create_client(self, config: dict, api_key: str) -> OpenAI:
        validate_model_base_url(
            config["base_url"],
            allow_private=bool(config.get("_allow_private", False)),
            resolve_dns=True,
            trusted_hosts=config.get("_trusted_hosts", ()),
        )
        return OpenAI(
            api_key=api_key,
            base_url=config["base_url"],
            timeout=float(config.get("timeout_seconds", 120)),
            max_retries=0,
        )

    def test_connection(self, config: dict, api_key: str) -> dict:
        client = self.create_client(config, api_key)
        response = client.models.list()
        models = [item.id for item in list(response.data)[:20]]
        return {"ok": True, "models": models}


class ProviderAdapterRegistry:
    def __init__(self):
        adapter = OpenAIProviderAdapter("openai_compatible")
        self._adapters = {
            "openai": adapter,
            "openai_compatible": adapter,
        }

    def get(self, provider: str) -> OpenAIProviderAdapter:
        key = provider.strip().lower().replace("-", "_")
        adapter = self._adapters.get(key)
        if adapter is None:
            raise ModelProviderConfigurationError(f"Unsupported model provider: {provider}")
        return adapter
