"""
AWS Secrets Manager integration.

Retrieves and injects API credentials at startup, keeping private keys
out of environment variables for production deployments.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


def inject_aws_secrets(settings: "Settings") -> None:
    """
    Pull secrets from AWS Secrets Manager and inject them into `settings`.
    Called once at startup when USE_AWS_SECRETS=true.

    Secrets are stored as JSON objects, e.g.:
        pmt/prod/polymarket  →  {"private_key": "...", "api_key": "...", ...}
        pmt/prod/kalshi      →  {"api_key": "...", "api_secret": "..."}
    """
    try:
        import boto3  # lazy import — optional dependency
    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        return

    client = boto3.client("secretsmanager", region_name=settings.aws_region)

    # Polymarket secrets
    _inject_secret(
        client=client,
        secret_name=settings.aws_secret_name_polymarket,
        settings=settings,
        field_map={
            "private_key": "polymarket_private_key",
            "api_key": "polymarket_api_key",
            "api_secret": "polymarket_api_secret",
            "api_passphrase": "polymarket_api_passphrase",
        },
    )

    # Kalshi secrets
    _inject_secret(
        client=client,
        secret_name=settings.aws_secret_name_kalshi,
        settings=settings,
        field_map={
            "api_key": "kalshi_api_key",
            "api_secret": "kalshi_api_secret",
        },
    )

    logger.info("AWS secrets injected successfully")


def _inject_secret(
    client: object,
    secret_name: str,
    settings: "Settings",
    field_map: dict[str, str],
) -> None:
    """Retrieve one secret and inject its values into settings fields."""
    try:
        response = client.get_secret_value(SecretId=secret_name)  # type: ignore[union-attr]
        raw = response.get("SecretString", "{}")
        data = json.loads(raw)

        for secret_key, settings_attr in field_map.items():
            if secret_key in data and data[secret_key]:
                from pydantic import SecretStr
                object.__setattr__(settings, settings_attr, SecretStr(data[secret_key]))
                logger.debug("Injected %s → %s", secret_key, settings_attr)

    except Exception as exc:
        logger.warning("Could not retrieve secret '%s': %s", secret_name, exc)
