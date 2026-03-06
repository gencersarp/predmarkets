"""
Centralised configuration via pydantic-settings.
All values come from environment variables or .env file.
Secrets are injected at startup from AWS Secrets Manager when USE_AWS_SECRETS=true.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class OrderType(str, Enum):
    IOC = "IOC"
    FOK = "FOK"
    GTC = "GTC"


class KalshiEnv(str, Enum):
    PROD = "prod"
    DEMO = "demo"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ Runtime
    pmt_mode: RuntimeMode = RuntimeMode.PAPER

    # ------------------------------------------------------------------ Database
    database_url: str = "sqlite+aiosqlite:///./pmt_local.db"

    # ------------------------------------------------------------------ Polymarket
    polygon_rpc_url: str = "https://polygon-rpc.com"
    polymarket_private_key: Optional[SecretStr] = None
    polymarket_api_key: Optional[SecretStr] = None
    polymarket_api_secret: Optional[SecretStr] = None
    polymarket_api_passphrase: Optional[SecretStr] = None
    polymarket_clob_endpoint: str = "https://clob.polymarket.com"
    polymarket_gamma_endpoint: str = "https://gamma-api.polymarket.com"
    ctf_exchange_address: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    # ------------------------------------------------------------------ Kalshi
    # Auth: Kalshi v2 uses email + password for session token.
    # kalshi_api_key   = your Kalshi account email
    # kalshi_api_secret = your Kalshi account password
    # For API key auth (RSA): set kalshi_api_key to the API key ID and
    # kalshi_private_key to the RSA private key PEM string.
    kalshi_api_key: Optional[SecretStr] = None
    kalshi_api_secret: Optional[SecretStr] = None
    kalshi_private_key: Optional[SecretStr] = None   # RSA PEM for key-based auth
    kalshi_env: KalshiEnv = KalshiEnv.DEMO

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_env == KalshiEnv.DEMO:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://trading-api.kalshi.com/trade-api/v2"

    # ------------------------------------------------------------------ AWS
    aws_region: str = "us-east-1"
    aws_secret_name_polymarket: str = "pmt/prod/polymarket"
    aws_secret_name_kalshi: str = "pmt/prod/kalshi"
    use_aws_secrets: bool = False

    # ------------------------------------------------------------------ Oracles
    newsapi_key: Optional[SecretStr] = None
    twitter_bearer_token: Optional[SecretStr] = None
    sports_api_key: Optional[SecretStr] = None
    fred_api_key: Optional[SecretStr] = None

    # ------------------------------------------------------------------ Risk
    max_portfolio_exposure_usd: float = 1000.0
    max_single_position_usd: float = 150.0
    kelly_fraction: float = Field(0.25, ge=0.01, le=1.0)
    max_drawdown_pct: float = Field(0.20, ge=0.01, le=1.0)
    max_correlation_exposure: float = Field(0.40, ge=0.0, le=1.0)
    aroc_minimum_annual: float = Field(0.30, ge=0.0)
    fee_edge_max_consumption: float = Field(0.40, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ Execution
    default_order_type: OrderType = OrderType.IOC
    slippage_tolerance_pct: float = 0.03
    gas_price_gwei_max: int = 100

    # ------------------------------------------------------------------ Observability
    log_level: str = "INFO"
    prometheus_port: int = 9090
    sentry_dsn: Optional[str] = None

    # ------------------------------------------------------------------ Paper Trading
    paper_initial_balance_usd: float = 1000.0

    @field_validator("pmt_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, v: str) -> str:
        return v.lower()

    @property
    def is_live(self) -> bool:
        return self.pmt_mode == RuntimeMode.LIVE

    @property
    def is_paper(self) -> bool:
        return self.pmt_mode == RuntimeMode.PAPER


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings object. Cached after first call."""
    settings = Settings()

    if settings.use_aws_secrets:
        # Lazy import to avoid hard dependency in non-AWS environments
        from src.utils.secrets import inject_aws_secrets
        inject_aws_secrets(settings)

    return settings
