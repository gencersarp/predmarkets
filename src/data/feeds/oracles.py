"""
Oracle & Fundamental Data Ingestion.

Sources:
  - NewsAPI: headline sentiment → probability impact
  - Twitter/X firehose: real-time event detection
  - FiveThirtyEight / RealClearPolitics RSS: polling averages
  - FRED API: macroeconomic data
  - Sports APIs: game scores / probabilities
  - On-chain whale tracking: Polymarket smart money
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import feedparser

from config.settings import get_settings
from src.core.models import NewsEvent, OracleEstimate

logger = logging.getLogger(__name__)

# --- Pre-compiled sentiment wordlists (simple; swap for VADER/FinBERT in prod)
_POSITIVE_WORDS = frozenset([
    "win", "wins", "passes", "approved", "confirmed", "rises", "gains",
    "surge", "bullish", "strong", "record", "beats", "above", "positive",
])
_NEGATIVE_WORDS = frozenset([
    "loses", "fails", "rejected", "denied", "crash", "drops", "falls",
    "miss", "below", "negative", "weak", "tumbles", "collapse",
])

# FiveThirtyEight-style RSS feeds (use publicly available RSS where possible)
_RSS_FEEDS: dict[str, str] = {
    "538_politics": "https://fivethirtyeight.com/politics/feed/",
    "realclearpolitics": "https://www.realclearpolitics.com/rss/rss_main.xml",
    "predictit_news": "https://www.predictit.org/feed",
}

# Polymarket whale wallet addresses to monitor (public on-chain)
_KNOWN_WHALE_WALLETS: list[str] = [
    # Known high-volume Polymarket wallets (from on-chain analysis)
    # Add your own curated list here
]


class OracleFeed:
    """
    Aggregates external fundamental data to produce OracleEstimate objects.
    Each estimate provides a "true probability" for a market outcome
    against which the market's implied probability is compared.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, OracleEstimate] = {}
        self._news_cache: list[NewsEvent] = []
        self._newsapi_warned: bool = False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        logger.info("OracleFeed started")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ---------------------------------------------------------------- News

    async def fetch_news(
        self, query: str, max_articles: int = 20
    ) -> list[NewsEvent]:
        """Fetch recent news articles matching `query` via NewsAPI."""
        api_key = self._settings.newsapi_key
        if not api_key:
            if not self._newsapi_warned:
                logger.warning("NEWSAPI_KEY not set — skipping news fetch (this warning will not repeat)")
                self._newsapi_warned = True
            return []

        assert self._session is not None
        params = {
            "q": query,
            "apiKey": api_key.get_secret_value(),
            "pageSize": max_articles,
            "sortBy": "publishedAt",
            "language": "en",
        }
        try:
            async with self._session.get(
                "https://newsapi.org/v2/everything", params=params
            ) as resp:
                if resp.status != 200:
                    logger.warning("NewsAPI returned %d", resp.status)
                    return []
                data = await resp.json()
        except Exception as exc:
            logger.warning("NewsAPI fetch error: %s", exc)
            return []

        events: list[NewsEvent] = []
        for article in data.get("articles", []):
            headline = article.get("title", "")
            sentiment = self._simple_sentiment(headline)
            event = NewsEvent(
                headline=headline,
                source=article.get("source", {}).get("name", ""),
                published_at=self._parse_datetime(article.get("publishedAt", "")),
                url=article.get("url", ""),
                sentiment_score=sentiment,
            )
            events.append(event)

        self._news_cache = events[:max_articles]
        return self._news_cache

    async def fetch_rss_estimates(self) -> list[OracleEstimate]:
        """
        Parse FiveThirtyEight / RCP RSS feeds to extract polling estimates.
        Returns OracleEstimate objects keyed by topic.
        """
        estimates: list[OracleEstimate] = []
        for source_name, url in _RSS_FEEDS.items():
            try:
                # feedparser is sync; run in executor
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, url)
                for entry in feed.entries[:10]:
                    title = entry.get("title", "")
                    prob = self._extract_probability_from_text(title)
                    if prob is not None:
                        estimates.append(
                            OracleEstimate(
                                source=source_name,
                                market_id="",  # will be matched by the caller
                                true_probability=prob,
                                model_name=f"rss_{source_name}",
                                raw_data={"title": title, "url": entry.get("link", "")},
                            )
                        )
            except Exception as exc:
                logger.warning("RSS feed %s error: %s", source_name, exc)

        return estimates

    async def fetch_fred_series(self, series_id: str) -> dict[str, Any]:
        """
        Fetch a FRED economic data series (e.g., unemployment rate, CPI).
        Useful for macro-conditional markets.
        """
        fred_key = self._settings.fred_api_key
        if not fred_key:
            return {}

        assert self._session is not None
        params = {
            "series_id": series_id,
            "api_key": fred_key.get_secret_value(),
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        try:
            async with self._session.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params=params,
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                observations = data.get("observations", [])
                if observations:
                    return {
                        "series_id": series_id,
                        "latest_value": float(observations[0].get("value", 0)),
                        "date": observations[0].get("date"),
                        "observations": observations[:5],
                    }
        except Exception as exc:
            logger.warning("FRED fetch error for %s: %s", series_id, exc)
        return {}

    async def fetch_polymarket_whale_activity(
        self,
        rpc_url: str = "",
        min_usdc: float = 5000.0,
        market_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch large recent trades from Polymarket's public data API.

        Uses the public /activity endpoint (no auth required) to surface:
        - Single trades > min_usdc (smart money indicator)
        - Multiple trades in the same direction within 60 minutes

        Returns a list of activity records enriched with:
            market_id, side, size_usd, price, wallet, timestamp, is_whale
        """
        assert self._session is not None
        results: list[dict[str, Any]] = []

        # Polymarket public API — no auth required
        base_url = "https://data-api.polymarket.com"

        try:
            # Fetch recent high-volume activity
            params: dict[str, Any] = {"limit": 200, "sortBy": "size", "sortDir": "DESC"}
            if market_ids:
                params["market"] = market_ids[0]  # API takes one market at a time

            async with self._session.get(
                f"{base_url}/activity",
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    logger.debug("Polymarket activity API returned %d", resp.status)
                    return []
                data = await resp.json()

            records = data if isinstance(data, list) else data.get("data", [])

            for r in records:
                size = float(r.get("usdcSize", r.get("amount", 0)))
                if size < min_usdc:
                    continue

                price = float(r.get("price", 0.5))
                outcome = str(r.get("outcome", "")).lower()
                side = "YES" if "yes" in outcome or "buy" in outcome else "NO"
                wallet = str(r.get("proxyWallet", r.get("user", "")))
                ts_raw = r.get("timestamp", r.get("createdAt", 0))
                try:
                    ts = datetime.utcfromtimestamp(float(ts_raw))
                except (ValueError, TypeError, OSError):
                    ts = datetime.now(timezone.utc)

                results.append({
                    "market_id": str(r.get("conditionId", r.get("market", ""))),
                    "side": side,
                    "size_usd": size,
                    "price": price,
                    "wallet": wallet,
                    "timestamp": ts.isoformat(),
                    "is_whale": size >= min_usdc * 2,
                    "title": str(r.get("title", "")),
                })

        except Exception as exc:
            logger.warning("Polymarket whale activity fetch failed: %s", exc)

        if results:
            logger.info(
                "Whale activity: %d large trades (>$%.0f) detected",
                len(results), min_usdc,
            )
        return results

    async def fetch_market_sentiment(
        self,
        market_title: str,
        recent_trades: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[OracleEstimate]:
        """
        Build an oracle estimate from:
          1. News sentiment on the market topic
          2. Whale/large-trade flow bias (if recent_trades provided)

        Returns an OracleEstimate or None if insufficient signal.
        """
        # News-based component
        news = await self.fetch_news(market_title[:60], max_articles=10)
        if not news:
            return None

        avg_sentiment = sum(n.sentiment_score for n in news) / len(news)
        # Map sentiment [-1, 1] to probability shift around 0.5
        # A sentiment of +0.5 → true_prob around 0.55
        sentiment_prob = 0.50 + avg_sentiment * 0.10
        sentiment_prob = max(0.05, min(0.95, sentiment_prob))

        # Trade-flow component
        if recent_trades:
            buy_vol = sum(
                t["size_usd"] for t in recent_trades if t.get("side") == "YES"
            )
            sell_vol = sum(
                t["size_usd"] for t in recent_trades if t.get("side") == "NO"
            )
            total = buy_vol + sell_vol
            if total > 0:
                ofi = (buy_vol - sell_vol) / total  # [-1, 1]
                # Blend: 60% news, 40% flow
                flow_prob = 0.50 + ofi * 0.15
                flow_prob = max(0.05, min(0.95, flow_prob))
                blended = 0.60 * sentiment_prob + 0.40 * flow_prob
            else:
                blended = sentiment_prob
        else:
            blended = sentiment_prob

        # Confidence is low for news-only signals; higher with flow data
        ci_width = 0.30 if not recent_trades else 0.20

        return OracleEstimate(
            source="news_sentiment",
            market_id="",
            true_probability=blended,
            confidence_interval_low=max(0.01, blended - ci_width / 2),
            confidence_interval_high=min(0.99, blended + ci_width / 2),
            model_name="sentiment_flow_blend",
            raw_data={
                "avg_sentiment": avg_sentiment,
                "n_articles": len(news),
                "market_title": market_title[:60],
            },
        )

    # ---------------------------------------------------------------- Probability Builders

    def build_election_oracle(
        self,
        candidate: str,
        poll_avg: float,
        fundamentals_weight: float = 0.3,
        polls_weight: float = 0.7,
    ) -> OracleEstimate:
        """
        Build an oracle estimate for an election market.
        Combines polling averages with fundamentals (approval rating, incumbency).

        poll_avg: current polling average (0-1)
        fundamentals_weight: weight given to non-polling factors
        """
        # Simple Bayesian update: prior from fundamentals (50% base), update with polls
        prior = 0.50
        posterior = polls_weight * poll_avg + fundamentals_weight * prior
        # Convert polling to probability via normal CDF (Abramowitz & Stegun)
        import math
        margin = poll_avg - 0.5
        # Sigma ~ 3% for presidential, ~5% for congressional
        sigma = 0.035
        prob = 0.5 + 0.5 * math.erf(margin / (sigma * math.sqrt(2)))
        prob = max(0.02, min(0.98, prob))

        return OracleEstimate(
            source="internal_election_model",
            market_id="",
            true_probability=prob,
            confidence_interval_low=max(0, prob - 0.08),
            confidence_interval_high=min(1, prob + 0.08),
            model_name="polls_plus_fundamentals",
            raw_data={
                "candidate": candidate,
                "poll_avg": poll_avg,
                "posterior": posterior,
            },
        )

    # ---------------------------------------------------------------- Helpers

    def _simple_sentiment(self, text: str) -> float:
        """
        Naive sentiment score between -1 and 1.
        In production, replace with VADER, FinBERT, or GPT-4o inference.
        """
        words = re.findall(r"\w+", text.lower())
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _extract_probability_from_text(self, text: str) -> Optional[float]:
        """
        Try to extract a probability from polling text.
        e.g., "Biden leads with 52%" → 0.52
        """
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
        if matches:
            # Take the first match as a polling number
            try:
                pct = float(matches[0])
                if 1 <= pct <= 99:
                    return pct / 100.0
            except ValueError:
                pass
        return None

    def _parse_datetime(self, s: str) -> datetime:
        if not s:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
