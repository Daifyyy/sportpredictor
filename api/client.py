import time
import logging
from typing import Any, Dict, Optional
from collections import deque
import requests
from api.cache import CacheManager
from config.settings import Settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket — chrání před přetečením API limitu."""
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: deque = deque()

    def wait(self):
        now = time.time()
        self._calls = deque(t for t in self._calls if now - t < self.period)
        if len(self._calls) >= self.max_calls:
            sleep_time = self.period - (now - self._calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: čekám {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self._calls.append(time.time())


class APIClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = CacheManager(settings.cache_db)
        self.rate_limiter = RateLimiter(settings.requests_per_minute)
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": settings.api_key,
        })

    def get(
        self,
        endpoint: str,
        params: Dict[str, Any],
        ttl: int = 3600,
        force_refresh: bool = False,
    ) -> Optional[Dict]:
        cache_key = f"{endpoint}:{sorted(params.items())}"

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return cached

        self.rate_limiter.wait()
        url = f"{self.settings.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("errors"):
                logger.error(f"API error: {data['errors']}")
                return None

            self.cache.set(cache_key, data, ttl)
            logger.debug(f"API fetch: {endpoint} {params}")
            return data

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
