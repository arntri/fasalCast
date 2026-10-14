"""
Minimal CEDA Agmarknet API client.

Auth: Bearer token read from env var CEDA_API_KEY (never hardcoded).
Throttling: minimum interval between requests + exponential backoff on 429/5xx.

Endpoints (per verified schema):
  GET  /agmarknet/commodities
  GET  /agmarknet/geographies        (commodity_id: query param, path fallback)
  POST /agmarknet/markets            {commodity_id, state_id, district_id, indicator}
  POST /agmarknet/prices             {commodity_id, state_id, district_id[], market_id[], from_date, to_date}
  POST /agmarknet/quantities         (same shape as prices)
"""

import os
import time
import random

import requests

BASE_URL = "https://api.ceda.ashoka.edu.in/v1"


class CedaError(RuntimeError):
    pass


class CedaClient:
    def __init__(self, base_url=BASE_URL, min_interval=1.0, max_retries=5, timeout=120,
                 api_key=None):
        # Explicit api_key lets a multi-key fleet run one client per key in its
        # own thread; env stays the default for single-key CLI use.
        key = api_key or os.environ.get("CEDA_API_KEY")
        if not key:
            raise CedaError(
                "CEDA_API_KEY not set. Run: export CEDA_API_KEY='your-key' "
                "(do not put the key in source or commit it)."
            )
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        # Env overrides so a flaky-gateway backfill can fail fast (fewer retries,
        # shorter per-request timeout) and mop up 504'd chunks on a later pass,
        # instead of burning ~31s of backoff / up to `timeout`s hanging per call.
        self.max_retries = int(os.environ.get("CEDA_MAX_RETRIES", max_retries))
        self.timeout = int(os.environ.get("CEDA_TIMEOUT", timeout))
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )

    # ---- transport with throttle + backoff ----

    def _request(self, method, path, **kwargs):
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            # throttle
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.max_retries:
                    raise CedaError(f"{method} {path} failed after retries: {e}")
                delay = (2 ** attempt) + random.random()
                print(f"    (network timeout on {path}, retry {attempt+1}/"
                      f"{self.max_retries} in {delay:.0f}s...)")
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt == self.max_retries:
                    raise CedaError(f"{method} {path} failed after retries: "
                                    f"{resp.status_code} {resp.text[:300]}")
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else (2 ** attempt) + random.random()
                time.sleep(delay)
                continue
            if resp.status_code == 401:
                raise CedaError("401 Unauthorized — check CEDA_API_KEY is valid.")
            if resp.status_code == 400:
                raise CedaError(f"400 Bad Request on {path}: {resp.text[:300]}")
            raise CedaError(f"{method} {path}: {resp.status_code} {resp.text[:300]}")

    # ---- response unwrapping ----

    @staticmethod
    def _unwrap(j):
        """Real API wraps responses as {"output": {"type", "message", "data"}}.
        Fall back to documented shapes just in case some endpoint differs."""
        if isinstance(j, dict):
            if "output" in j and isinstance(j["output"], dict):
                out = j["output"]
                if out.get("type") not in (None, "success"):
                    raise CedaError(f"API returned type={out.get('type')}: "
                                    f"{out.get('message')}")
                return out.get("data", [])
            for k in ("data", "commodities", "geographies"):
                if k in j:
                    return j[k]
        raise CedaError(f"Unrecognized response shape: {str(j)[:300]}")

    # ---- endpoints ----

    def commodities(self):
        return self._unwrap(self._request("GET", "/agmarknet/commodities"))

    def geographies(self, commodity_id):
        """Schema note: unconfirmed whether commodity_id is a query param or
        path segment. Try query param first, fall back to path on 400."""
        try:
            j = self._request("GET", "/agmarknet/geographies",
                              params={"commodity_id": commodity_id})
        except CedaError as e:
            if "400" not in str(e):
                raise
            j = self._request("GET", f"/agmarknet/geographies/{commodity_id}")
        return self._unwrap(j)

    def markets(self, commodity_id, state_id, district_id, indicator="price"):
        """Docs show scalar district_id, but the API may want an array (like
        /prices). Try scalar first; on failure, retry with array shape."""
        base = {"commodity_id": commodity_id, "state_id": state_id,
                "indicator": indicator}
        try:
            body = dict(base, district_id=district_id)
            return self._unwrap(self._request("POST", "/agmarknet/markets",
                                              json=body))
        except CedaError as e:
            # Only a 400 means "wrong request shape" — the one case the array
            # variant might fix. On 504/5xx/429 the scalar body was fine and the
            # server just glitched; retrying the (malformed) array form only 500s
            # and burns another retry-backoff cycle, so propagate instead.
            if "400" not in str(e):
                raise
            print(f"    (scalar district_id 400'd: {str(e)[:100]} — "
                  f"trying array variant...)")
            body = dict(base, district_id=[district_id])
            return self._unwrap(self._request("POST", "/agmarknet/markets",
                                              json=body))

    def prices(self, commodity_id, state_id, district_ids, market_ids, from_date, to_date):
        body = {"commodity_id": commodity_id, "state_id": state_id,
                "district_id": list(district_ids), "market_id": list(market_ids),
                "from_date": from_date, "to_date": to_date}
        return self._unwrap(self._request("POST", "/agmarknet/prices", json=body))

    def quantities(self, commodity_id, state_id, district_ids, market_ids, from_date, to_date):
        body = {"commodity_id": commodity_id, "state_id": state_id,
                "district_id": list(district_ids), "market_id": list(market_ids),
                "from_date": from_date, "to_date": to_date}
        return self._unwrap(self._request("POST", "/agmarknet/quantities", json=body))
