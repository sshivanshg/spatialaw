import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class AlertClientConfig:
    endpoint: str
    api_key: Optional[str] = None
    timeout_s: float = 5.0
    max_retries: int = 3
    backoff_s: float = 1.5


class AlertClient:
    """HTTP webhook client to send intrusion alerts to a central server."""

    def __init__(self, cfg: AlertClientConfig):
        self.cfg = cfg

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def send(self, payload: Dict[str, Any]) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = requests.post(
                    self.cfg.endpoint,
                    data=json.dumps(payload),
                    headers=self._headers(),
                    timeout=self.cfg.timeout_s,
                )
                if 200 <= resp.status_code < 300:
                    return resp
                # retry on 5xx
                if resp.status_code >= 500:
                    time.sleep(self.cfg.backoff_s * attempt)
                    continue
                return resp
            except Exception as e:  # network errors
                last_exc = e
                time.sleep(self.cfg.backoff_s * attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("Alert send failed without exception")

