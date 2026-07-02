#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM客户端模块。"""

import logging
import time
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LLMClient:
    def __init__(self, llm_config: Dict, retry_attempts: int = 3, retry_delay: float = 2.0, timeout: int = 120):
        self.llm_config = llm_config
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        self.session = requests.Session()
        retry = Retry(
            total=retry_attempts,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._refresh_headers()

    def _refresh_headers(self):
        self.headers = {
            "Authorization": f"Bearer {self.llm_config.get('api_key', '')}",
            "Content-Type": "application/json",
        }

    def update_config(self, llm_config: Dict):
        self.llm_config = llm_config
        self._refresh_headers()

    def call_api(self, prompt: str) -> Optional[str]:
        url = self.llm_config['api_url']
        payload = {
            "model": self.llm_config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_config.get('temperature', 0),
            "max_tokens": self.llm_config.get('max_tokens', 8000),
        }
        try:
            resp = self.session.post(url, headers=self.headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.warning(f"API调用异常: {e}")
            return None

    def call_api_with_retry(self, prompt: str) -> Optional[str]:
        for attempt in range(self.retry_attempts):
            response = self.call_api(prompt)
            if response:
                return response
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        return None

