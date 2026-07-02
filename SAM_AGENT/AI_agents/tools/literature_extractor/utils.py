#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工具函数模块。"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional


MODULE_DIR = Path(__file__).resolve().parent
NEEDS_REVIEW_FILE = MODULE_DIR / "needs_review.txt"


def extract_doi_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    acs_url_match = re.search(r'https?://(?:pubs\.)?acs\.org/doi/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', text, re.IGNORECASE)
    if acs_url_match:
        return acs_url_match.group(1).rstrip(').,;]')

    doi_matches = re.findall(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', text, re.IGNORECASE)
    for doi in doi_matches:
        cleaned = doi.rstrip(').,;]')
        if len(cleaned) > 8:
            return cleaned
    return None


def parse_llm_json_response(response: str, default_value: Any = None, logger: logging.Logger = None) -> Optional[Dict]:
    if not response:
        return default_value

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        response_clean = response.strip()
        if response_clean.startswith('```json'):
            response_clean = response_clean[7:]
        if response_clean.endswith('```'):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_clean[json_start:json_end]
            return json.loads(json_str)
        logger.error("No valid JSON format found")
        return default_value
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.debug(f"Response content first 500 chars: {response[:500] if response else 'None'}")
        return default_value
    except Exception as e:
        logger.error(f"JSON parsing exception: {e}")
        return default_value


def record_needs_review(reference_id: str, reason: str, review_file: Path = NEEDS_REVIEW_FILE):
    if not reference_id:
        return

    try:
        os.makedirs(review_file.parent, exist_ok=True)
        with open(review_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {reference_id}: {reason}\n")
        logger = logging.getLogger(__name__)
        logger.warning(f"已记录需复查: {reference_id} - {reason}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"记录复查信息失败: {e}")


def extract_sam_name_from_system_name(system_name: str) -> str:
    name = system_name.lower().strip()
    substrate_patterns = [
        'on ito', 'on fto', 'on sno2', 'on sno₂', 'on tio2', 'on tio₂',
        'on nio', 'on niox', 'on nio_x', 'on nicox',
        '/ito', '/fto', '/sno2', '/sno₂', '/tio2', '/tio₂', '/nio', '/niox', '/nio_x',
        'ito/', 'fto/', 'sno2/', 'sno₂/', 'tio2/', 'tio₂/', 'nio/', 'niox/', 'nio_x/',
    ]
    for pattern in substrate_patterns:
        name = name.replace(pattern, '')

    role_patterns = ['(baseline)', '(control)', '(target)', 'baseline', 'control', 'target']
    for pattern in role_patterns:
        name = name.replace(pattern, '')

    name = name.replace('/', '').replace('_', '').strip()
    return ' '.join(name.split())


def extract_pce_from_card(knowledge_card: Dict) -> Optional[float]:
    try:
        pce_str = knowledge_card.get('device_performance', {}).get('best_performance', {}).get('pce', 'N/A')
        if not pce_str or pce_str == 'N/A':
            return None
        return float(str(pce_str).replace('%', '').strip())
    except Exception:
        return None

