#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM知识提取模块。"""

import json
import logging
from pathlib import Path
from typing import Dict

from .llm_client import LLMClient
from ..utils import parse_llm_json_response


MODULE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = MODULE_DIR


class LLMExtractor:
    def __init__(self, llm_client: LLMClient, logger: logging.Logger = None):
        self.llm = llm_client
        self.logger = logger or logging.getLogger(__name__)

    def load_template(self, system_type: str) -> Dict:
        template_name = 'mixed_system_knowledge_card_template.json' if system_type == 'mixed' else 'single_system_knowledge_card_template.json'
        template_path = TEMPLATE_DIR / template_name
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载模板失败 {template_path}: {e}")
            return {}

    def extract_knowledge_with_llm(self, markdown_content: str, paper_metadata: Dict, system_type: str, template: Dict, target_system: Dict = None, has_multiple: bool = False) -> Dict:
        template_str = json.dumps(template, ensure_ascii=False, indent=2)
        system_role = (target_system or {}).get('role', 'target')
        base_prompt = f"""
You are a professional scientific literature knowledge extraction expert. Please carefully read the following scientific paper in the field of perovskite solar cells and extract relevant information according to the provided knowledge card template.

System Type: {system_type}
System Role: {system_role}

Paper Content (Full Markdown Format):
{markdown_content}

Knowledge Card Template:
{template_str}

【IMPORTANT NOTES】
- Metadata fields (DOI, paper_title, journal, year) will be AUTO-FILLED by code
- Do NOT extract these metadata fields, focus ONLY on scientific knowledge
- Leave metadata fields as they are in the template (e.g., \"N/A\" or \"AUTO_FILLED\")
- **full_name MUST be the English full name**
"""
        if has_multiple and target_system:
            specific_instructions = f"""

【IMPORTANT】This paper studied multiple independent systems. Currently, only extract data for the following system:
System Index: {target_system.get('system_index')}
System Name: {target_system.get('system_name')}
System Description: {target_system.get('brief_description', '')}

Only extract this specific system and only perovskite solar cell data.
"""
        else:
            specific_instructions = """

【IMPORTANT】This paper studied a single system. Extract all relevant perovskite solar cell data for this system.
"""
        general_instructions = """

Please extract information according to the template structure exactly.
- Use N/A for unavailable fields.
- Device performance must reflect this system only.
- Mixed systems must describe each component and role.
- Single systems must include full_name, abbreviation, and smiles when available.
- Output valid JSON only.
"""
        prompt = base_prompt + specific_instructions + general_instructions
        response = self.llm.call_api_with_retry(prompt)
        knowledge_card = parse_llm_json_response(response, default_value=template, logger=self.logger)
        return knowledge_card if knowledge_card and knowledge_card != template else template

