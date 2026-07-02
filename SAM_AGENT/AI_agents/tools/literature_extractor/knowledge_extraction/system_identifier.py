#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体系识别模块。"""

import json
import logging
from typing import Dict, List, Optional

from .llm_client import LLMClient
from ..utils import parse_llm_json_response, record_needs_review


class SystemIdentifier:
    def __init__(self, llm_client: LLMClient, logger: logging.Logger = None):
        self.llm = llm_client
        self.logger = logger or logging.getLogger(__name__)

    def identify_all_systems_with_types(self, markdown_content: str, reference_id: str = "") -> Dict:
        initial_result = self._identify_systems_stage1(markdown_content, reference_id)
        if not initial_result or not initial_result.get('systems'):
            return self._get_default_result(reference_id, '第一阶段识别失败')
        return self._verify_systems_stage2(initial_result, markdown_content, reference_id)

    def _identify_systems_stage1(self, markdown_content: str, reference_id: str) -> Optional[Dict]:
        content_excerpt = markdown_content[:50000] if len(markdown_content) > 50000 else markdown_content
        prompt = f"""
You are a scientific literature analyst. Identify ALL SAM-related systems experimentally studied in this perovskite solar cell paper.

Paper Content:
{content_excerpt}

【Core Task】
Extract ALL distinct SAM systems that have:
- Experimental fabrication in THIS paper (not literature review)
- Device performance data (J-V curves, PCE values, etc.)
- Appear in Methods/Results sections

【System Information】
For each system, provide:
- system_name: Descriptive name (include concentration if mentioned, e.g., \"2PACz (0.3 mg/mL)\")
- system_type: \"single\" (pure SAM) or \"mixed\" (SAM + other molecules/ions)
- role: \"baseline\" (control) or \"target\" (research subject)
- brief_description: One sentence description
- is_sam_related: true (SAM-based) or false (traditional HTL like PEDOT:PSS)

【Key Rules】
1. **Substrate ≠ Component**: SAM on ITO/FTO/SnO₂/NiOx = single SAM (substrate is NOT a mixing component)
2. **Only experimental systems**: Ignore Introduction mentions, only extract systems with actual data
3. **Separate systems**: If same SAM tested with different additives separately → separate systems
4. **Concentration variants**: If same SAM at different concentrations → include concentration in name
5. **Single-junction perovskite solar cells ONLY**:
   - ONLY extract systems from single-junction perovskite solar cells (PSC)
   - EXCLUDE systems from organic photovoltaics (OPV/OSC), tandem/multi-junction devices, DSSC, etc.
   - If the paper studies both PSC and other technologies, ONLY extract PSC systems

【Output Format】
{{
    \"total_systems\": N,
    \"systems\": [
        {{
            \"system_index\": 1,
            \"system_name\": \"SAM name with details\",
            \"system_type\": \"single\" or \"mixed\",
            \"role\": \"baseline\" or \"target\",
            \"brief_description\": \"Brief description\",
            \"is_sam_related\": true or false
        }}
    ]
}}

Return JSON only.
"""
        response = self.llm.call_api_with_retry(prompt)
        data = parse_llm_json_response(response, logger=self.logger)
        if data and 'systems' in data:
            self.logger.info(f"✓ 第一阶段识别: {len(data.get('systems', []))} 个体系")
            return data
        return None

    def _verify_systems_stage2(self, systems_result: Dict, markdown_content: str, reference_id: str) -> Dict:
        systems = systems_result.get('systems', [])
        if not systems:
            return systems_result

        systems_json = json.dumps(systems, ensure_ascii=False, indent=2)
        content_excerpt = markdown_content[:80000] if len(markdown_content) > 80000 else markdown_content
        prompt = f"""Review and correct identified SAM systems from this perovskite solar cell paper.

【Systems】
{systems_json}

【Paper】
{content_excerpt}

【Tasks】
1. **Add missing**: All experimental SAM systems from Methods/Results (single-junction PSC only)
2. **Remove**:
   - Duplicates (same SAM+substrate+conditions)
   - Non-experimental (Introduction only, no data)
   - Non-PSC: OPV/OSC/tandem/DSSC/QDSC
3. **Merge**: Same SAM at different concentrations → \"SAM (conc1, conc2, conc3)\"
4. **Fix system_type**:
   - \"SAM on NiOx/ITO/FTO/SnO₂/TiO₂\" → \"single\"
   - \"SAM + modifier\" → \"mixed\"
5. **Match paper**: Exact names/abbreviations
6. **Verify**: types, roles, PSC-only
7. **Only keep one optimal target**:
   - Baseline/control systems must always be kept
   - For the same SAM tested at multiple target concentrations/formulations, keep ONLY the optimal system
   - Mention sub-optimal concentrations inside `brief_description`

【Output】
{{
    \"total_systems\": N,
    \"systems\": [{{\"system_index\": 1, \"system_name\": \"...\", \"system_type\": \"single/mixed\", \"role\": \"baseline/target\", \"brief_description\": \"...\", \"is_sam_related\": true/false}}],
    \"_corrections\": [\"- correction 1\", \"- correction 2\"],
    \"_completeness_check\": \"completeness note\"
}}

**CRITICAL**: Only single-junction perovskite solar cells. Exclude OPV/OSC/tandem/DSSC."""
        response = self.llm.call_api_with_retry(prompt)
        verified_data = parse_llm_json_response(response, logger=self.logger)
        if verified_data and 'systems' in verified_data:
            return verified_data
        return systems_result

    def _get_default_result(self, reference_id: str, reason: str) -> Dict:
        default_result = {
            'total_systems': 1,
            'systems': [{
                'system_index': 1,
                'system_name': '主要体系',
                'system_type': 'single',
                'role': 'target',
                'brief_description': '',
                'is_sam_related': True,
            }],
            '_warning': '使用默认值，请人工检查',
            '_needs_review': True,
            '_reason': reason,
        }
        record_needs_review(reference_id, reason)
        return default_result

