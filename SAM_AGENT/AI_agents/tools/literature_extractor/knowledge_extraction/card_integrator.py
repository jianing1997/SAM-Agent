#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识卡片整合模块。"""

import logging
import time
from typing import Dict, List

from ..utils import extract_pce_from_card, extract_sam_name_from_system_name


class CardIntegrator:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def integrate_single_system_card(self, knowledge_card: Dict, paper_metadata: Dict) -> Dict:
        if 'metadata' in knowledge_card:
            knowledge_card['metadata']['DOI'] = paper_metadata.get('DOI', 'N/A')
            knowledge_card['metadata']['document_id'] = paper_metadata.get('document_id', '')
            knowledge_card['metadata']['paper_title'] = paper_metadata.get('paper_title', '')
            knowledge_card['metadata']['journal'] = paper_metadata.get('journal', '')
            knowledge_card['metadata']['year'] = paper_metadata.get('year', '')

        if 'molecule_identity' in knowledge_card:
            for field in ['full_name', 'abbreviation', 'smiles']:
                knowledge_card['molecule_identity'].setdefault(field, '')

        device_perf = knowledge_card.setdefault('device_performance', {})
        best_perf = device_perf.setdefault('best_performance', {})
        for field in ['voc', 'jsc', 'ff', 'pce', 'scan_direction']:
            best_perf.setdefault(field, 'N/A')
        reference_perf = device_perf.setdefault('reference_performance', {})
        for field in ['device_type', 'voc', 'jsc', 'ff', 'pce']:
            reference_perf.setdefault(field, 'N/A')

        knowledge_card['extraction_metadata'] = {
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'extractor_version': '1.0',
            'system_type': 'single',
            'level': paper_metadata.get('level', ''),
            'justification': paper_metadata.get('justification', ''),
        }
        return knowledge_card

    def integrate_mixed_system_card(self, knowledge_card: Dict, paper_metadata: Dict) -> Dict:
        if 'metadata' in knowledge_card:
            knowledge_card['metadata']['DOI'] = paper_metadata.get('DOI', 'N/A')
            knowledge_card['metadata']['document_id'] = paper_metadata.get('document_id', '')
            knowledge_card['metadata']['paper_title'] = paper_metadata.get('paper_title', '')
            knowledge_card['metadata']['journal'] = paper_metadata.get('journal', '')
            knowledge_card['metadata']['year'] = paper_metadata.get('year', '')

        if 'system_identity' in knowledge_card:
            knowledge_card['system_identity'].setdefault('system_type', 'mixed')
            if 'components' in knowledge_card['system_identity']:
                components = knowledge_card['system_identity']['components']
                components.setdefault('main_sam', {"name": '', "abbreviation": '', "concentration": '', "function": ''})
                modifiers = components.setdefault('modifiers', [])
                if not isinstance(modifiers, list):
                    components['modifiers'] = [modifiers]

        device_perf = knowledge_card.setdefault('device_performance', {})
        best_perf = device_perf.setdefault('best_performance', {})
        for field in ['voc', 'jsc', 'ff', 'pce', 'scan_direction']:
            best_perf.setdefault(field, 'N/A')
        reference_perf = device_perf.setdefault('reference_performance', {})
        for field in ['device_type', 'voc', 'jsc', 'ff', 'pce']:
            reference_perf.setdefault(field, 'N/A')

        if 'design_insights' in knowledge_card:
            knowledge_card['design_insights'].setdefault('synergistic_mechanisms', '')

        knowledge_card['extraction_metadata'] = {
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'extractor_version': '1.0',
            'system_type': 'mixed',
            'level': paper_metadata.get('level', ''),
            'justification': paper_metadata.get('justification', ''),
        }
        return knowledge_card

    def integrate_knowledge_card(self, knowledge_card: Dict, paper_metadata: Dict, system_type: str, system_info: Dict = None, all_systems: List[Dict] = None) -> Dict:
        result = self.integrate_mixed_system_card(knowledge_card, paper_metadata) if system_type == 'mixed' else self.integrate_single_system_card(knowledge_card, paper_metadata)
        if not system_info:
            return result

        document_id = paper_metadata.get('document_id', paper_metadata.get('DOI', ''))
        system_index = system_info.get('system_index', 1)
        system_count = system_info.get('system_count', 1)
        system_name = system_info.get('system_name', '')
        system_role = system_info.get('role', 'target')
        has_multiple = system_count > 1
        system_id = f"{document_id}_system{system_index}" if has_multiple else document_id

        metadata = result.setdefault('metadata', {})
        metadata['system_id'] = system_id
        metadata['system_role'] = system_role
        metadata['system_index'] = system_index
        metadata['total_systems'] = system_count
        metadata['system_name'] = system_name

        baseline_reference = 'N/A'
        if all_systems and system_role == 'target':
            baseline_systems = [s for s in all_systems if s.get('role') == 'baseline']
            if baseline_systems:
                baseline_idx = baseline_systems[0].get('system_index', 1)
                baseline_reference = f"{document_id}_system{baseline_idx}" if has_multiple else document_id
        metadata['baseline_reference'] = baseline_reference
        metadata['related_systems'] = [f"{document_id}_system{i}" for i in range(1, system_count + 1) if i != system_index] if has_multiple else []

        extraction_metadata = result.setdefault('extraction_metadata', {})
        extraction_metadata['system_type'] = system_type
        extraction_metadata['has_multiple_systems'] = has_multiple
        extraction_metadata['target_application'] = 'perovskite_solar_cell'
        return result

    def deduplicate_knowledge_cards(self, knowledge_cards: List[Dict], document_id: str = '') -> List[Dict]:
        if len(knowledge_cards) <= 1:
            return knowledge_cards

        unique_cards = []
        for card1 in knowledge_cards:
            is_duplicate = False
            card1_sam_name = extract_sam_name_from_system_name(card1.get('metadata', {}).get('system_name', ''))
            card1_type = card1.get('extraction_metadata', {}).get('system_type', 'single')
            card1_role = card1.get('metadata', {}).get('system_role', 'target')
            card1_pce = extract_pce_from_card(card1)
            for card2 in unique_cards:
                card2_sam_name = extract_sam_name_from_system_name(card2.get('metadata', {}).get('system_name', ''))
                card2_type = card2.get('extraction_metadata', {}).get('system_type', 'single')
                card2_role = card2.get('metadata', {}).get('system_role', 'target')
                card2_pce = extract_pce_from_card(card2)
                if not (card1_sam_name == card2_sam_name and card1_sam_name):
                    continue
                if card1_type != card2_type or card1_role != card2_role:
                    continue
                if card1_pce is not None and card2_pce is not None and abs(card1_pce - card2_pce) < 0.2:
                    is_duplicate = True
                    break
                if card1_pce is None and card2_pce is None:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_cards.append(card1)
        return self._reindex_knowledge_cards(unique_cards, document_id) if len(unique_cards) != len(knowledge_cards) else unique_cards

    def _reindex_knowledge_cards(self, knowledge_cards: List[Dict], document_id: str) -> List[Dict]:
        system_count = len(knowledge_cards)
        has_multiple = system_count > 1
        for new_idx, card in enumerate(knowledge_cards, 1):
            metadata = card.setdefault('metadata', {})
            metadata['system_index'] = new_idx
            metadata['total_systems'] = system_count
            metadata['system_id'] = f"{document_id}_system{new_idx}" if has_multiple else document_id
            metadata['related_systems'] = [f"{document_id}_system{i}" for i in range(1, system_count + 1) if i != new_idx] if has_multiple else []
            if metadata.get('system_role') == 'target':
                baseline_cards = [c for c in knowledge_cards if c.get('metadata', {}).get('system_role') == 'baseline']
                if baseline_cards:
                    baseline_idx = knowledge_cards.index(baseline_cards[0]) + 1
                    metadata['baseline_reference'] = f"{document_id}_system{baseline_idx}" if has_multiple else document_id
                else:
                    metadata['baseline_reference'] = 'N/A'
        return knowledge_cards

