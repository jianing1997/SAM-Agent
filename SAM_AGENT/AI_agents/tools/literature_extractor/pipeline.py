#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文献提取主流程。"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .image_extraction import MoleculeImageExtractor
from .knowledge_extraction import CardIntegrator, LLMClient, LLMExtractor, MinerUInterface, SystemIdentifier
from .utils import extract_doi_from_text, parse_llm_json_response


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[3]
DEFAULT_CONFIG_PATH = MODULE_DIR / 'config.json'


class LiteratureExtractor:
    def __init__(self, open_ai_key=None, deepseek_key=None, llm_model: str = 'deepseek', config_path: Optional[str] = None, verbose: bool = True):
        self.open_ai_key = open_ai_key
        self.deepseek_key = deepseek_key
        self.llm_model = llm_model
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config(self.config_path)
        self._normalize_paths()

        self.mineru = MinerUInterface(self.config.get('mineru', {}))
        self.llm = LLMClient(
            self.config['llm'],
            retry_attempts=self.config.get('retry_attempts', 3),
            retry_delay=self.config.get('retry_delay', 2.0),
        )
        self.system_identifier = SystemIdentifier(self.llm, self.logger)
        self.llm_extractor = LLMExtractor(self.llm, self.logger)
        self.card_integrator = CardIntegrator(self.logger)
        self.molecule_image_extractor = MoleculeImageExtractor(
            self.config.get('molecule_image_extractor', {}),
            self.logger,
        )
        self._override_runtime_config()

    def _load_config(self, config_path: Path) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _normalize_paths(self):
        mineru_cfg = self.config.setdefault('mineru', {})
        local_md_dir = Path(mineru_cfg.get('local_md_dir', 'temp_markdown'))
        if not local_md_dir.is_absolute():
            local_md_dir = MODULE_DIR / local_md_dir
        mineru_output_dir = Path(mineru_cfg.get('mineru_output_dir', 'MinerU/output'))
        if not mineru_output_dir.is_absolute():
            mineru_output_dir = PROJECT_ROOT / mineru_output_dir
        mineru_cfg['local_md_dir'] = str(local_md_dir)
        mineru_cfg['mineru_output_dir'] = str(mineru_output_dir)
        output_dir = Path(self.config.get('output_dir', 'knowledge_cards'))
        if not output_dir.is_absolute():
            output_dir = MODULE_DIR / output_dir
        self.config['output_dir'] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        molecule_cfg = self.config.setdefault('molecule_image_extractor', {})
        molecule_output_dir = Path(molecule_cfg.get('output_dir', 'molecule_image_results'))
        if not molecule_output_dir.is_absolute():
            molecule_output_dir = MODULE_DIR / molecule_output_dir
        molecule_cfg['output_dir'] = str(molecule_output_dir)
        molecule_output_dir.mkdir(parents=True, exist_ok=True)

    def _override_runtime_config(self):
        llm_cfg = self.config.setdefault('llm', {})
        env_openai_key = os.getenv('OPENAI_API_KEY', '')
        env_openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.chatanywhere.tech/v1')
        env_openai_model = os.getenv('OPENAI_MODEL', 'gpt-4.1')
        env_deepseek_key = os.getenv('DEEPSEEK_API_KEY', '')
        env_deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        env_deepseek_model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

        if self.llm_model == 'chatgpt':
            api_key = self.open_ai_key or env_openai_key
            if not api_key:
                raise ValueError('LiteratureExtractor requires OPENAI_API_KEY or open_ai_key when llm_model is chatgpt.')
            base_url = (os.getenv('OPENAI_BASE_URL') or env_openai_base_url).rstrip('/')
            llm_cfg.update({
                'api_key': api_key,
                'model': os.getenv('OPENAI_MODEL') or env_openai_model,
                'api_url': f'{base_url}/chat/completions',
            })
        else:
            api_key = self.deepseek_key or env_deepseek_key
            if not api_key:
                raise ValueError('LiteratureExtractor requires DEEPSEEK_API_KEY or deepseek_key when llm_model is deepseek.')
            base_url = (os.getenv('DEEPSEEK_BASE_URL') or env_deepseek_base_url).rstrip('/')
            llm_cfg.update({
                'api_key': api_key,
                'model': os.getenv('DEEPSEEK_MODEL') or env_deepseek_model,
                'api_url': f'{base_url}/chat/completions',
            })
        self.llm.update_config(llm_cfg)
        self.mineru.update_paths(
            self.config['mineru']['local_md_dir'],
            self.config['mineru']['mineru_output_dir'],
        )

    def _extract_metadata_with_llm(self, markdown_content: str) -> Dict:
        excerpt = markdown_content[:20000]
        prompt = f"""
You are a metadata extraction assistant. Read the following scientific paper content and extract its bibliographic metadata.

Paper Content (excerpt):
{excerpt}

Return a JSON object with the following fields:
{{
  \"paper_title\": \"N/A or the full paper title\",
  \"journal\": \"N/A or the journal/conference name\",
  \"year\": \"N/A or the publication year (4 digits)\"
}}

Rules:
- Focus on the paper title, journal, and publication year.
- Do not infer any field from the filename.

Only output the JSON.
"""
        response = self.llm.call_api_with_retry(prompt)
        metadata = parse_llm_json_response(response, default_value={}, logger=self.logger)
        for key in ['paper_title', 'journal', 'year']:
            metadata[key] = metadata.get(key, 'N/A')
        return metadata

    def process_single_paper(self, pdf_path: str) -> Dict:
        filename = Path(pdf_path).name
        document_id = Path(filename).stem.strip()

        markdown_content = self.mineru.pdf_to_markdown(pdf_path)
        if not markdown_content:
            raise ValueError(f'PDF转换失败: {pdf_path}')

        inferred = self._extract_metadata_with_llm(markdown_content)
        extracted_doi = extract_doi_from_text(markdown_content)
        paper_metadata = {
            'DOI': extracted_doi or 'N/A',
            'document_id': document_id,
            'paper_title': inferred.get('paper_title', 'N/A') or 'N/A',
            'journal': inferred.get('journal', 'N/A') or 'N/A',
            'year': inferred.get('year', 'N/A') or 'N/A',
            'authors': 'N/A',
            'abstract': 'N/A',
            'level': '',
            'justification': '',
        }

        all_systems_result = self.system_identifier.identify_all_systems_with_types(markdown_content=markdown_content, reference_id=paper_metadata['DOI'])
        sam_systems = all_systems_result.get('systems', [])
        if self.molecule_image_extractor.enabled:
            self.molecule_image_extractor.extract_from_pdf(pdf_path)
        if not sam_systems:
            return {'doi': paper_metadata['DOI'], 'document_id': document_id, 'system_count': 0, 'knowledge_cards': []}

        system_count = len(sam_systems)
        has_multiple = system_count > 1
        knowledge_cards = []
        for idx, target_system in enumerate(sam_systems, 1):
            system_type = target_system.get('system_type', 'single')
            template = self.llm_extractor.load_template(system_type)
            if not template:
                continue
            knowledge_card = self.llm_extractor.extract_knowledge_with_llm(
                markdown_content=markdown_content,
                paper_metadata=paper_metadata,
                system_type=system_type,
                template=template,
                target_system=target_system,
                has_multiple=has_multiple,
            )
            final_card = self.card_integrator.integrate_knowledge_card(
                knowledge_card=knowledge_card,
                paper_metadata=paper_metadata,
                system_type=system_type,
                system_info={
                    'system_index': target_system.get('system_index', idx),
                    'system_count': system_count,
                    'system_name': target_system.get('system_name', ''),
                    'role': target_system.get('role', 'target'),
                },
                all_systems=sam_systems,
            )
            knowledge_cards.append(final_card)

        if len(knowledge_cards) > 1:
            knowledge_cards = self.card_integrator.deduplicate_knowledge_cards(knowledge_cards, document_id)
            system_count = len(knowledge_cards)

        return {'doi': paper_metadata.get('DOI', 'N/A'), 'document_id': document_id, 'system_count': system_count, 'knowledge_cards': knowledge_cards}

    def extract(self, path: str) -> Dict:
        if not path or not str(path).lower().endswith('.pdf'):
            raise ValueError('Literature_Extractor only supports PDF path input.')
        result = self.process_single_paper(path)
        return {
            'doi': result.get('doi', 'N/A'),
            'document_id': result.get('document_id', Path(path).stem),
            'system_count': result.get('system_count', 0),
            'knowledge_cards': result.get('knowledge_cards', []),
            'source_path': str(path),
        }

    def extract_as_json(self, path: str) -> str:
        return json.dumps(self.extract(path), ensure_ascii=False, indent=2)

