#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文献中的分子结构图识别接口。"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Heavy image-recognition dependencies are imported lazily in extract_from_pdf so
# the literature extractor can run when molecule image extraction is disabled.

MODULE_DIR = Path(__file__).resolve().parent


class MoleculeImageExtractor:
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = bool(self.config.get('enabled', False))

        output_dir = Path(self.config.get('output_dir', MODULE_DIR.parent / 'molecule_image_results'))
        if not output_dir.is_absolute():
            output_dir = MODULE_DIR.parent / output_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_pdf(self, pdf_path: str) -> Dict:
        pdf_path_obj = Path(pdf_path)
        document_id = pdf_path_obj.stem

        if not self.enabled:
            return {
                'enabled': False,
                'document_id': document_id,
                'candidate_count': 0,
                'candidates': [],
            }

        from .core.dected_img import dected_img

        work_dir = self.output_dir
        target_dir = work_dir / document_id
        if target_dir.exists():
            shutil.rmtree(target_dir)

        dected_img(str(pdf_path_obj), str(work_dir))

        norm_dir = target_dir / 'norm'
        org_dir = target_dir / 'org'
        if not norm_dir.exists():
            self.logger.warning(f'未检测到结构图输出目录: {norm_dir}')
            return {
                'enabled': True,
                'document_id': document_id,
                'candidate_count': 0,
                'candidates': [],
                'output_dir': str(target_dir),
            }

        try:
            from .core.decimer_smiles import predict_SMILES
        except Exception as exc:
            self.logger.warning(f'DECIMER SMILES模型加载失败，将仅输出结构图候选: {exc}')
            predict_SMILES = None

        candidates: List[Dict] = []
        for norm_file in sorted(norm_dir.iterdir()):
            if norm_file.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            org_file = org_dir / norm_file.name.replace('_norm_', '_org_')
            if predict_SMILES is None:
                smiles = 'N/A'
            else:
                try:
                    smiles = predict_SMILES(str(norm_file))
                except Exception as exc:
                    self.logger.warning(f'SMILES识别失败: {norm_file} - {exc}')
                    smiles = 'N/A'

            candidates.append({
                'norm_image_path': str(norm_file),
                'org_image_path': str(org_file) if org_file.exists() else 'N/A',
                'smiles': smiles,
            })

        return {
            'enabled': True,
            'document_id': document_id,
            'candidate_count': len(candidates),
            'candidates': candidates,
            'output_dir': str(target_dir),
        }

