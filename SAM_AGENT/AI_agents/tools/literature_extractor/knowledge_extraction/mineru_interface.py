#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinerU PDF解析接口模块。"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import requests


class MinerUInterface:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.url = self.config.get('mineru_url', 'http://127.0.0.1:8000/file_parse')
        self.local_md_dir = self.config.get('local_md_dir', 'temp_markdown')
        self.mineru_output_dir = self.config.get('mineru_output_dir', 'MinerU/output')
        os.makedirs(self.local_md_dir, exist_ok=True)

    def update_paths(self, local_md_dir: str, mineru_output_dir: str):
        self.local_md_dir = local_md_dir
        self.mineru_output_dir = mineru_output_dir
        os.makedirs(self.local_md_dir, exist_ok=True)

    def pdf_to_markdown(self, pdf_path: str) -> Optional[str]:
        try:
            pdf_name = Path(pdf_path).stem
            pdf_dir_md_path = str(Path(pdf_path).with_suffix('.md'))
            md_path = os.path.join(self.local_md_dir, f"{pdf_name}.md")

            for candidate_md_path in [pdf_dir_md_path, md_path]:
                if os.path.exists(candidate_md_path):
                    self.logger.info(f"Using existing Markdown file: {candidate_md_path}")
                    with open(candidate_md_path, 'r', encoding='utf-8') as f:
                        return f.read()

            params = {
                "parse_method": self.config.get('parse_method', 'ocr'),
                "output_dir": self.mineru_output_dir,
                "return_middle_json": "false",
                "return_model_output": "false",
                "return_md": "true",
                "return_images": "false",
                "end_page_id": "99999",
                "start_page_id": "0",
                "lang_list": "ch",
                "server_url": "string",
                "return_content_list": "false",
                "backend": "pipeline",
                "table_enable": "true",
                "formula_enable": "true",
                "response_format_zip": "false",
            }
            headers = {"accept": "application/json"}

            with open(pdf_path, "rb") as f:
                files = {"files": (os.path.basename(pdf_path), f, "application/pdf")}
                response = requests.post(self.url, params=params, headers=headers, files=files, timeout=300)

            if response.status_code != 200:
                self.logger.error(f"MinerU processing failed: {pdf_path}, status code: {response.status_code}")
                self.logger.error(f"Response: {response.text[:500]}")
                return None

            md_content = self._extract_markdown_from_response(response, pdf_name)
            if not md_content or len(md_content.strip()) <= 100:
                self.logger.error(f"Could not extract valid Markdown content from MinerU response for {pdf_path}")
                return None

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            return md_content
        except Exception as e:
            self.logger.error(f"PDF conversion exception {pdf_path}: {e}")
            return None

    def _extract_markdown_from_response(self, response: requests.Response, pdf_name: str) -> Optional[str]:
        try:
            response_json = response.json()
            md_content = response_json.get('markdown') or response_json.get('md') or response_json.get('content')
            if md_content:
                return md_content

            task_id = response_json.get('task_id') or response_json.get('taskId')
            output_path = response_json.get('output_path') or response_json.get('outputPath')
            if task_id:
                potential_md_paths = [
                    os.path.join(self.mineru_output_dir, task_id, pdf_name, 'auto', f'{pdf_name}.md'),
                    os.path.join(self.mineru_output_dir, task_id, 'auto', f'{pdf_name}.md'),
                    os.path.join(self.mineru_output_dir, task_id, f'{pdf_name}.md'),
                ]
                max_wait_time = 60
                wait_interval = 2
                waited_time = 0
                while waited_time < max_wait_time:
                    for potential_path in potential_md_paths:
                        if os.path.exists(potential_path):
                            file_size = os.path.getsize(potential_path)
                            time.sleep(1)
                            if os.path.getsize(potential_path) == file_size:
                                with open(potential_path, 'r', encoding='utf-8') as f:
                                    return f.read()
                    time.sleep(wait_interval)
                    waited_time += wait_interval
            elif output_path and os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    return f.read()

            if isinstance(response_json, str):
                return response_json
            if 'text' in response_json:
                return response_json['text']

            if os.path.exists(self.mineru_output_dir):
                target_filename = f'{pdf_name}.md'
                current_time = time.time()
                found_files = []
                for root, _, files in os.walk(self.mineru_output_dir):
                    for file in files:
                        if file == target_filename:
                            potential_md = os.path.join(root, file)
                            file_mtime = os.path.getmtime(potential_md)
                            if file_mtime > current_time - 600:
                                found_files.append((potential_md, file_mtime))
                if found_files:
                    found_files.sort(key=lambda x: x[1], reverse=True)
                    with open(found_files[0][0], 'r', encoding='utf-8') as f:
                        return f.read()
            return None
        except json.JSONDecodeError:
            md_content = response.text
            return md_content if md_content and len(md_content.strip()) > 100 else None
        except Exception as e:
            self.logger.error(f"Error extracting Markdown from response: {e}")
            return None

