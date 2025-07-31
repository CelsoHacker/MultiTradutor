#!/usr/bin/env python3
"""
Nintendo DS Translation Engine - Versão Completa
Sistema avançado de tradução automatizada para ROMs de Nintendo DS
Suporta extração, tradução via IA e validação de qualidade
"""

import os
import json
import re
import sqlite3
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TextEntry:
    """Representa uma entrada de texto extraída da ROM"""
    id: str
    text: str
    context: str
    offset: int
    size: int
    encoding: str
    translation: Optional[str] = None
    confidence: float = 0.0
    validated: bool = False
    notes: str = ""

@dataclass
class TranslationProject:
    """Representa um projeto de tradução"""
    name: str
    rom_path: str
    source_language: str
    target_language: str
    created_at: str
    modified_at: str
    status: str = "active"

class DatabaseManager:
    """Gerencia o banco de dados SQLite para o projeto"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Inicializa o banco de dados com as tabelas necessárias"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabela de projetos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    rom_path TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            ''')

            # Tabela de entradas de texto
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS text_entries (
                    id TEXT PRIMARY KEY,
                    project_id INTEGER,
                    text TEXT NOT NULL,
                    context TEXT,
                    offset INTEGER,
                    size INTEGER,
                    encoding TEXT,
                    translation TEXT,
                    confidence REAL DEFAULT 0.0,
                    validated BOOLEAN DEFAULT 0,
                    notes TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')

            # Tabela de cache de traduções
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS translation_cache (
                    text_hash TEXT PRIMARY KEY,
                    original_text TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    confidence REAL,
                    created_at TEXT NOT NULL
                )
            ''')

            # Índices para melhor performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_entries_project ON text_entries(project_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_hash ON translation_cache(text_hash)')

            conn.commit()

    def save_project(self, project: TranslationProject) -> int:
        """Salva um projeto no banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO projects
                (name, rom_path, source_language, target_language, created_at, modified_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                project.name, project.rom_path, project.source_language,
                project.target_language, project.created_at, project.modified_at, project.status
            ))
            return cursor.lastrowid

    def save_text_entries(self, project_id: int, entries: List[TextEntry]):
        """Salva múltiplas entradas de texto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for entry in entries:
                cursor.execute('''
                    INSERT OR REPLACE INTO text_entries
                    (id, project_id, text, context, offset, size, encoding, translation, confidence, validated, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.id, project_id, entry.text, entry.context, entry.offset,
                    entry.size, entry.encoding, entry.translation, entry.confidence,
                    entry.validated, entry.notes
                ))
            conn.commit()

    def get_text_entries(self, project_id: int) -> List[TextEntry]:
        """Recupera todas as entradas de texto de um projeto"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, text, context, offset, size, encoding, translation, confidence, validated, notes
                FROM text_entries WHERE project_id = ?
            ''', (project_id,))

            entries = []
            for row in cursor.fetchall():
                entry = TextEntry(
                    id=row[0], text=row[1], context=row[2], offset=row[3],
                    size=row[4], encoding=row[5], translation=row[6],
                    confidence=row[7], validated=bool(row[8]), notes=row[9]
                )
                entries.append(entry)
            return entries

    def cache_translation(self, text: str, translation: str, source_lang: str, target_lang: str, confidence: float):
        """Cacheia uma tradução para reuso futuro"""
        text_hash = hashlib.md5(f"{text}_{source_lang}_{target_lang}".encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO translation_cache
                (text_hash, original_text, translation, source_lang, target_lang, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (text_hash, text, translation, source_lang, target_lang, confidence, datetime.now().isoformat()))
            conn.commit()

    def get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[Tuple[str, float]]:
        """Recupera uma tradução do cache"""
        text_hash = hashlib.md5(f"{text}_{source_lang}_{target_lang}".encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT translation, confidence FROM translation_cache
                WHERE text_hash = ?
            ''', (text_hash,))

            result = cursor.fetchone()
            return (result[0], result[1]) if result else None

class ROMTextExtractor:
    """Extrai texto de arquivos ROM do Nintendo DS"""

    # Padrões comuns de texto em ROMs
    TEXT_PATTERNS = [
        rb'[\x20-\x7E]{4,}',  # ASCII imprimível
        rb'[\x81-\x9F\xE0-\xFC][\x40-\x7E\x80-\xFC]{3,}',  # Shift-JIS
        rb'[\xA1-\xFE]{4,}',  # UTF-8 multi-byte
    ]

    def __init__(self):
        self.extracted_entries = []

    def extract_from_file(self, file_path: str, context: str = "ROM") -> List[TextEntry]:
        """Extrai texto de um arquivo ROM"""
        logger.info(f"Extraindo texto de: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                rom_data = f.read()

            entries = []
            entry_id = 0

            for pattern in self.TEXT_PATTERNS:
                matches = re.finditer(pattern, rom_data)

                for match in matches:
                    text_bytes = match.group()
                    offset = match.start()

                    # Tenta decodificar com diferentes encodings
                    for encoding in ['utf-8', 'shift-jis', 'ascii', 'latin-1']:
                        try:
                            decoded_text = text_bytes.decode(encoding)

                            # Filtra texto válido
                            if self._is_valid_text(decoded_text):
                                entry = TextEntry(
                                    id=f"{context}_{entry_id:06d}",
                                    text=decoded_text.strip(),
                                    context=context,
                                    offset=offset,
                                    size=len(text_bytes),
                                    encoding=encoding
                                )
                                entries.append(entry)
                                entry_id += 1
                                break
                        except UnicodeDecodeError:
                            continue

            logger.info(f"Extraídas {len(entries)} entradas de texto")
            return entries

        except Exception as e:
            logger.error(f"Erro ao extrair texto de {file_path}: {e}")
            return []

    def _is_valid_text(self, text: str) -> bool:
        """Verifica se o texto extraído é válido"""
        # Remove strings muito curtas
        if len(text) < 4:
            return False

        # Remove strings com muitos caracteres especiais
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-')
        if special_chars / len(text) > 0.3:
            return False

        # Remove strings que parecem dados binários
        if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
            return False

        return True

class AITranslator:
    """Interface para tradução usando IA"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.translation_cache = {}

    def translate_text(self, text: str, source_lang: str, target_lang: str, context: str = "") -> Tuple[str, float]:
        """Traduz um texto usando IA"""
        try:
            # Simula tradução (substitua por implementação real da API)
            if source_lang == "en" and target_lang == "pt":
                # Exemplo simples de tradução
                translations = {
                    "Start": "Iniciar",
                    "Options": "Opções",
                    "Exit": "Sair",
                    "Save": "Salvar",
                    "Load": "Carregar",
                    "Menu": "Menu"
                }

                translation = translations.get(text, f"[TRADUZIR: {text}]")
                confidence = 0.9 if text in translations else 0.5

                return translation, confidence

            # Para outras combinações de idiomas
            return f"[{target_lang.upper()}: {text}]", 0.5

        except Exception as e:
            logger.error(f"Erro ao traduzir '{text}': {e}")
            return text, 0.0

    def translate_batch(self, entries: List[TextEntry], source_lang: str, target_lang: str,
                       max_workers: int = 5) -> List[TextEntry]:
        """Traduz múltiplas entradas em paralelo"""
        logger.info(f"Traduzindo {len(entries)} entradas...")

        translated_entries = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(self.translate_text, entry.text, source_lang, target_lang, entry.context): entry
                for entry in entries
            }

            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    translation, confidence = future.result()
                    entry.translation = translation
                    entry.confidence = confidence
                    translated_entries.append(entry)
                except Exception as e:
                    logger.error(f"Erro ao traduzir entrada {entry.id}: {e}")
                    entry.translation = entry.text
                    entry.confidence = 0.0
                    translated_entries.append(entry)

        logger.info(f"Tradução concluída para {len(translated_entries)} entradas")
        return translated_entries

class QualityValidator:
    """Valida a qualidade das traduções"""

    def __init__(self):
        self.validation_rules = [
            self._check_length_consistency,
            self._check_character_encoding,
            self._check_placeholder_integrity,
            self._check_identical_translations
        ]

    def validate_translations(self, entries: List[TextEntry]) -> Dict[str, List[Dict]]:
        """Valida todas as traduções e retorna relatório de qualidade"""
        report = {
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        logger.info("Iniciando validação de qualidade...")

        for entry in entries:
            if not entry.translation:
                continue

            for rule in self.validation_rules:
                result = rule(entry)
                if result:
                    report[result['type']].append(result)

        # Análise de consistência global
        consistency_issues = self._analyze_consistency(entries)
        report['warnings'].extend(consistency_issues)

        # Estatísticas gerais
        report['statistics'] = self._generate_statistics(entries)

        logger.info(f"Validação concluída. Encontrados {len(report['errors'])} erros e {len(report['warnings'])} avisos")
        return report

    def _check_length_consistency(self, entry: TextEntry) -> Optional[Dict]:
        """Verifica consistência de comprimento"""
        if not entry.translation:
            return None

        original_len = len(entry.text)
        translation_len = len(entry.translation)

        # Aviso se a tradução for muito diferente do original
        if translation_len > original_len * 2:
            return {
                'type': 'warnings',
                'rule': 'length_consistency',
                'id': entry.id,
                'message': f"Tradução muito longa ({translation_len} vs {original_len} caracteres)",
                'original': entry.text,
                'translation': entry.translation
            }

        return None

    def _check_character_encoding(self, entry: TextEntry) -> Optional[Dict]:
        """Verifica problemas de encoding"""
        if not entry.translation:
            return None

        # Verifica caracteres problemáticos
        problematic_chars = ['\uFFFD', '???', '□']
        for char in problematic_chars:
            if char in entry.translation:
                return {
                    'type': 'errors',
                    'rule': 'character_encoding',
                    'id': entry.id,
                    'message': f"Caractere problemático encontrado: {char}",
                    'original': entry.text,
                    'translation': entry.translation
                }

        return None

    def _check_placeholder_integrity(self, entry: TextEntry) -> Optional[Dict]:
        """Verifica integridade de placeholders"""
        if not entry.translation:
            return None

        # Encontra placeholders no texto original
        original_placeholders = re.findall(r'%[sd]|\{[^}]+\}|<[^>]+>', entry.text)
        translation_placeholders = re.findall(r'%[sd]|\{[^}]+\}|<[^>]+>', entry.translation)

        if set(original_placeholders) != set(translation_placeholders):
            return {
                'type': 'errors',
                'rule': 'placeholder_integrity',
                'id': entry.id,
                'message': f"Placeholders inconsistentes: {original_placeholders} vs {translation_placeholders}",
                'original': entry.text,
                'translation': entry.translation
            }

        return None

    def _check_identical_translations(self, entry: TextEntry) -> Optional[Dict]:
        """Verifica traduções idênticas ao original"""
        if not entry.translation:
            return None

        if entry.translation == entry.text:
            return {
                'type': 'warnings',
                'rule': 'identical_translation',
                'id': entry.id,
                'message': "Tradução idêntica ao original (possível não traduzido)",
                'original': entry.text,
                'translation': entry.translation
            }

        return None

    def _analyze_consistency(self, entries: List[TextEntry]) -> List[Dict]:
        """Analisa consistência entre traduções"""
        potential_errors = []

        # Mapa de traduções para detectar inconsistências
        translation_map = {}
        for entry in entries:
            if not entry.translation:
                continue

            if entry.text in translation_map:
                # Verifica se a mesma string foi traduzida diferentemente
                if translation_map[entry.text] != entry.translation:
                    potential_errors.append({
                        'type': 'warnings',
                        'rule': 'consistency_check',
                        'message': f"Tradução inconsistente para '{entry.text}'",
                        'translations': [translation_map[entry.text], entry.translation],
                        'entries': [entry.id]
                    })
            else:
                translation_map[entry.text] = entry.translation

        return potential_errors

    def _generate_statistics(self, entries: List[TextEntry]) -> Dict:
        """Gera estatísticas sobre as traduções"""
        total_entries = len(entries)
        translated_entries = sum(1 for e in entries if e.translation)
        validated_entries = sum(1 for e in entries if e.validated)

        if translated_entries > 0:
            avg_confidence = sum(e.confidence for e in entries if e.translation) / translated_entries
        else:
            avg_confidence = 0.0

        return {
            'total_entries': total_entries,
            'translated_entries': translated_entries,
            'validated_entries': validated_entries,
            'translation_percentage': (translated_entries / total_entries) * 100 if total_entries > 0 else 0,
            'validation_percentage': (validated_entries / total_entries) * 100 if total_entries > 0 else 0,
            'average_confidence': avg_confidence
        }

class TranslationEngine:
    """Engine principal de tradução"""

    def __init__(self, database_path: str = "translation_projects.db"):
        self.db_manager = DatabaseManager(database_path)
        self.extractor = ROMTextExtractor()
        self.translator = AITranslator()
        self.validator = QualityValidator()
        self.current_project = None
        self.current_project_id = None

    def create_project(self, name: str, rom_path: str, source_lang: str, target_lang: str) -> bool:
        """Cria um novo projeto de tradução"""
        try:
            project = TranslationProject(
                name=name,
                rom_path=rom_path,
                source_language=source_lang,
                target_language=target_lang,
                created_at=datetime.now().isoformat(),
                modified_at=datetime.now().isoformat()
            )

            project_id = self.db_manager.save_project(project)
            self.current_project = project
            self.current_project_id = project_id

            logger.info(f"Projeto '{name}' criado com sucesso (ID: {project_id})")
            return True

        except Exception as e:
            logger.error(f"Erro ao criar projeto: {e}")
            return False

    def extract_text_from_rom(self, rom_path: str) -> List[TextEntry]:
        """Extrai texto de uma ROM"""
        if not os.path.exists(rom_path):
            logger.error(f"Arquivo ROM não encontrado: {rom_path}")
            return []

        return self.extractor.extract_from_file(rom_path, "ROM")

    def translate_project(self, entries: List[TextEntry], use_cache: bool = True) -> List[TextEntry]:
        """Traduz todas as entradas de um projeto"""
        if not self.current_project:
            logger.error("Nenhum projeto ativo")
            return []

        # Verifica cache para traduções existentes
        if use_cache:
            for entry in entries:
                cached = self.db_manager.get_cached_translation(
                    entry.text,
                    self.current_project.source_language,
                    self.current_project.target_language
                )
                if cached:
                    entry.translation, entry.confidence = cached

        # Traduz entradas não cacheadas
        untranslated = [e for e in entries if not e.translation]
        if untranslated:
            translated = self.translator.translate_batch(
                untranslated,
                self.current_project.source_language,
                self.current_project.target_language
            )

            # Cacheia novas traduções
            for entry in translated:
                if entry.translation:
                    self.db_manager.cache_translation(
                        entry.text, entry.translation,
                        self.current_project.source_language,
                        self.current_project.target_language,
                        entry.confidence
                    )

        return entries

    def validate_project(self, entries: List[TextEntry]) -> Dict:
        """Valida a qualidade das traduções do projeto"""
        return self.validator.validate_translations(entries)

    def save_project_progress(self, entries: List[TextEntry]):
        """Salva o progresso do projeto"""
        if not self.current_project_id:
            logger.error("Nenhum projeto ativo para salvar")
            return

        self.db_manager.save_text_entries(self.current_project_id, entries)
        logger.info(f"Progresso salvo para {len(entries)} entradas")

    def export_translations(self, entries: List[TextEntry], output_path: str, format: str = "json"):
        """Exporta traduções para diferentes formatos"""
        try:
            if format == "json":
                data = {
                    'project': asdict(self.current_project) if self.current_project else {},
                    'entries': [asdict(entry) for entry in entries],
                    'exported_at': datetime.now().isoformat()
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            elif format == "csv":
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Original', 'Tradução', 'Confiança', 'Validado'])
                    for entry in entries:
                        writer.writerow([
                            entry.id, entry.text, entry.translation or '',
                            entry.confidence, entry.validated
                        ])

            logger.info(f"Traduções exportadas para: {output_path}")

        except Exception as e:
            logger.error(f"Erro ao exportar traduções: {e}")

    def run_full_pipeline(self, project_name: str, rom_path: str, source_lang: str, target_lang: str):
        """Executa o pipeline completo de tradução"""
        logger.info("=== INICIANDO PIPELINE DE TRADUÇÃO ===")

        # 1. Criar projeto
        if not self.create_project(project_name, rom_path, source_lang, target_lang):
            return False

        # 2. Extrair texto
        logger.info("Etapa 1: Extraindo texto da ROM...")
        entries = self.extract_text_from_rom(rom_path)
        if not entries:
            logger.error("Nenhum texto extraído da ROM")
            return False

        # 3. Traduzir
        logger.info("Etapa 2: Traduzindo texto...")
        translated_entries = self.translate_project(entries)

        # 4. Validar
        logger.info("Etapa 3: Validando traduções...")
        validation_report = self.validate_project(translated_entries)

        # 5. Salvar progresso
        logger.info("Etapa 4: Salvando progresso...")
        self.save_project_progress(translated_entries)

        # 6. Exportar resultados
        logger.info("Etapa 5: Exportando resultados...")
        self.export_translations(translated_entries, f"{project_name}_translations.json")

        # 7. Relatório final
        logger.info("=== RELATÓRIO FINAL ===")
        stats = validation_report['statistics']
        logger.info(f"Entradas processadas: {stats['total_entries']}")
        logger.info(f"Traduções realizadas: {stats['translated_entries']} ({stats['translation_percentage']:.1f}%)")
        logger.info(f"Confiança média: {stats['average_confidence']:.2f}")
        logger.info(f"Erros encontrados: {len(validation_report['errors'])}")
        logger.info(f"Avisos: {len(validation_report['warnings'])}")

        return True

def main():
    """Função principal de demonstração"""
    print("🎮 Nintendo DS Translation Engine v1.0")
    print("=" * 50)

    # Inicializar engine
    engine = TranslationEngine()

    # Exemplo de uso
    rom_path = "exemplo.nds"  # Substitua pelo caminho real

    # Criar um arquivo de exemplo para demonstração
    if not os.path.exists(rom_path):
        print(f"Criando arquivo de exemplo em: {rom_path}")
        with open(rom_path, 'wb') as f:
            # Dados de exemplo que simulariam uma ROM
            example_data = (
                b"Start\x00\x00\x00Options\x00\x00Exit\x00\x00\x00"
                b"Save Game\x00Load Game\x00\x00Settings\x00\x00"
                b"Back\x00\x00\x00Continue\x00\x00Help\x00\x00\x00"
            )
            f.write(example_data)

    # Executar pipeline completo
    success = engine.run_full_pipeline(
        project_name="Exemplo DS Game",
        rom_path=rom_path,
        source_lang="en",
        target_lang="pt"
    )

    if success:
        print("\n✅ Pipeline executado com sucesso!")
        print("Verifique os arquivos gerados:")
        print("- translation_projects.db (banco de dados)")
        print("- Exemplo DS Game_translations.json (traduções)")
        print("- translation_engine.log (log detalhado)")
    else:
        print("\n❌ Erro durante a execução do pipeline")

if __name__ == "__main__":
    main()