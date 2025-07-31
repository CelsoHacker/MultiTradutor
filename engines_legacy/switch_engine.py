#!/usr/bin/env python3
"""
Nintendo Switch ROM Translation Engine
Automatiza tradução de jogos Switch (NSP/XCI) usando IA
Lida com estruturas complexas de arquivos modernos
"""

import os
import json
import hashlib
import logging
import struct
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
import time
import tempfile
import shutil

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('switch_translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SwitchFileType(Enum):
    """Tipos de arquivo Switch suportados"""
    NSP = "nsp"  # Nintendo Submission Package
    XCI = "xci"  # NX Card Image
    NCA = "nca"  # Nintendo Content Archive
    ROMFS = "romfs"  # ROM File System
    EXEFS = "exefs"  # Executable File System

class TextFileFormat(Enum):
    """Formatos de arquivo de texto comuns no Switch"""
    MSBT = "msbt"  # Message Studio Binary Text
    JSON = "json"  # JSON localizado
    XML = "xml"   # XML localizado
    TXT = "txt"   # Texto simples
    YAML = "yaml" # YAML localizado
    CSV = "csv"   # CSV localizado

@dataclass
class SwitchTextEntry:
    """Entrada de texto específica para Switch"""
    file_path: str
    file_format: TextFileFormat
    entry_id: str
    original_text: str
    translated_text: str = ""
    context: str = ""
    attributes: Dict[str, Any] = None
    character_limit: Optional[int] = None
    is_translated: bool = False

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class SwitchGameMetadata:
    """Metadados específicos do jogo Switch"""
    title_id: str
    game_name: str
    version: str
    file_type: SwitchFileType
    file_size: int
    checksum: str
    supported_languages: List[str]
    text_entries: List[SwitchTextEntry] = None
    total_text_files: int = 0
    extraction_path: str = ""

    def __post_init__(self):
        if self.text_entries is None:
            self.text_entries = []

class MSBTParser:
    """Parser para arquivos MSBT (Message Studio Binary Text)"""

    def __init__(self):
        self.encoding = 'utf-16le'

    def parse_msbt(self, file_path: Path) -> List[SwitchTextEntry]:
        """Extrai texto de arquivos MSBT"""
        entries = []

        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            # Verifica header MSBT
            if data[:8] != b'MsgStdBn':
                logger.warning(f"Arquivo {file_path} não é um MSBT válido")
                return entries

            # Parse do header
            header = struct.unpack('<8sHHHHHH', data[:20])
            section_count = header[6]

            offset = 20
            sections = []

            # Lê seções
            for i in range(section_count):
                section_header = struct.unpack('<4sI8s', data[offset:offset+16])
                section_type = section_header[0]
                section_size = section_header[1]

                sections.append({
                    'type': section_type,
                    'size': section_size,
                    'offset': offset + 16
                })

                offset += 16 + section_size
                # Alinhamento para 16 bytes
                offset = (offset + 15) & ~15

            # Processa seção TXT2 (texto)
            for section in sections:
                if section['type'] == b'TXT2':
                    entries.extend(self._parse_txt2_section(
                        data[section['offset']:section['offset'] + section['size']],
                        file_path
                    ))

            logger.info(f"Extraídas {len(entries)} entradas de {file_path}")
            return entries

        except Exception as e:
            logger.error(f"Erro ao processar MSBT {file_path}: {e}")
            return entries

    def _parse_txt2_section(self, section_data: bytes, file_path: Path) -> List[SwitchTextEntry]:
        """Processa seção TXT2 do MSBT"""
        entries = []

        try:
            # Lê número de strings
            string_count = struct.unpack('<I', section_data[:4])[0]
            offset = 4

            # Lê offsets das strings
            string_offsets = []
            for i in range(string_count):
                string_offset = struct.unpack('<I', section_data[offset:offset+4])[0]
                string_offsets.append(string_offset)
                offset += 4

            # Extrai strings
            for i, string_offset in enumerate(string_offsets):
                try:
                    # Calcula próximo offset
                    next_offset = string_offsets[i + 1] if i + 1 < len(string_offsets) else len(section_data)

                    # Extrai string UTF-16LE
                    string_data = section_data[string_offset:next_offset]

                    # Remove null terminators
                    string_data = string_data.rstrip(b'\x00')

                    if string_data:
                        text = string_data.decode('utf-16le', errors='ignore')

                        # Remove caracteres de controle específicos do Switch
                        text = self._clean_switch_text(text)

                        if text.strip():
                            entry = SwitchTextEntry(
                                file_path=str(file_path),
                                file_format=TextFileFormat.MSBT,
                                entry_id=f"string_{i:04d}",
                                original_text=text,
                                context=f"MSBT entry {i}"
                            )
                            entries.append(entry)

                except Exception as e:
                    logger.debug(f"Erro ao extrair string {i}: {e}")
                    continue

            return entries

        except Exception as e:
            logger.error(f"Erro ao processar seção TXT2: {e}")
            return entries

    def _clean_switch_text(self, text: str) -> str:
        """Remove caracteres de controle específicos do Switch"""
        # Remove tags de controle comuns
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\x0E[^\x0F]*\x0F', '', text)  # Tags de cor
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Caracteres de controle
        return text.strip()

class SwitchFileExtractor:
    """Extrator de arquivos Switch (NSP/XCI)"""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix='switch_extract_'))
        self.supported_extensions = ['.nsp', '.xci']
        self.text_extractors = {
            TextFileFormat.MSBT: MSBTParser(),
            TextFileFormat.JSON: self._extract_json_text,
            TextFileFormat.XML: self._extract_xml_text,
            TextFileFormat.TXT: self._extract_txt_text,
        }

    def extract_game_files(self, game_path: Path) -> Optional[Path]:
        """Extrai arquivos do jogo Switch"""

        if game_path.suffix.lower() == '.nsp':
            return self._extract_nsp(game_path)
        elif game_path.suffix.lower() == '.xci':
            return self._extract_xci(game_path)
        else:
            logger.error(f"Formato não suportado: {game_path.suffix}")
            return None

    def _extract_nsp(self, nsp_path: Path) -> Optional[Path]:
        """Extrai arquivos NSP (basicamente um PFS0)"""
        try:
            extract_path = self.temp_dir / f"nsp_{nsp_path.stem}"
            extract_path.mkdir(parents=True, exist_ok=True)

            with open(nsp_path, 'rb') as f:
                # Lê header PFS0
                header = f.read(16)
                if header[:4] != b'PFS0':
                    logger.error(f"NSP inválido: {nsp_path}")
                    return None

                file_count = struct.unpack('<I', header[4:8])[0]
                string_table_size = struct.unpack('<I', header[8:12])[0]

                # Lê entradas de arquivo
                file_entries = []
                for i in range(file_count):
                    entry_data = f.read(24)
                    offset, size, name_offset = struct.unpack('<QQI', entry_data[:20])
                    file_entries.append({
                        'offset': offset,
                        'size': size,
                        'name_offset': name_offset
                    })

                # Lê tabela de strings
                string_table = f.read(string_table_size)

                # Extrai arquivos
                for i, entry in enumerate(file_entries):
                    # Obtém nome do arquivo
                    name_start = entry['name_offset']
                    name_end = string_table.find(b'\x00', name_start)
                    if name_end == -1:
                        name_end = len(string_table)

                    filename = string_table[name_start:name_end].decode('utf-8')

                    # Lê dados do arquivo
                    f.seek(16 + file_count * 24 + string_table_size + entry['offset'])
                    file_data = f.read(entry['size'])

                    # Salva arquivo
                    output_path = extract_path / filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_path, 'wb') as out_file:
                        out_file.write(file_data)

                logger.info(f"NSP extraído em: {extract_path}")
                return extract_path

        except Exception as e:
            logger.error(f"Erro ao extrair NSP {nsp_path}: {e}")
            return None

    def _extract_xci(self, xci_path: Path) -> Optional[Path]:
        """Extrai arquivos XCI (implementação básica)"""
        # XCI é mais complexo - implementação simplificada
        logger.warning("Extração XCI ainda não implementada completamente")
        return None

    def find_text_files(self, extracted_path: Path) -> List[Path]:
        """Encontra arquivos de texto nos arquivos extraídos"""
        text_files = []

        # Padrões de busca para arquivos de texto
        patterns = [
            '**/*.msbt',
            '**/*message*.json',
            '**/*text*.json',
            '**/*localization*.json',
            '**/*lang*.json',
            '**/*.xml',
            '**/*strings*.txt',
        ]

        for pattern in patterns:
            text_files.extend(extracted_path.glob(pattern))

        # Filtra por diretórios de idioma
        language_dirs = ['en', 'ja', 'pt', 'es', 'fr', 'de', 'it', 'ko', 'zh']
        for lang_dir in language_dirs:
            for ext in ['.msbt', '.json', '.xml', '.txt']:
                text_files.extend(extracted_path.glob(f'**/{lang_dir}/**/*{ext}'))

        return list(set(text_files))  # Remove duplicatas

    def extract_text_from_files(self, text_files: List[Path]) -> List[SwitchTextEntry]:
        """Extrai texto de todos os arquivos encontrados"""
        all_entries = []

        for file_path in text_files:
            try:
                # Determina formato do arquivo
                file_format = self._detect_file_format(file_path)

                # Extrai texto baseado no formato
                if file_format == TextFileFormat.MSBT:
                    entries = self.text_extractors[file_format].parse_msbt(file_path)
                elif file_format in self.text_extractors:
                    entries = self.text_extractors[file_format](file_path)
                else:
                    logger.warning(f"Formato não suportado: {file_path}")
                    continue

                all_entries.extend(entries)

            except Exception as e:
                logger.error(f"Erro ao processar arquivo {file_path}: {e}")
                continue

        return all_entries

    def _detect_file_format(self, file_path: Path) -> TextFileFormat:
        """Detecta formato do arquivo de texto"""
        extension = file_path.suffix.lower()

        format_map = {
            '.msbt': TextFileFormat.MSBT,
            '.json': TextFileFormat.JSON,
            '.xml': TextFileFormat.XML,
            '.txt': TextFileFormat.TXT,
            '.yaml': TextFileFormat.YAML,
            '.yml': TextFileFormat.YAML,
            '.csv': TextFileFormat.CSV,
        }

        return format_map.get(extension, TextFileFormat.TXT)

    def _extract_json_text(self, file_path: Path) -> List[SwitchTextEntry]:
        """Extrai texto de arquivos JSON"""
        entries = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries = self._traverse_json(data, file_path, "")

        except Exception as e:
            logger.error(f"Erro ao processar JSON {file_path}: {e}")

        return entries

    def _traverse_json(self, data: Any, file_path: Path, path: str) -> List[SwitchTextEntry]:
        """Traversa recursivamente JSON procurando por texto"""
        entries = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, str) and len(value.strip()) > 2:
                    # Filtra strings que parecem ser texto de interface
                    if self._is_ui_text(value):
                        entry = SwitchTextEntry(
                            file_path=str(file_path),
                            file_format=TextFileFormat.JSON,
                            entry_id=current_path,
                            original_text=value,
                            context=f"JSON path: {current_path}"
                        )
                        entries.append(entry)

                elif isinstance(value, (dict, list)):
                    entries.extend(self._traverse_json(value, file_path, current_path))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"

                if isinstance(item, str) and len(item.strip()) > 2:
                    if self._is_ui_text(item):
                        entry = SwitchTextEntry(
                            file_path=str(file_path),
                            file_format=TextFileFormat.JSON,
                            entry_id=current_path,
                            original_text=item,
                            context=f"JSON array index: {i}"
                        )
                        entries.append(entry)

                elif isinstance(item, (dict, list)):
                    entries.extend(self._traverse_json(item, file_path, current_path))

        return entries

    def _is_ui_text(self, text: str) -> bool:
        """Verifica se o texto parece ser de interface do usuário"""
        # Remove strings muito curtas ou que são claramente IDs
        if len(text) < 3:
            return False

        # Remove strings que são só números, IDs ou paths
        if re.match(r'^[\d\._\-/\\]+$', text):
            return False

        # Remove strings que são claramente identificadores
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', text) and text.islower():
            return False

        return True

    def _extract_xml_text(self, file_path: Path) -> List[SwitchTextEntry]:
        """Extrai texto de arquivos XML"""
        entries = []

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            for elem in root.iter():
                if elem.text and elem.text.strip():
                    if self._is_ui_text(elem.text):
                        entry = SwitchTextEntry(
                            file_path=str(file_path),
                            file_format=TextFileFormat.XML,
                            entry_id=f"{elem.tag}_{elem.attrib.get('id', 'unknown')}",
                            original_text=elem.text.strip(),
                            context=f"XML element: {elem.tag}"
                        )
                        entries.append(entry)

        except Exception as e:
            logger.error(f"Erro ao processar XML {file_path}: {e}")

        return entries

    def _extract_txt_text(self, file_path: Path) -> List[SwitchTextEntry]:
        """Extrai texto de arquivos TXT simples"""
        entries = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if line and self._is_ui_text(line):
                    entry = SwitchTextEntry(
                        file_path=str(file_path),
                        file_format=TextFileFormat.TXT,
                        entry_id=f"line_{i:04d}",
                        original_text=line,
                        context=f"Line {i+1}"
                    )
                    entries.append(entry)

        except Exception as e:
            logger.error(f"Erro ao processar TXT {file_path}: {e}")

        return entries

    def cleanup(self):
        """Limpa arquivos temporários"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

class SwitchTranslationService:
    """Serviço de tradução especializado para Switch"""

    def __init__(self, source_lang: str = "ja", target_lang: str = "pt"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translation_cache = {}
        self.load_cache()

        # Contextos específicos para games
        self.game_contexts = {
            'menu': ['menu', 'button', 'option', 'setting'],
            'dialogue': ['dialogue', 'conversation', 'speech'],
            'item': ['item', 'weapon', 'armor', 'consumable'],
            'story': ['story', 'narrative', 'cutscene'],
            'tutorial': ['tutorial', 'help', 'guide', 'instruction'],
            'ui': ['ui', 'interface', 'hud', 'status']
        }

    def translate_entry(self, entry: SwitchTextEntry) -> str:
        """Traduz uma entrada de texto com contexto de jogo"""

        # Verifica cache
        cache_key = self._get_cache_key(entry)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        # Detecta contexto do jogo
        context_type = self._detect_game_context(entry)

        # Traduz com contexto específico
        translated = self._translate_with_context(entry.original_text, context_type, entry.context)

        # Aplica pós-processamento específico do Switch
        translated = self._post_process_switch_text(translated, entry)

        # Salva no cache
        self.translation_cache[cache_key] = translated
        self.save_cache()

        return translated

    def _get_cache_key(self, entry: SwitchTextEntry) -> str:
        """Gera chave de cache para a entrada"""
        return f"{entry.original_text}_{entry.file_format.value}_{entry.context}"

    def _detect_game_context(self, entry: SwitchTextEntry) -> str:
        """Detecta contexto específico do jogo"""
        text_lower = entry.original_text.lower()
        context_lower = entry.context.lower()
        file_path_lower = entry.file_path.lower()

        # Analisa path do arquivo
        if 'menu' in file_path_lower or 'ui' in file_path_lower:
            return 'ui'
        elif 'dialogue' in file_path_lower or 'conversation' in file_path_lower:
            return 'dialogue'
        elif 'item' in file_path_lower or 'equipment' in file_path_lower:
            return 'item'
        elif 'story' in file_path_lower or 'scenario' in file_path_lower:
            return 'story'
        elif 'tutorial' in file_path_lower or 'help' in file_path_lower:
            return 'tutorial'

        # Analisa contexto da entrada
        for context_type, keywords in self.game_contexts.items():
            if any(keyword in context_lower for keyword in keywords):
                return context_type

        # Analisa o texto em si
        if any(word in text_lower for word in ['press', 'button', 'select', 'choose']):
            return 'ui'
        elif any(word in text_lower for word in ['sword', 'potion', 'armor', 'hp', 'mp']):
            return 'item'
        elif len(text_lower) > 100:  # Textos longos geralmente são diálogos
            return 'dialogue'

        return 'general'

    def _translate_with_context(self, text: str, context_type: str, context: str) -> str:
        """Traduz texto com contexto específico"""

        # Aqui você integraria com APIs de tradução
        # Por enquanto, usando traduções mockadas mais sofisticadas

        context_prompts = {
            'ui': "Traduza este texto de interface de jogo:",
            'dialogue': "Traduza este diálogo de jogo:",
            'item': "Traduza este nome/descrição de item:",
            'story': "Traduza esta narrativa de jogo:",
            'tutorial': "Traduza esta instrução de tutorial:",
            'menu': "Traduza este item de menu:",
            'general': "Traduza este texto de jogo:"
        }

        prompt = context_prompts.get(context_type, context_prompts['general'])

        # Implementação mock mais sofisticada
        return self._advanced_mock_translate(text, context_type)

    def _advanced_mock_translate(self, text: str, context_type: str) -> str:
        """Tradução mock mais sofisticada baseada em contexto"""

        # Dicionários específicos por contexto
        translations = {
            'ui': {
                'New Game': 'Novo Jogo',
                'Continue': 'Continuar',
                'Load Game': 'Carregar Jogo',
                'Save Game': 'Salvar Jogo',
                'Settings': 'Configurações',
                'Options': 'Opções',
                'Exit': 'Sair',
                'Back': 'Voltar',
                'Confirm': 'Confirmar',
                'Cancel': 'Cancelar',
                'Yes': 'Sim',
                'No': 'Não',
                'OK': 'OK',
                'Menu': 'Menu',
                'Pause': 'Pausar',
                'Resume': 'Continuar',
                'Restart': 'Reiniciar',
                'Level': 'Nível',
                'Stage': 'Estágio',
                'Score': 'Pontuação',
                'Lives': 'Vidas',
                'Health': 'Vida',
                'Mana': 'Mana',
                'Experience': 'Experiência',
                'Gold': 'Ouro',
                'Money': 'Dinheiro'
            },
            'item': {
                'Sword': 'Espada',
                'Shield': 'Escudo',
                'Potion': 'Poção',
                'Armor': 'Armadura',
                'Weapon': 'Arma',
                'Helmet': 'Capacete',
                'Boots': 'Botas',
                'Ring': 'Anel',
                'Necklace': 'Colar',
                'Bow': 'Arco',
                'Arrow': 'Flecha',
                'Magic': 'Magia',
                'Spell': 'Feitiço',
                'Key': 'Chave',
                'Gem': 'Gema',
                'Crystal': 'Cristal'
            },
            'dialogue': {
                'Hello': 'Olá',
                'Goodbye': 'Tchau',
                'Thanks': 'Obrigado',
                'Sorry': 'Desculpe',
                'Help': 'Ajuda',
                'Please': 'Por favor',
                'Welcome': 'Bem-vindo',
                'Good luck': 'Boa sorte',
                'Take care': 'Se cuide',
                'See you': 'Até logo'
            }
        }

        # Busca tradução direta
        context_dict = translations.get(context_type, translations['ui'])
        if text in context_dict:
            return context_dict[text]

        # Busca tradução parcial
        for english, portuguese in context_dict.items():
            if english.lower() in text.lower():
                return text.replace(english, portuguese)

        # Fallback para tradução geral
        general_translations = {
            'Start': 'Iniciar',
            'Play': 'Jogar',
            'Stop': 'Parar',
            'Next': 'Próximo',
            'Previous': 'Anterior',
            'Up': 'Cima',
            'Down': 'Baixo',
            'Left': 'Esquerda',
            'Right': 'Direita',
            'Jump': 'Pular',
            'Run': 'Correr',
            'Attack': 'Atacar',
            'Defend': 'Defender',
            'Use': 'Usar',
            'Open': 'Abrir',
            'Close': 'Fechar',
            'Enter': 'Entrar',
            'Exit': 'Sair',
            'On': 'Ligado',
            'Off': 'Desligado',
            'Full Screen': 'Tela Cheia',
            'Windowed': 'Janela',
            'Volume': 'Volume',
            'Music': 'Música',
            'Sound': 'Som',
            'Graphics': 'Gráficos',
            'Controls': 'Controles',
            'Language': 'Idioma'
        }

        for english, portuguese in general_translations.items():
            if english.lower() in text.lower():
                return text.replace(english, portuguese)

        return f"[TRADUZIR: {text}]"

    def _post_process_switch_text(self, translated: str, entry: SwitchTextEntry) -> str:
        """Pós-processa texto traduzido para Switch"""

        # Aplica limite de caracteres se especificado
        if entry.character_limit and len(translated) > entry.character_limit:
            # Tenta abreviar primeiro
            abbreviated = self._abbreviate_text(translated)
            if len(abbreviated) <= entry.character_limit:
                return abbreviated

            # Se ainda for muito longo, trunca
            return translated[:entry.character_limit-3] + "..."

        return translated

    def _abbreviate_text(self, text: str) -> str:
        """Abrevia texto mantendo legibilidade"""
        abbreviations = {
            'Configurações': 'Config',
            'Opções': 'Opts',
            'Continuar': 'Cont',
            'Experiência': 'Exp',
            'Pontuação': 'Pts',
            'Dinheiro': '$',
            'Inventário': 'Inv',
            'Equipamento': 'Equip',
            'Habilidades': 'Hab',
            'Estatísticas': 'Stats'
        }

        for full, abbrev in abbreviations.items():
            text = text.replace(full, abbrev)

        return text

    def load_cache(self):
        """Carrega cache de traduções"""
        try:
            with open('switch_translation_cache.json', 'r', encoding='utf-8') as f:
                self.translation_cache = json.load(f)
        except FileNotFoundError:
            self.translation_cache = {}

    def save_cache(self):
        """Salva cache de traduções"""
        with open('switch_translation_cache.json', 'w', encoding='utf-8') as f:
            json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)

class SwitchROMPatcher:
    """Aplica patches de tradução em jogos Switch"""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.msbt_patcher = MSBTPatcher()

    def patch_game(self, extracted_path: Path, metadata: SwitchGameMetadata, output_path: Path) -> bool:
        """Aplica patches de tradução no jogo"""
        try:
            logger.info(f"Iniciando patch do jogo: {metadata.game_name}")

            # Cria diretório de saída
            patch_dir = self.temp_dir / "patched_game"
            if patch_dir.exists():
                shutil.rmtree(patch_dir)

            # Copia arquivos originais
            shutil.copytree(extracted_path, patch_dir)

            # Aplica patches por tipo de arquivo
            patched_files = 0

            # Agrupa entradas por arquivo
            files_to_patch = {}
            for entry in metadata.text_entries:
                if entry.is_translated:
                    if entry.file_path not in files_to_patch:
                        files_to_patch[entry.file_path] = []
                    files_to_patch[entry.file_path].append(entry)

            # Aplica patches em cada arquivo
            for file_path, entries in files_to_patch.items():
                try:
                    original_file = Path(file_path)
                    patched_file = patch_dir / original_file.name

                    # Encontra arquivo no diretório patch
                    for potential_file in patch_dir.rglob(original_file.name):
                        if self._patch_file(potential_file, entries):
                            patched_files += 1
                            break

                except Exception as e:
                    logger.error(f"Erro ao aplicar patch em {file_path}: {e}")
                    continue

            logger.info(f"Patched {patched_files} arquivos")

            # Reempacota o jogo
            return self._repack_game(patch_dir, metadata, output_path)

        except Exception as e:
            logger.error(f"Erro ao aplicar patches: {e}")
            return False

    def _patch_file(self, file_path: Path, entries: List[SwitchTextEntry]) -> bool:
        """Aplica patch em um arquivo específico"""
        try:
            # Determina tipo de arquivo
            file_format = self._detect_file_format(file_path)

            if file_format == TextFileFormat.MSBT:
                return self.msbt_patcher.patch_msbt(file_path, entries)
            elif file_format == TextFileFormat.JSON:
                return self._patch_json_file(file_path, entries)
            elif file_format == TextFileFormat.XML:
                return self._patch_xml_file(file_path, entries)
            elif file_format == TextFileFormat.TXT:
                return self._patch_txt_file(file_path, entries)
            else:
                logger.warning(f"Formato não suportado para patch: {file_format}")
                return False

        except Exception as e:
            logger.error(f"Erro ao aplicar patch em {file_path}: {e}")
            return False

    def _detect_file_format(self, file_path: Path) -> TextFileFormat:
        """Detecta formato do arquivo"""
        extension = file_path.suffix.lower()

        format_map = {
            '.msbt': TextFileFormat.MSBT,
            '.json': TextFileFormat.JSON,
            '.xml': TextFileFormat.XML,
            '.txt': TextFileFormat.TXT,
            '.yaml': TextFileFormat.YAML,
            '.yml': TextFileFormat.YAML,
            '.csv': TextFileFormat.CSV,
        }

        return format_map.get(extension, TextFileFormat.TXT)

    def _patch_json_file(self, file_path: Path, entries: List[SwitchTextEntry]) -> bool:
        """Aplica patch em arquivo JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Aplica traduções
            for entry in entries:
                if entry.is_translated:
                    self._apply_json_translation(data, entry)

            # Salva arquivo modificado
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            logger.error(f"Erro ao aplicar patch JSON: {e}")
            return False

    def _apply_json_translation(self, data: dict, entry: SwitchTextEntry):
        """Aplica tradução em estrutura JSON"""
        # Parse do caminho da entrada (ex: "menu.options.sound")
        path_parts = entry.entry_id.split('.')

        current = data
        for part in path_parts[:-1]:
            if '[' in part and ']' in part:
                # Array index
                key, index = part.split('[')
                index = int(index.rstrip(']'))
                current = current[key][index]
            else:
                current = current[part]

        # Aplica tradução
        final_key = path_parts[-1]
        if '[' in final_key and ']' in final_key:
            key, index = final_key.split('[')
            index = int(index.rstrip(']'))
            current[key][index] = entry.translated_text
        else:
            current[final_key] = entry.translated_text

    def _patch_xml_file(self, file_path: Path, entries: List[SwitchTextEntry]) -> bool:
        """Aplica patch em arquivo XML"""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Cria mapeamento de entradas
            entry_map = {entry.entry_id: entry for entry in entries if entry.is_translated}

            # Aplica traduções
            for elem in root.iter():
                elem_id = f"{elem.tag}_{elem.attrib.get('id', 'unknown')}"
                if elem_id in entry_map:
                    elem.text = entry_map[elem_id].translated_text

            # Salva arquivo modificado
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
            return True

        except Exception as e:
            logger.error(f"Erro ao aplicar patch XML: {e}")
            return False

    def _patch_txt_file(self, file_path: Path, entries: List[SwitchTextEntry]) -> bool:
        """Aplica patch em arquivo TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Cria mapeamento de entradas
            entry_map = {}
            for entry in entries:
                if entry.is_translated and entry.entry_id.startswith('line_'):
                    line_num = int(entry.entry_id.split('_')[1])
                    entry_map[line_num] = entry.translated_text

            # Aplica traduções
            for line_num, translated_text in entry_map.items():
                if line_num < len(lines):
                    lines[line_num] = translated_text + '\n'

            # Salva arquivo modificado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return True

        except Exception as e:
            logger.error(f"Erro ao aplicar patch TXT: {e}")
            return False

    def _repack_game(self, patched_dir: Path, metadata: SwitchGameMetadata, output_path: Path) -> bool:
        """Reempacota o jogo modificado"""
        try:
            if metadata.file_type == SwitchFileType.NSP:
                return self._repack_nsp(patched_dir, output_path)
            elif metadata.file_type == SwitchFileType.XCI:
                return self._repack_xci(patched_dir, output_path)
            else:
                logger.error(f"Tipo de arquivo não suportado: {metadata.file_type}")
                return False

        except Exception as e:
            logger.error(f"Erro ao reempacotar jogo: {e}")
            return False

    def _repack_nsp(self, patched_dir: Path, output_path: Path) -> bool:
        """Reempacota NSP"""
        try:
            # Coleta todos os arquivos
            files_to_pack = []
            for file_path in patched_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(patched_dir)
                    files_to_pack.append((file_path, str(relative_path)))

            # Cria header PFS0
            file_count = len(files_to_pack)

            # Calcula tamanho da string table
            string_table = b''
            string_offsets = []
            for _, filename in files_to_pack:
                string_offsets.append(len(string_table))
                string_table += filename.encode('utf-8') + b'\x00'

            # Alinhamento
            while len(string_table) % 16 != 0:
                string_table += b'\x00'

            string_table_size = len(string_table)

            # Calcula offsets dos arquivos
            current_offset = 0
            file_entries = []

            for i, (file_path, filename) in enumerate(files_to_pack):
                file_size = file_path.stat().st_size

                file_entries.append({
                    'offset': current_offset,
                    'size': file_size,
                    'name_offset': string_offsets[i]
                })

                # Alinhamento para próximo arquivo
                current_offset += file_size
                current_offset = (current_offset + 15) & ~15

            # Escreve NSP
            with open(output_path, 'wb') as f:
                # Header PFS0
                f.write(b'PFS0')
                f.write(struct.pack('<I', file_count))
                f.write(struct.pack('<I', string_table_size))
                f.write(b'\x00' * 4)  # Reserved

                # File entries
                for entry in file_entries:
                    f.write(struct.pack('<Q', entry['offset']))
                    f.write(struct.pack('<Q', entry['size']))
                    f.write(struct.pack('<I', entry['name_offset']))
                    f.write(b'\x00' * 4)  # Reserved

                # String table
                f.write(string_table)

                # File data
                for file_path, _ in files_to_pack:
                    with open(file_path, 'rb') as src:
                        shutil.copyfileobj(src, f)

                    # Alinhamento
                    pos = f.tell()
                    aligned_pos = (pos + 15) & ~15
                    f.write(b'\x00' * (aligned_pos - pos))

            logger.info(f"NSP reempacotado salvo em: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao reempacotar NSP: {e}")
            return False

    def _repack_xci(self, patched_dir: Path, output_path: Path) -> bool:
        """Reempacota XCI (implementação básica)"""
        logger.warning("Reempacotamento XCI ainda não implementado")
        return False

class MSBTPatcher:
    """Patcher especializado para arquivos MSBT"""

    def patch_msbt(self, file_path: Path, entries: List[SwitchTextEntry]) -> bool:
        """Aplica patch em arquivo MSBT"""
        try:
            # Lê arquivo original
            with open(file_path, 'rb') as f:
                data = bytearray(f.read())

            # Verifica se é MSBT válido
            if data[:8] != b'MsgStdBn':
                return False

            # Cria mapeamento de entradas
            entry_map = {}
            for entry in entries:
                if entry.is_translated and entry.entry_id.startswith('string_'):
                    index = int(entry.entry_id.split('_')[1])
                    entry_map[index] = entry.translated_text

            # Aplica patches
            return self._patch_msbt_strings(data, entry_map, file_path)

        except Exception as e:
            logger.error(f"Erro ao aplicar patch MSBT: {e}")
            return False

    def _patch_msbt_strings(self, data: bytearray, entry_map: Dict[int, str], file_path: Path) -> bool:
        """Aplica patches nas strings do MSBT"""
        try:
            # Parse do header
            header = struct.unpack('<8sHHHHHH', data[:20])
            section_count = header[6]

            offset = 20
            sections = []

            # Lê seções
            for i in range(section_count):
                section_header = struct.unpack('<4sI8s', data[offset:offset+16])
                section_type = section_header[0]
                section_size = section_header[1]

                sections.append({
                    'type': section_type,
                    'size': section_size,
                    'offset': offset + 16,
                    'data_offset': offset
                })

                offset += 16 + section_size
                offset = (offset + 15) & ~15

            # Encontra seção TXT2
            txt2_section = None
            for section in sections:
                if section['type'] == b'TXT2':
                    txt2_section = section
                    break

            if not txt2_section:
                logger.error("Seção TXT2 não encontrada")
                return False

            # Reconstrói seção TXT2 com traduções
            new_txt2_data = self._rebuild_txt2_section(
                data[txt2_section['offset']:txt2_section['offset'] + txt2_section['size']],
                entry_map
            )

            # Reconstrói arquivo completo
            new_data = self._rebuild_msbt_file(data, sections, txt2_section, new_txt2_data)

            # Salva arquivo modificado
            with open(file_path, 'wb') as f:
                f.write(new_data)

            return True

        except Exception as e:
            logger.error(f"Erro ao reconstruir MSBT: {e}")
            return False

    def _rebuild_txt2_section(self, section_data: bytes, entry_map: Dict[int, str]) -> bytes:
        """Reconstrói seção TXT2 com traduções"""

        # Lê número de strings
        string_count = struct.unpack('<I', section_data[:4])[0]

        # Lê offsets originais
        original_offsets = []
        for i in range(string_count):
            offset = struct.unpack('<I', section_data[4 + i*4:8 + i*4])[0]
            original_offsets.append(offset)

        # Constrói novas strings
        new_strings = []
        for i in range(string_count):
            if i in entry_map:
                # Usa tradução
                text = entry_map[i]
            else:
                # Extrai texto original
                start_offset = original_offsets[i]
                end_offset = original_offsets[i + 1] if i + 1 < len(original_offsets) else len(section_data)

                original_data = section_data[start_offset:end_offset].rstrip(b'\x00')
                text = original_data.decode('utf-16le', errors='ignore')

            # Codifica como UTF-16LE
            encoded = text.encode('utf-16le') + b'\x00\x00'
            new_strings.append(encoded)

        # Constrói nova seção
        new_section_data = bytearray()

        # Header com número de strings
        new_section_data.extend(struct.pack('<I', string_count))

        # Calcula novos offsets
        new_offsets = []
        current_offset = 4 + string_count * 4

        for string_data in new_strings:
            new_offsets.append(current_offset)
            current_offset += len(string_data)

        # Escreve offsets
        for offset in new_offsets:
            new_section_data.extend(struct.pack('<I', offset))

        # Escreve strings
        for string_data in new_strings:
            new_section_data.extend(string_data)

        # Alinhamento
        while len(new_section_data) % 16 != 0:
            new_section_data.append(0)

        return bytes(new_section_data)

    def _rebuild_msbt_file(self, original_data: bytearray, sections: List[dict],
                          txt2_section: dict, new_txt2_data: bytes) -> bytes:
        """Reconstrói arquivo MSBT completo"""

        new_data = bytearray()

        # Copia header original
        new_data.extend(original_data[:20])

        # Atualiza seções
        current_offset = 20

        for section in sections:
            if section['type'] == b'TXT2':
                # Usa nova seção TXT2
                section_data = new_txt2_data
            else:
                # Copia seção original
                section_data = original_data[section['offset']:section['offset'] + section['size']]

            # Escreve header da seção
            section_header = struct.pack('<4sI8s', section['type'], len(section_data), b'\x00' * 8)
            new_data.extend(section_header)

            # Escreve dados da seção
            new_data.extend(section_data)

            # Alinhamento
            while len(new_data) % 16 != 0:
                new_data.append(0)

        return bytes(new_data)

class SwitchTranslationEngine:
    """Engine principal para tradução de jogos Switch"""

    def __init__(self, source_lang: str = "ja", target_lang: str = "pt"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.temp_dir = Path(tempfile.mkdtemp(prefix='switch_translation_'))

        self.extractor = SwitchFileExtractor(self.temp_dir)
        self.translator = SwitchTranslationService(source_lang, target_lang)
        self.patcher = SwitchROMPatcher(self.temp_dir)

        self.stats = {
            'total_games': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'total_text_entries': 0,
            'translated_entries': 0,
            'total_text_files': 0
        }

    def translate_game(self, game_path: Path, output_path: Path) -> Optional[SwitchGameMetadata]:
        """Traduz um jogo Switch individual"""

        logger.info(f"Iniciando tradução de: {game_path.name}")

        try:
            # Extrai arquivos do jogo
            extracted_path = self.extractor.extract_game_files(game_path)
            if not extracted_path:
                logger.error(f"Falha ao extrair {game_path.name}")
                return None

            # Encontra arquivos de texto
            text_files = self.extractor.find_text_files(extracted_path)
            logger.info(f"Encontrados {len(text_files)} arquivos de texto")

            if not text_files:
                logger.warning(f"Nenhum arquivo de texto encontrado em {game_path.name}")
                return None

            # Extrai texto dos arquivos
            text_entries = self.extractor.extract_text_from_files(text_files)
            logger.info(f"Extraídas {len(text_entries)} entradas de texto")

            # Cria metadados
            metadata = SwitchGameMetadata(
                title_id=self._extract_title_id(game_path),
                game_name=game_path.stem,
                version="1.0.0",
                file_type=SwitchFileType.NSP if game_path.suffix.lower() == '.nsp' else SwitchFileType.XCI,
                file_size=game_path.stat().st_size,
                checksum=self._calculate_checksum(game_path),
                supported_languages=["ja", "en"],
                text_entries=text_entries,
                total_text_files=len(text_files),
                extraction_path=str(extracted_path)
            )

            # Traduz entradas de texto
            translated_count = 0
            for entry in text_entries:
                try:
                    entry.translated_text = self.translator.translate_entry(entry)
                    entry.is_translated = True
                    translated_count += 1

                    # Pequeno delay para evitar sobrecarga
                    time.sleep(0.05)

                except Exception as e:
                    logger.warning(f"Erro ao traduzir entrada {entry.entry_id}: {e}")
                    continue

            logger.info(f"Traduzidas {translated_count} de {len(text_entries)} entradas")

            # Aplica patches
            if self.patcher.patch_game(extracted_path, metadata, output_path):
                logger.info(f"Jogo traduzido salvo em: {output_path}")
                self.stats['successful_translations'] += 1
            else:
                logger.error(f"Falha ao aplicar patches em {game_path.name}")
                self.stats['failed_translations'] += 1

            # Atualiza estatísticas
            self.stats['total_text_entries'] += len(text_entries)
            self.stats['translated_entries'] += translated_count
            self.stats['total_text_files'] += len(text_files)

            return metadata

        except Exception as e:
            logger.error(f"Erro ao traduzir {game_path.name}: {e}")
            self.stats['failed_translations'] += 1
            return None

    def translate_game_directory(self, input_dir: Path, output_dir: Path):
        """Traduz todos os jogos de um diretório"""

        # Encontra jogos Switch
        game_files = []
        for ext in ['.nsp', '.xci']:
            game_files.extend(input_dir.glob(f"*{ext}"))

        logger.info(f"Encontrados {len(game_files)} jogos Switch para traduzir")
        self.stats['total_games'] = len(game_files)

        # Cria diretório de saída
        output_dir.mkdir(parents=True, exist_ok=True)

        # Traduz cada jogo
        translated_games = []
        for game_file in game_files:
            try:
                output_file = output_dir / f"{game_file.stem}_PT{game_file.suffix}"
                metadata = self.translate_game(game_file, output_file)

                if metadata:
                    translated_games.append(metadata)
                    # Salva metadados
                    metadata_file = output_dir / f"{game_file.stem}_metadata.json"
                    self._save_metadata(metadata, metadata_file)

            except KeyboardInterrupt:
                logger.info("Tradução interrompida pelo usuário")
                break
            except Exception as e:
                logger.error(f"Erro fatal ao traduzir {game_file.name}: {e}")
                continue

        # Salva relatório
        self._save_translation_report(output_dir, translated_games)

    def _extract_title_id(self, game_path: Path) -> str:
        """Extrai Title ID do jogo (implementação básica)"""
        # Implementação simplificada - normalmente seria extraído do metadata
        return game_path.stem[:16] if len(game_path.stem) >= 16 else "unknown"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcula checksum SHA256 do arquivo"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _save_metadata(self, metadata: SwitchGameMetadata, path: Path):
        """Salva metadados em arquivo JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2, default=str)

    def _save_translation_report(self, output_dir: Path, translated_games: List[SwitchGameMetadata]):
        """Salva relatório de tradução"""
        report_path = output_dir / "switch_translation_report.json"

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'console': 'Nintendo Switch',
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'statistics': self.stats,
            'translated_games': [
                {
                    'game_name': game.game_name,
                    'title_id': game.title_id,
                    'file_type': game.file_type.value,
                    'text_entries': len(game.text_entries),
                    'text_files': game.total_text_files
                }
                for game in translated_games
            ]
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Relatório salvo em: {report_path}")
        self._print_summary()

    def _print_summary(self):
        """Exibe resumo das estatísticas"""
        print("\n" + "="*70)
        print("RESUMO DA TRADUÇÃO - NINTENDO SWITCH")
        print("="*70)
        print(f"Total de jogos processados: {self.stats['total_games']}")
        print(f"Traduções bem-sucedidas: {self.stats['successful_translations']}")
        print(f"Traduções falharam: {self.stats['failed_translations']}")
        print(f"Total de arquivos de texto: {self.stats['total_text_files']}")
        print(f"Total de entradas de texto: {self.stats['total_text_entries']}")
        print(f"Entradas traduzidas: {self.stats['translated_entries']}")

        if self.stats['total_text_entries'] > 0:
            success_rate = (self.stats['translated_entries'] / self.stats['total_text_entries']) * 100
            print(f"Taxa de sucesso: {success_rate:.1f}%")

        print("="*70)

    def cleanup(self):
        """Limpa arquivos temporários"""
        self.extractor.cleanup()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

def main():
    """Função principal - exemplo de uso"""

    # Configurações
    input_directory = Path("switch_games")
    output_directory = Path("switch_games_traduzidos")

    # Cria engine
    engine = SwitchTranslationEngine(
        source_lang="ja",
        target_lang="pt"
    )

    try:
        # Processa jogos
        if input_directory.exists():
            engine.translate_game_directory(input_directory, output_directory)
        else:
            print(f"Diretório {input_directory} não encontrado!")
            print("Crie o diretório e coloque seus jogos Switch (.nsp/.xci) lá.")

            # Exemplo de uso individual
            print("\nExemplo de uso para jogo individual:")
            print("engine.translate_game(Path('meu_jogo.nsp'), Path('meu_jogo_PT.nsp'))")

    finally:
        # Limpa arquivos temporários
        engine.cleanup()

if __name__ == "__main__":
    main()