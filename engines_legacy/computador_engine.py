#!/usr/bin/env python3
"""
Computador Engine - Sistema de Tradução de Jogos
Automatiza a extração, tradução e injeção de textos em jogos retro e modernos

Desenvolvido para ROM hacking e localização de jogos
"""

import os
import re
import json
import struct
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import hashlib

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GamePlatform(Enum):
    """Plataformas de jogos suportadas"""
    NES = "nes"
    SNES = "snes"
    GB = "gameboy"
    GBA = "gameboy_advance"
    N64 = "nintendo64"
    PSX = "playstation"
    PS2 = "playstation2"
    PC = "pc"
    ANDROID = "android"
    IOS = "ios"
    # Engines populares para indies
    UNITY = "unity"
    GODOT = "godot"
    GAMEMAKER = "gamemaker"
    CONSTRUCT = "construct"
    RENPY = "renpy"
    RPG_MAKER = "rpgmaker"
    LÖVE = "love2d"
    DEFOLD = "defold"

class TextEncoding(Enum):
    """Codificações de texto suportadas"""
    ASCII = "ascii"
    UTF8 = "utf-8"
    UTF16 = "utf-16"
    SHIFT_JIS = "shift_jis"
    EUC_JP = "euc-jp"
    ISO_8859_1 = "iso-8859-1"
    CUSTOM = "custom"

@dataclass
class TextBlock:
    """Representa um bloco de texto extraído do jogo"""
    id: str
    original_text: str
    translated_text: str = ""
    context: str = ""
    offset: int = 0
    length: int = 0
    encoding: TextEncoding = TextEncoding.UTF8
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.length:
            self.length = len(self.original_text)

class ComputadorEngine:
    """
    Engine principal para tradução de jogos

    Like a Swiss Army knife for game translation - but sharper and more precise!
    """

    def __init__(self, game_path: str, platform: GamePlatform):
        self.game_path = Path(game_path)
        self.platform = platform
        self.text_blocks: List[TextBlock] = []
        self.config = self._load_config()
        self.translation_cache = {}

        # Patterns comuns para diferentes plataformas
        self.text_patterns = {
            GamePlatform.NES: [
                rb'[\x20-\x7E]{4,}',  # ASCII básico
                rb'[\x80-\xFF]{2,}',  # Caracteres customizados
            ],
            GamePlatform.SNES: [
                rb'[\x20-\x7E]{4,}',
                rb'[\x80-\xFF]{2,}',
            ],
            GamePlatform.PSX: [
                rb'[\x20-\x7E]{4,}',
                rb'\x00[\x20-\x7E]{3,}\x00',  # Strings null-terminated
            ],
            GamePlatform.PC: [
                rb'[\x20-\x7E]{4,}',
                rb'[\x00-\xFF]{2,}?\x00',  # Strings com terminador
            ],
            # Patterns específicos para engines de indies
            GamePlatform.UNITY: [
                rb'[\x20-\x7E]{4,}',
                rb'\x00[\x20-\x7E]{3,}\x00',
                rb'[\x20-\x7E]{4,}\x00',  # Strings C-style
                rb'[\x01-\x04][\x20-\x7E]{3,}',  # Prefixed strings
            ],
            GamePlatform.GODOT: [
                rb'[\x20-\x7E]{4,}',
                rb'\x04[\x20-\x7E]{3,}',  # Godot string format
                rb'[\x20-\x7E]{4,}\x00',
            ],
            GamePlatform.GAMEMAKER: [
                rb'[\x20-\x7E]{4,}',
                rb'\x00[\x20-\x7E]{3,}\x00',
                rb'[\x20-\x7E]{4,}\x00',
            ],
            GamePlatform.RENPY: [
                rb'[\x20-\x7E]{4,}',
                rb'[\x20-\x7E]{4,}\x0A',  # Texto com quebras de linha
                rb'[\x20-\x7E]{4,}\x00',
            ],
            GamePlatform.RPG_MAKER: [
                rb'[\x20-\x7E]{4,}',
                rb'\x00[\x20-\x7E]{3,}\x00',
                rb'[\x81-\x9F\xE0-\xEF][\x40-\x7E\x80-\xFC]{2,}',  # Shift-JIS
            ],
            GamePlatform.LÖVE: [
                rb'[\x20-\x7E]{4,}',
                rb'[\x20-\x7E]{4,}\x00',
                rb'[\x20-\x7E]{4,}\x0A',
            ]
        }

        logger.info(f"Computador Engine inicializado para {platform.value}")

    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações específicas da plataforma"""
        config_path = Path("configs") / f"{self.platform.value}.json"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Configuração padrão
        return {
            "max_text_length": 256,
            "preserve_formatting": True,
            "auto_wrap": True,
            "font_width": 8,
            "line_height": 16,
            "encoding": "utf-8"
        }

    def scan_for_text(self, min_length: int = 4) -> List[TextBlock]:
        """
        Escaneia o arquivo do jogo procurando por texto

        Como um metal detector, mas para strings perdidas no código!
        """
        if not self.game_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.game_path}")

        logger.info(f"Escaneando {self.game_path.name} em busca de texto...")

        with open(self.game_path, 'rb') as f:
            data = f.read()

        text_blocks = []
        patterns = self.text_patterns.get(self.platform, [rb'[\x20-\x7E]{4,}'])

        for pattern in patterns:
            matches = re.finditer(pattern, data)

            for match in matches:
                try:
                    # Tentativa de decodificação
                    raw_text = match.group(0)

                    # Remove null bytes para análise
                    clean_text = raw_text.replace(b'\x00', b'')

                    if len(clean_text) < min_length:
                        continue

                    # Tenta diferentes encodings
                    decoded_text = self._decode_text(clean_text)

                    if decoded_text and self._is_likely_text(decoded_text):
                        block = TextBlock(
                            id=f"text_{match.start():08x}",
                            original_text=decoded_text,
                            offset=match.start(),
                            length=len(raw_text),
                            metadata={
                                "pattern": pattern.decode('unicode_escape'),
                                "raw_bytes": raw_text.hex()
                            }
                        )
                        text_blocks.append(block)

                except (UnicodeDecodeError, ValueError):
                    continue

        # Remove duplicatas
        unique_blocks = []
        seen_texts = set()

        for block in text_blocks:
            if block.original_text not in seen_texts:
                unique_blocks.append(block)
                seen_texts.add(block.original_text)

        self.text_blocks = unique_blocks
        logger.info(f"Encontrados {len(unique_blocks)} blocos de texto únicos")

        return unique_blocks

    def _decode_text(self, raw_bytes: bytes) -> Optional[str]:
        """Tenta decodificar texto usando diferentes encodings"""
        encodings = [
            TextEncoding.UTF8,
            TextEncoding.ASCII,
            TextEncoding.SHIFT_JIS,
            TextEncoding.ISO_8859_1,
        ]

        for encoding in encodings:
            try:
                return raw_bytes.decode(encoding.value)
            except UnicodeDecodeError:
                continue

        return None

    def _is_likely_text(self, text: str) -> bool:
        """Verifica se uma string parece ser texto legível"""
        if not text:
            return False

        # Critérios básicos para texto válido
        printable_chars = sum(1 for c in text if c.isprintable())
        ratio = printable_chars / len(text)

        # Pelo menos 70% dos caracteres devem ser imprimíveis
        if ratio < 0.7:
            return False

        # Evita sequências repetitivas
        if len(set(text)) < 3:
            return False

        # Verifica se parece com texto real (não apenas símbolos)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)

        return alpha_ratio > 0.3

    def extract_text_advanced(self, custom_patterns: List[str] = None) -> List[TextBlock]:
        """
        Extração avançada com patterns customizados

        Para quando você precisa de precisão cirúrgica na extração!
        """
        if custom_patterns:
            additional_patterns = [pattern.encode() for pattern in custom_patterns]
            self.text_patterns[self.platform].extend(additional_patterns)

        # Técnicas específicas por plataforma
        if self.platform == GamePlatform.PSX:
            return self._extract_psx_text()
        elif self.platform == GamePlatform.PC:
            return self._extract_pc_text()
        else:
            return self.scan_for_text()

    def _extract_psx_text(self) -> List[TextBlock]:
        """Extração especializada para jogos de PlayStation"""
        logger.info("Usando extração especializada para PSX")

        with open(self.game_path, 'rb') as f:
            data = f.read()

        text_blocks = []

        # Procura por tabelas de strings
        for i in range(0, len(data) - 4, 4):
            # Verifica se parece com um ponteiro
            ptr = struct.unpack('<I', data[i:i+4])[0]

            if 0x80000000 <= ptr <= 0x801FFFFF:  # Faixa típica de RAM do PSX
                # Calcula offset relativo
                offset = ptr - 0x80000000

                if offset < len(data):
                    # Extrai string na posição
                    string_data = self._extract_null_terminated_string(data, offset)

                    if string_data and len(string_data) > 3:
                        decoded = self._decode_text(string_data)

                        if decoded and self._is_likely_text(decoded):
                            block = TextBlock(
                                id=f"psx_string_{offset:08x}",
                                original_text=decoded,
                                offset=offset,
                                length=len(string_data),
                                metadata={
                                    "pointer_location": i,
                                    "pointer_value": f"0x{ptr:08x}"
                                }
                            )
                            text_blocks.append(block)

        self.text_blocks = text_blocks
        return text_blocks

    def _extract_pc_text(self) -> List[TextBlock]:
        """Extração para jogos de PC modernos"""
        logger.info("Usando extração para PC")

        # Verifica se é um executável
        if self.game_path.suffix.lower() in ['.exe', '.dll']:
            return self._extract_pe_strings()
        else:
            return self.scan_for_text()

    def _extract_pe_strings(self) -> List[TextBlock]:
        """Extrai strings de executáveis PE (Windows)"""
        with open(self.game_path, 'rb') as f:
            data = f.read()

        text_blocks = []

        # Procura por strings Unicode e ASCII
        unicode_pattern = rb'(?:[\x20-\x7E]\x00){4,}'
        ascii_pattern = rb'[\x20-\x7E]{4,}'

        for pattern in [unicode_pattern, ascii_pattern]:
            matches = re.finditer(pattern, data)

            for match in matches:
                raw_text = match.group(0)

                if pattern == unicode_pattern:
                    # Remove null bytes para Unicode
                    clean_text = raw_text.replace(b'\x00', b'')
                    encoding = TextEncoding.UTF16
                else:
                    clean_text = raw_text
                    encoding = TextEncoding.ASCII

                try:
                    decoded_text = clean_text.decode(encoding.value)

                    if self._is_likely_text(decoded_text):
                        block = TextBlock(
                            id=f"pe_string_{match.start():08x}",
                            original_text=decoded_text,
                            offset=match.start(),
                            length=len(raw_text),
                            encoding=encoding,
                            metadata={"type": "pe_string"}
                        )
                        text_blocks.append(block)

                except UnicodeDecodeError:
                    continue

        self.text_blocks = text_blocks
        return text_blocks

    def _extract_null_terminated_string(self, data: bytes, offset: int) -> bytes:
        """Extrai string terminada em null de uma posição específica"""
        end_pos = offset

        while end_pos < len(data) and data[end_pos] != 0:
            end_pos += 1

        return data[offset:end_pos]

    def translate_text(self, text: str, target_language: str = "pt-BR") -> str:
        """
        Traduz texto usando cache inteligente

        Aqui você pode integrar com APIs de tradução como Google Translate,
        OpenAI, ou sistemas offline como MarianMT
        """

        # Verifica cache primeiro
        cache_key = hashlib.md5(f"{text}_{target_language}".encode()).hexdigest()

        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        # Placeholder para integração com API de tradução
        # TODO: Implementar integração com serviço de tradução
        translated = f"[{target_language}] {text}"

        # Salva no cache
        self.translation_cache[cache_key] = translated

        return translated

    def auto_translate_blocks(self, target_language: str = "pt-BR") -> None:
        """Traduz automaticamente todos os blocos de texto"""
        logger.info(f"Iniciando tradução automática para {target_language}")

        for block in self.text_blocks:
            if not block.translated_text:
                block.translated_text = self.translate_text(block.original_text, target_language)

        logger.info(f"Tradução concluída para {len(self.text_blocks)} blocos")

    def export_translation_table(self, output_path: str = "translation_table.json") -> None:
        """Exporta tabela de tradução para edição manual"""
        export_data = {
            "game_info": {
                "name": self.game_path.name,
                "platform": self.platform.value,
                "hash": self._calculate_file_hash()
            },
            "text_blocks": [asdict(block) for block in self.text_blocks]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Tabela de tradução exportada: {output_path}")

    def import_translation_table(self, table_path: str) -> None:
        """Importa tabela de tradução editada"""
        with open(table_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstrói blocos de texto
        self.text_blocks = []

        for block_data in data["text_blocks"]:
            # Converte encoding string de volta para enum
            if "encoding" in block_data:
                try:
                    block_data["encoding"] = TextEncoding(block_data["encoding"])
                except ValueError:
                    block_data["encoding"] = TextEncoding.UTF8

            block = TextBlock(**block_data)
            self.text_blocks.append(block)

        logger.info(f"Tabela de tradução importada: {len(self.text_blocks)} blocos")

    def _calculate_file_hash(self) -> str:
        """Calcula hash SHA-256 do arquivo do jogo"""
        with open(self.game_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def patch_game(self, output_path: str = None) -> str:
        """
        Aplica traduções ao jogo original

        O momento da verdade - quando sua tradução ganha vida!
        """
        if not output_path:
            output_path = str(self.game_path.parent / f"{self.game_path.stem}_translated{self.game_path.suffix}")

        logger.info(f"Aplicando patch de tradução...")

        # Cria cópia do arquivo original
        with open(self.game_path, 'rb') as f:
            game_data = bytearray(f.read())

        # Aplica traduções
        patches_applied = 0

        for block in self.text_blocks:
            if block.translated_text and block.offset > 0:
                try:
                    # Codifica texto traduzido
                    encoded_translation = block.translated_text.encode(block.encoding.value)

                    # Verifica se cabe no espaço original
                    if len(encoded_translation) <= block.length:
                        # Aplica patch
                        game_data[block.offset:block.offset + len(encoded_translation)] = encoded_translation

                        # Preenche resto com zeros se necessário
                        remaining = block.length - len(encoded_translation)
                        if remaining > 0:
                            game_data[block.offset + len(encoded_translation):block.offset + block.length] = b'\x00' * remaining

                        patches_applied += 1
                    else:
                        logger.warning(f"Tradução muito longa para o bloco {block.id}")

                except Exception as e:
                    logger.error(f"Erro ao aplicar patch no bloco {block.id}: {e}")

        # Salva arquivo patcheado
        with open(output_path, 'wb') as f:
            f.write(game_data)

        logger.info(f"Patch aplicado com sucesso! {patches_applied} blocos traduzidos")
        logger.info(f"Arquivo traduzido salvo em: {output_path}")

        return output_path

    def generate_report(self) -> Dict[str, Any]:
        """Gera relatório detalhado da tradução"""
        total_blocks = len(self.text_blocks)
        translated_blocks = sum(1 for block in self.text_blocks if block.translated_text)

        total_chars = sum(len(block.original_text) for block in self.text_blocks)
        translated_chars = sum(len(block.translated_text) for block in self.text_blocks if block.translated_text)

        report = {
            "game_info": {
                "file": self.game_path.name,
                "platform": self.platform.value,
                "size": self.game_path.stat().st_size,
                "hash": self._calculate_file_hash()
            },
            "translation_stats": {
                "total_text_blocks": total_blocks,
                "translated_blocks": translated_blocks,
                "translation_progress": f"{(translated_blocks/total_blocks)*100:.1f}%" if total_blocks > 0 else "0%",
                "total_characters": total_chars,
                "translated_characters": translated_chars,
                "character_progress": f"{(translated_chars/total_chars)*100:.1f}%" if total_chars > 0 else "0%"
            },
            "top_text_blocks": [
                {
                    "id": block.id,
                    "original": block.original_text[:50] + "..." if len(block.original_text) > 50 else block.original_text,
                    "translated": block.translated_text[:50] + "..." if len(block.translated_text) > 50 else block.translated_text,
                    "length": len(block.original_text)
                }
                for block in sorted(self.text_blocks, key=lambda x: len(x.original_text), reverse=True)[:10]
            ]
        }

        return report

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo para diferentes plataformas

    # Para jogos de SNES
    # engine = ComputadorEngine("game.smc", GamePlatform.SNES)

    # Para jogos de PC
    # engine = ComputadorEngine("game.exe", GamePlatform.PC)

    # Para jogos de PlayStation
    # engine = ComputadorEngine("game.bin", GamePlatform.PSX)

    print("🎮 Computador Engine - Sistema de Tradução de Jogos")
    print("=" * 50)
    print("Exemplo de uso:")
    print()
    print("# Inicializar engine")
    print("engine = ComputadorEngine('meu_jogo.smc', GamePlatform.SNES)")
    print()
    print("# Escanear texto")
    print("text_blocks = engine.scan_for_text()")
    print()
    print("# Traduzir automaticamente")
    print("engine.auto_translate_blocks('pt-BR')")
    print()
    print("# Exportar para edição")
    print("engine.export_translation_table('traducao.json')")
    print()
    print("# Aplicar patch")
    print("engine.patch_game('jogo_traduzido.smc')")
    print()
    print("Pronto para hackear alguns jogos! 🚀")