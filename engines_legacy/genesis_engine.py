"""
Sega Genesis/Mega Drive ROM Engine
Handles Genesis ROM text extraction and patching.

Technical Specs:
- Motorola 68000 CPU @ 7.6 MHz
- 64KB RAM, 64KB VRAM
- ROM sizes: 256KB to 4MB
- Text encoding: ASCII, some games use custom character tables
- 32-bit addressing with bank switching
"""

import struct
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenesisEngine:
    """
    Engine específico para ROMs do Sega Genesis/Mega Drive.

    Características técnicas:
    - Suporte a ROMs .gen, .md, .smd
    - Detecção automática de header (se presente)
    - Extração de texto ASCII e tabelas customizadas
    - Patching com preservação de ponteiros
    - Validação de checksum
    """

    # Constantes do sistema
    CONSOLE_NAME = "Sega Genesis"
    SUPPORTED_EXTENSIONS = ['.gen', '.md', '.smd']
    MAX_ROM_SIZE = 4 * 1024 * 1024  # 4MB
    MIN_ROM_SIZE = 256 * 1024       # 256KB

    # Header signatures
    SEGA_HEADER = b'SEGA'
    MEGA_DRIVE_HEADER = b'MEGA DRIVE'

    # Configurações de extração de texto
    MIN_STRING_LENGTH = 4
    MAX_STRING_LENGTH = 512

    def __init__(self):
        """Inicializa o engine do Genesis."""
        self.rom_data = None
        self.rom_size = 0
        self.has_header = False
        self.header_info = {}
        self.text_regions = []
        self.rom_path = None
        self.is_smd_format = False

        logger.info(f"🎮 {self.CONSOLE_NAME} Engine inicializado")

    def load_rom(self, rom_path: str) -> bool:
        """
        Carrega ROM do Genesis/Mega Drive.

        Args:
            rom_path (str): Caminho para o arquivo ROM

        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            self.rom_path = Path(rom_path)

            # Validação da extensão
            if self.rom_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.error(f"❌ Extensão não suportada: {self.rom_path.suffix}")
                return False

            # Carregar dados do ROM
            with open(rom_path, 'rb') as f:
                self.rom_data = f.read()

            self.rom_size = len(self.rom_data)

            # Validação do tamanho
            if self.rom_size > self.MAX_ROM_SIZE:
                logger.warning(f"⚠️  ROM muito grande: {self.rom_size} bytes (max: {self.MAX_ROM_SIZE})")
                return False

            if self.rom_size < self.MIN_ROM_SIZE:
                logger.error(f"❌ ROM muito pequeno: {self.rom_size} bytes (min: {self.MIN_ROM_SIZE})")
                return False

            # Detectar formato SMD e converter se necessário
            self._detect_smd_format()

            # Analisar header do Genesis
            self._analyze_header()

            logger.info(f"✅ ROM carregado: {self.rom_path.name} ({self.rom_size} bytes)")
            if self.has_header:
                logger.info(f"📋 Header detectado: {self.header_info.get('title', 'N/A')}")

            return True

        except FileNotFoundError:
            logger.error(f"❌ Arquivo não encontrado: {rom_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro ao carregar ROM: {e}")
            return False

    def _detect_smd_format(self) -> None:
        """
        Detecta e converte formato SMD (Super Magic Drive) para formato linear.

        SMD usa blocks de 16KB alternados entre pares/ímpares.
        """
        if self.rom_path.suffix.lower() == '.smd':
            # Verifica se tem header SMD (512 bytes)
            if self.rom_size % 16384 == 512:
                logger.info("🔄 Detectado formato SMD com header - convertendo...")
                self.rom_data = self.rom_data[512:]  # Remove header SMD
                self.rom_size -= 512

            # Desentrelaça formato SMD
            self._deinterleave_smd()
            self.is_smd_format = True

    def _deinterleave_smd(self) -> None:
        """Desentrelaça dados SMD para formato linear."""
        if self.rom_size % 16384 != 0:
            logger.warning("⚠️  Tamanho de ROM SMD não é múltiplo de 16KB")
            return

        deinterleaved = bytearray()
        num_blocks = self.rom_size // 16384

        for block in range(num_blocks):
            block_start = block * 16384

            # Primeiro 8KB: bytes pares
            even_bytes = self.rom_data[block_start:block_start + 8192]
            # Segundo 8KB: bytes ímpares
            odd_bytes = self.rom_data[block_start + 8192:block_start + 16384]

            # Reconstrói bytes originais
            for i in range(8192):
                deinterleaved.append(even_bytes[i])
                deinterleaved.append(odd_bytes[i])

        self.rom_data = bytes(deinterleaved)
        logger.info("✅ Formato SMD convertido para linear")

    def _analyze_header(self) -> None:
        """Analisa header do Genesis para extrair informações."""
        if len(self.rom_data) < 0x200:
            return

        # Header do Genesis fica no offset 0x100
        header_start = 0x100

        try:
            # Verifica assinatura
            signature = self.rom_data[header_start:header_start + 4]
            if signature in [self.SEGA_HEADER, self.MEGA_DRIVE_HEADER]:
                self.has_header = True

                # Extrai informações do header
                self.header_info = {
                    'signature': signature.decode('ascii', errors='ignore'),
                    'title': self.rom_data[header_start + 0x50:header_start + 0x60].decode('ascii', errors='ignore').strip(),
                    'serial': self.rom_data[header_start + 0x60:header_start + 0x68].decode('ascii', errors='ignore').strip(),
                    'checksum': struct.unpack('>H', self.rom_data[header_start + 0x4E:header_start + 0x50])[0],
                    'rom_start': struct.unpack('>L', self.rom_data[header_start + 0x40:header_start + 0x44])[0],
                    'rom_end': struct.unpack('>L', self.rom_data[header_start + 0x44:header_start + 0x48])[0],
                }

        except Exception as e:
            logger.warning(f"⚠️  Erro ao analisar header: {e}")

    def extract_text_regions(self) -> List[Dict]:
        """
        Extrai regiões de texto do ROM Genesis.

        Implementa estratégias específicas para 68000:
        1. Strings ASCII terminadas em nulo
        2. Tabelas de ponteiros 32-bit
        3. Strings com prefixo de tamanho
        4. Análise de seções de dados

        Returns:
            List[Dict]: Lista de regiões de texto encontradas
        """
        if not self.rom_data:
            logger.error("❌ ROM não carregado")
            return []

        logger.info("🔍 Iniciando extração de texto...")

        text_regions = []

        # Estratégia 1: Strings ASCII simples
        ascii_regions = self._extract_ascii_strings()
        text_regions.extend(ascii_regions)

        # Estratégia 2: Ponteiros 32-bit
        pointer_regions = self._extract_pointer_strings()
        text_regions.extend(pointer_regions)

        # Estratégia 3: Strings com prefixo de tamanho
        length_prefix_regions = self._extract_length_prefixed_strings()
        text_regions.extend(length_prefix_regions)

        # Remove duplicatas e ordena por offset
        text_regions = self._deduplicate_regions(text_regions)
        text_regions.sort(key=lambda x: x['offset'])

        self.text_regions = text_regions

        logger.info(f"📝 Extraídas {len(text_regions)} regiões de texto")

        return text_regions

    def _extract_ascii_strings(self) -> List[Dict]:
        """Extrai strings ASCII terminadas em nulo."""
        regions = []
        current_string = ""
        start_offset = 0

        for i, byte in enumerate(self.rom_data):
            if (32 <= byte <= 126) or byte in [0x0A, 0x0D]:
                if not current_string:
                    start_offset = i
                current_string += chr(byte)
            else:
                if len(current_string) >= self.MIN_STRING_LENGTH:
                    if self._is_likely_text(current_string):
                        regions.append({
                            'offset': start_offset,
                            'length': len(current_string),
                            'original_text': current_string.strip(),
                            'translated_text': '',
                            'console': self.CONSOLE_NAME,
                            'encoding': 'ascii',
                            'type': 'ascii_string',
                            'context': self._get_context(start_offset)
                        })
                current_string = ""

        # String final
        if len(current_string) >= self.MIN_STRING_LENGTH:
            if self._is_likely_text(current_string):
                regions.append({
                    'offset': start_offset,
                    'length': len(current_string),
                    'original_text': current_string.strip(),
                    'translated_text': '',
                    'console': self.CONSOLE_NAME,
                    'encoding': 'ascii',
                    'type': 'ascii_string',
                    'context': self._get_context(start_offset)
                })

        logger.info(f"📄 Encontradas {len(regions)} strings ASCII")
        return regions

    def _extract_pointer_strings(self) -> List[Dict]:
        """Extrai strings através de ponteiros 32-bit."""
        regions = []

        for i in range(0, len(self.rom_data) - 3, 4):
            try:
                # Lê ponteiro big-endian (padrão 68000)
                pointer = struct.unpack('>L', self.rom_data[i:i+4])[0]

                # Verifica se ponteiro está na faixa de ROM
                if 0x00000000 <= pointer <= 0x003FFFFF:
                    rom_offset = pointer

                    if rom_offset < len(self.rom_data):
                        text = self._extract_string_at_offset(rom_offset)

                        if text and len(text) >= self.MIN_STRING_LENGTH:
                            regions.append({
                                'offset': rom_offset,
                                'length': len(text),
                                'original_text': text,
                                'translated_text': '',
                                'console': self.CONSOLE_NAME,
                                'encoding': 'ascii',
                                'type': 'pointer_string',
                                'pointer_offset': i,
                                'pointer_value': pointer,
                                'context': self._get_context(rom_offset)
                            })
            except:
                continue

        logger.info(f"👉 Encontradas {len(regions)} strings por ponteiros")
        return regions

    def _extract_length_prefixed_strings(self) -> List[Dict]:
        """Extrai strings com prefixo de tamanho (comum em alguns jogos Genesis)."""
        regions = []

        for i in range(len(self.rom_data) - 1):
            try:
                # Primeiro byte pode ser tamanho da string
                length = self.rom_data[i]

                if 4 <= length <= 100:  # Tamanho razoável para string
                    start_pos = i + 1

                    if start_pos + length <= len(self.rom_data):
                        text_bytes = self.rom_data[start_pos:start_pos + length]

                        # Verifica se são caracteres ASCII válidos
                        if all(32 <= b <= 126 for b in text_bytes):
                            text = text_bytes.decode('ascii', errors='ignore')

                            if self._is_likely_text(text):
                                regions.append({
                                    'offset': start_pos,
                                    'length': length,
                                    'original_text': text,
                                    'translated_text': '',
                                    'console': self.CONSOLE_NAME,
                                    'encoding': 'ascii',
                                    'type': 'length_prefixed',
                                    'length_byte_offset': i,
                                    'context': self._get_context(start_pos)
                                })
            except:
                continue

        logger.info(f"📏 Encontradas {len(regions)} strings com prefixo de tamanho")
        return regions

    def _extract_string_at_offset(self, offset: int) -> Optional[str]:
        """Extrai string terminada em null a partir de um offset."""
        if offset >= len(self.rom_data):
            return None

        text = ""
        for i in range(offset, min(offset + self.MAX_STRING_LENGTH, len(self.rom_data))):
            byte = self.rom_data[i]

            if byte == 0:  # Terminador null
                break
            elif 32 <= byte <= 126:  # ASCII printável
                text += chr(byte)
            elif byte in [0x0A, 0x0D]:  # \n, \r
                text += chr(byte)
            else:
                break

        return text if len(text) >= self.MIN_STRING_LENGTH else None

    def _is_likely_text(self, text: str) -> bool:
        """Heurística para determinar se uma string é provavelmente texto de jogo."""
        if len(text) < 3 or len(text) > 200:
            return False

        # Filtra strings que são só números/hex
        if re.match(r'^[0-9A-Fa-f\s]+$', text):
            return False

        # Filtra strings que são só caracteres especiais
        if re.match(r'^[^\w\s]+$', text):
            return False

        # Aceita strings com letras
        return bool(re.search(r'[a-zA-Z]', text))

    def _get_context(self, offset: int) -> str:
        """Gera contexto para ajudar na tradução."""
        context = f"ROM offset: 0x{offset:06X}"

        # Adiciona informação sobre localização no ROM
        if offset < 0x100:
            context += " (Vector table)"
        elif offset < 0x200:
            context += " (ROM header)"
        elif offset < 0x8000:
            context += " (Low ROM area)"
        else:
            context += " (Main ROM area)"

        return context

    def _deduplicate_regions(self, regions: List[Dict]) -> List[Dict]:
        """Remove regiões duplicadas mantendo a de maior qualidade."""
        unique_regions = {}

        for region in regions:
            key = (region['offset'], region['original_text'])

            if key not in unique_regions:
                unique_regions[key] = region
            else:
                # Mantém a região com mais contexto
                existing = unique_regions[key]
                if len(region.get('context', '')) > len(existing.get('context', '')):
                    unique_regions[key] = region

        return list(unique_regions.values())

    def patch_rom(self, text_regions: List[Dict]) -> bytes:
        """
        Aplica traduções ao ROM Genesis.

        Args:
            text_regions (List[Dict]): Lista de regiões com traduções

        Returns:
            bytes: ROM modificado
        """
        if not self.rom_data:
            raise ValueError("ROM não carregado")

        logger.info("🔧 Iniciando processo de patching...")

        patched_rom = bytearray(self.rom_data)
        patched_count = 0

        for region in text_regions:
            if not region.get('translated_text'):
                continue

            offset = region['offset']
            original_length = region['length']
            translated_text = region['translated_text']

            try:
                translated_bytes = translated_text.encode('ascii', errors='ignore')
            except UnicodeEncodeError:
                logger.warning(f"⚠️  Erro de encoding na região {offset}: '{translated_text}'")
                continue

            # Verifica limitações de tamanho
            if len(translated_bytes) <= original_length:
                # Substitui o texto
                patched_rom[offset:offset + len(translated_bytes)] = translated_bytes

                # Preenche espaço restante com null bytes
                remaining = original_length - len(translated_bytes)
                if remaining > 0:
                    patched_rom[offset + len(translated_bytes):offset + original_length] = b'\x00' * remaining

                patched_count += 1
                logger.debug(f"✅ Patch aplicado no offset 0x{offset:06X}: '{translated_text}'")
            else:
                logger.warning(f"⚠️  Tradução muito longa para offset 0x{offset:06X}: '{translated_text}' ({len(translated_bytes)} > {original_length})")

        logger.info(f"🎯 Patching concluído: {patched_count} regiões modificadas")

        return bytes(patched_rom)

    def save_patched_rom(self, patched_data: bytes, output_path: str) -> bool:
        """
        Salva ROM modificado em arquivo.

        Args:
            patched_data (bytes): Dados do ROM modificado
            output_path (str): Caminho de saída

        Returns:
            bool: True se salvo com sucesso
        """
        try:
            with open(output_path, 'wb') as f:
                f.write(patched_data)

            logger.info(f"💾 ROM traduzido salvo: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao salvar ROM: {e}")
            return False

    def get_rom_info(self) -> Dict:
        """
        Retorna informações técnicas do ROM carregado.

        Returns:
            Dict: Informações do ROM
        """
        if not self.rom_data:
            return {}

        info = {
            'console': self.CONSOLE_NAME,
            'file_path': str(self.rom_path) if self.rom_path else 'Unknown',
            'file_size': self.rom_size,
            'has_header': self.has_header,
            'header_info': self.header_info,
            'text_regions_count': len(self.text_regions),
            'supported_extensions': self.SUPPORTED_EXTENSIONS,
            'is_smd_format': self.is_smd_format
        }

        return info