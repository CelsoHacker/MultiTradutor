"""
Sega Master System ROM Engine
Handles SMS/Game Gear ROM text extraction and patching.

Technical Specs:
- Z80 CPU @ 3.58 MHz
- 8KB RAM, 16KB VRAM
- ROM sizes: 8KB to 512KB
- Text encoding: ASCII, some Japanese games use Shift-JIS
- Simple linear ROM addressing
"""

import struct
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MasterSystemEngine:
    """
    Engine específico para ROMs do Sega Master System e Game Gear.

    Características técnicas:
    - Suporte a ROMs .sms e .gg
    - Detecção automática de header (se presente)
    - Extração de texto ASCII e caracteres especiais
    - Patching com preservação de tamanho original
    - Validação de checksum TMR SEGA
    """

    # Constantes do sistema
    CONSOLE_NAME = "Sega Master System"
    SUPPORTED_EXTENSIONS = ['.sms', '.gg']
    MAX_ROM_SIZE = 512 * 1024  # 512KB
    MIN_ROM_SIZE = 8 * 1024    # 8KB

    # Header signature (nem todos os ROMs têm)
    TMR_SEGA_SIGNATURE = b'TMR SEGA'

    # Configurações de extração de texto
    MIN_STRING_LENGTH = 4
    MAX_STRING_LENGTH = 256

    def __init__(self):
        """Inicializa o engine do Master System."""
        self.rom_data = None
        self.rom_size = 0
        self.has_header = False
        self.header_offset = 0
        self.text_regions = []
        self.rom_path = None

        logger.info(f"🎮 {self.CONSOLE_NAME} Engine inicializado")

    def load_rom(self, rom_path: str) -> bool:
        """
        Carrega ROM do Master System/Game Gear.

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

            # Detectar header TMR SEGA
            self._detect_header()

            logger.info(f"✅ ROM carregado: {self.rom_path.name} ({self.rom_size} bytes)")
            if self.has_header:
                logger.info(f"📋 Header TMR SEGA detectado no offset 0x{self.header_offset:04X}")

            return True

        except FileNotFoundError:
            logger.error(f"❌ Arquivo não encontrado: {rom_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro ao carregar ROM: {e}")
            return False

    def _detect_header(self) -> None:
        """Detecta se o ROM possui header TMR SEGA."""
        # Header geralmente está no final do ROM
        # Procura nos últimos 16 bytes
        search_start = max(0, self.rom_size - 16)

        for i in range(search_start, self.rom_size - 8):
            if self.rom_data[i:i+8] == self.TMR_SEGA_SIGNATURE:
                self.has_header = True
                self.header_offset = i
                break

    def extract_text_regions(self) -> List[Dict]:
        """
        Extrai regiões de texto do ROM Master System.

        Implementa múltiplas estratégias:
        1. Strings ASCII printáveis
        2. Tabelas de caracteres customizadas
        3. Ponteiros de texto (se detectados)

        Returns:
            List[Dict]: Lista de regiões de texto encontradas
        """
        if not self.rom_data:
            logger.error("❌ ROM não carregado")
            return []

        logger.info("🔍 Iniciando extração de texto...")

        text_regions = []

        # Estratégia 1: Strings ASCII
        ascii_regions = self._extract_ascii_strings()
        text_regions.extend(ascii_regions)

        # Estratégia 2: Ponteiros de texto (avançado)
        pointer_regions = self._extract_pointer_strings()
        text_regions.extend(pointer_regions)

        # Remove duplicatas e ordena por offset
        text_regions = self._deduplicate_regions(text_regions)
        text_regions.sort(key=lambda x: x['offset'])

        self.text_regions = text_regions

        logger.info(f"📝 Extraídas {len(text_regions)} regiões de texto")

        return text_regions

    def _extract_ascii_strings(self) -> List[Dict]:
        """Extrai strings ASCII printáveis do ROM."""
        regions = []
        current_string = ""
        start_offset = 0

        for i, byte in enumerate(self.rom_data):
            # ASCII printável + alguns caracteres especiais
            if (32 <= byte <= 126) or byte in [0x0A, 0x0D]:  # Inclui \n e \r
                if not current_string:
                    start_offset = i
                current_string += chr(byte)
            else:
                # Terminador de string encontrado
                if len(current_string) >= self.MIN_STRING_LENGTH:
                    # Filtrar strings que parecem ser código/dados
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

        # Verificar string final
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
        """
        Extrai strings através de ponteiros (método avançado).

        Master System usa endereçamento de 16-bit.
        Procura por padrões de ponteiros seguidos de texto.
        """
        regions = []

        # Procura por tabelas de ponteiros
        for i in range(0, len(self.rom_data) - 1, 2):
            try:
                # Lê ponteiro little-endian
                pointer = struct.unpack('<H', self.rom_data[i:i+2])[0]

                # Verifica se ponteiro aponta para área válida do ROM
                if 0x8000 <= pointer <= 0xFFFF:
                    # Converte endereço do Z80 para offset do ROM
                    rom_offset = pointer - 0x8000

                    if rom_offset < len(self.rom_data):
                        # Extrai string a partir do ponteiro
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
            except struct.error:
                # Ignora erros de struct.unpack
                continue
            except Exception:
                # Ignora outras exceções
                continue

        logger.info(logger.info(f"Encontradas {len(strings)} strings no offset {hex(offset)}")

        return strings

    def _get_context(self, offset, size=32):
        """Extrai contexto ao redor do offset para análise"""
        try:
            start = max(0, offset - size)
            end = min(len(self.rom_data), offset + size)
            context = self.rom_data[start:end]
            return context.hex()
        except Exception:
            return ""

    def _is_valid_string_char(self, char):
        """Verifica se o caractere é válido para uma string"""
        # Caracteres ASCII imprimíveis e alguns controles específicos
        return (32 <= char <= 126) or char in [0x0A, 0x0D, 0x00]

    def _decode_string(self, data, encoding='ascii'):
        """Decodifica dados binários em string"""
        try:
            if encoding == 'ascii':
                # Remove caracteres nulos e não imprimíveis
                clean_data = bytes(b for b in data if 32 <= b <= 126)
                return clean_data.decode('ascii', errors='ignore')
            elif encoding == 'shift_jis':
                return data.decode('shift_jis', errors='ignore')
            else:
                return data.decode(encoding, errors='ignore')
        except Exception:
            return ""

    def analyze_rom_structure(self):
        """Análise estrutural completa da ROM"""
        analysis = {
            'header': self._analyze_header(),
            'memory_map': self._analyze_memory_mapping(),
            'code_sections': self._find_code_sections(),
            'data_sections': self._find_data_sections(),
            'strings': self.find_strings(),
            'compression': self._detect_compression(),
            'checksum': self._verify_checksum()
        }

        logger.info("Análise estrutural da ROM concluída")
        return analysis

    def _analyze_header(self):
        """Analisa o cabeçalho da ROM"""
        if len(self.rom_data) < 0x8000:
            return {'error': 'ROM muito pequena para análise'}

        # Verifica assinatura TMR SEGA no offset 0x7FF0
        tmr_offset = 0x7FF0
        if len(self.rom_data) > tmr_offset + 10:
            signature = self.rom_data[tmr_offset:tmr_offset + 8]
            if signature == b'TMR SEGA':
                return {
                    'type': 'SMS/GG ROM with header',
                    'signature': signature.decode('ascii'),
                    'region': self._get_region_info(tmr_offset),
                    'checksum_offset': tmr_offset + 10
                }

        return {'type': 'ROM without standard header'}

    def _get_region_info(self, tmr_offset):
        """Extrai informações de região do cabeçalho"""
        try:
            region_byte = self.rom_data[tmr_offset + 15] if len(self.rom_data) > tmr_offset + 15 else 0
            regions = {
                0x40: 'SMS Japan',
                0x50: 'SMS Export',
                0x60: 'GG Japan',
                0x70: 'GG Export'
            }
            return regions.get(region_byte & 0xF0, 'Unknown')
        except Exception:
            return 'Unknown'

    def _analyze_memory_mapping(self):
        """Analisa o mapeamento de memória da ROM"""
        rom_size = len(self.rom_data)

        # Determina o tipo de mapeamento baseado no tamanho
        if rom_size <= 0x8000:  # 32KB
            return {'type': 'No mapper (32KB)', 'banks': 1}
        elif rom_size <= 0x10000:  # 64KB
            return {'type': 'No mapper (64KB)', 'banks': 2}
        elif rom_size <= 0x20000:  # 128KB
            return {'type': 'Sega mapper', 'banks': rom_size // 0x4000}
        else:
            return {'type': 'Sega mapper (large)', 'banks': rom_size // 0x4000}

    def _find_code_sections(self):
        """Identifica seções de código executável"""
        code_sections = []

        # Procura por padrões típicos de código Z80
        for i in range(0, len(self.rom_data) - 3, 0x100):
            section_data = self.rom_data[i:i+0x100]

            # Heurística: procura por instruções Z80 comuns
            code_indicators = 0
            for j in range(len(section_data) - 1):
                opcode = section_data[j]

                # Instruções comuns do Z80
                if opcode in [0x21, 0x31, 0x01, 0x11, 0x22, 0x32, 0x3A, 0x06, 0x0E, 0x16, 0x1E, 0x26, 0x2E, 0x36, 0x3E]:
                    code_indicators += 1
                elif opcode == 0xCD:  # CALL
                    code_indicators += 2
                elif opcode == 0xC3:  # JP
                    code_indicators += 2

            if code_indicators > 10:  # Threshold para considerar como código
                code_sections.append({
                    'offset': i,
                    'size': 0x100,
                    'confidence': min(code_indicators / 20.0, 1.0)
                })

        return code_sections

    def _find_data_sections(self):
        """Identifica seções de dados"""
        data_sections = []

        # Procura por padrões de dados (tabelas, gráficos, etc.)
        for i in range(0, len(self.rom_data) - 0x100, 0x100):
            section_data = self.rom_data[i:i+0x100]

            # Heurística para dados gráficos (patterns repetitivos)
            if self._is_graphics_data(section_data):
                data_sections.append({
                    'offset': i,
                    'size': 0x100,
                    'type': 'graphics',
                    'confidence': 0.8
                })
            elif self._is_table_data(section_data):
                data_sections.append({
                    'offset': i,
                    'size': 0x100,
                    'type': 'table',
                    'confidence': 0.6
                })

        return data_sections

    def _is_graphics_data(self, data):
        """Verifica se os dados podem ser gráficos"""
        # Procura por padrões típicos de tiles 8x8
        zero_count = data.count(0)
        ff_count = data.count(0xFF)

        # Gráficos tendem a ter muitos zeros e 0xFF
        return (zero_count + ff_count) > len(data) * 0.3

    def _is_table_data(self, data):
        """Verifica se os dados podem ser uma tabela"""
        # Procura por padrões repetitivos que indicam tabelas
        patterns = {}
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                pattern = (data[i], data[i+1])
                patterns[pattern] = patterns.get(pattern, 0) + 1

        # Se há muitos padrões repetidos, pode ser uma tabela
        return len(patterns) < len(data) / 4

    def _detect_compression(self):
        """Detecta possível compressão nos dados"""
        # Análise de entropia para detectar compressão
        entropy = self._calculate_entropy(self.rom_data)

        if entropy > 7.5:
            return {'detected': True, 'type': 'possibly_compressed', 'entropy': entropy}
        else:
            return {'detected': False, 'entropy': entropy}

    def _calculate_entropy(self, data):
        """Calcula a entropia dos dados"""
        import math

        if not data:
            return 0

        # Conta a frequência de cada byte
        frequencies = {}
        for byte in data:
            frequencies[byte] = frequencies.get(byte, 0) + 1

        # Calcula a entropia
        entropy = 0
        data_len = len(data)
        for count in frequencies.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability)

        return entropy

    def _verify_checksum(self):
        """Verifica o checksum da ROM"""
        # Checksum simples: soma de todos os bytes
        checksum = sum(self.rom_data) & 0xFFFF

        # Para ROMs com cabeçalho, verifica o checksum armazenado
        if len(self.rom_data) > 0x7FFA:
            stored_checksum = struct.unpack('<H', self.rom_data[0x7FFA:0x7FFC])[0]

            # Calcula checksum excluindo a área do próprio checksum
            calculated_checksum = (sum(self.rom_data[:0x7FFA]) + sum(self.rom_data[0x7FFC:])) & 0xFFFF

            return {
                'stored': stored_checksum,
                'calculated': calculated_checksum,
                'valid': stored_checksum == calculated_checksum
            }

        return {'simple_checksum': checksum}

    def patch_rom(self, patches):
        """Aplica patches na ROM"""
        if not self.rom_data:
            return False

        patched_data = bytearray(self.rom_data)

        for patch in patches:
            try:
                offset = patch['offset']
                new_data = patch['data']

                if isinstance(new_data, str):
                    new_data = bytes.fromhex(new_data)

                if offset + len(new_data) <= len(patched_data):
                    patched_data[offset:offset+len(new_data)] = new_data
                    logger.info(f"Patch aplicado no offset {hex(offset)}: {len(new_data)} bytes")
                else:
                    logger.warning(f"Patch no offset {hex(offset)} excede o tamanho da ROM")

            except Exception as e:
                logger.error(f"Erro ao aplicar patch: {e}")
                return False

        self.rom_data = bytes(patched_data)
        return True

    def save_rom(self, filename):
        """Salva a ROM modificada"""
        try:
            with open(filename, 'wb') as f:
                f.write(self.rom_data)
            logger.info(f"ROM salva como {filename}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar ROM: {e}")
            return False

    def export_analysis(self, filename):
        """Exporta análise para arquivo JSON"""
        try:
            analysis = self.analyze_rom_structure()

            # Converte bytes para hex strings para serialização JSON
            def convert_bytes_to_hex(obj):
                if isinstance(obj, bytes):
                    return obj.hex()
                elif isinstance(obj, dict):
                    return {k: convert_bytes_to_hex(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_bytes_to_hex(item) for item in obj]
                return obj

            analysis_serializable = convert_bytes_to_hex(analysis)

            with open(filename, 'w') as f:
                json.dump(analysis_serializable, f, indent=2)

            logger.info(f"Análise exportada para {filename}")
            return True
        except Exception as e:
            logger.error(f"Erro ao exportar análise: {e}")
            return False

# Exemplo de uso
if __name__ == "__main__":
    # Inicializa o emulador
    emulator = SMSEmulator()

    # Carrega uma ROM
    rom_file = "sonic.sms"  # Substitua pelo caminho da sua ROM

    if emulator.load_rom(rom_file):
        print(f"ROM carregada: {rom_file}")
        print(f"Tamanho: {len(emulator.rom_data)} bytes")

        # Executa análise completa
        analysis = emulator.analyze_rom_structure()
        print("\n=== ANÁLISE DA ROM ===")
        print(f"Tipo: {analysis['header'].get('type', 'Desconhecido')}")
        print(f"Mapeamento: {analysis['memory_map']['type']}")
        print(f"Seções de código encontradas: {len(analysis['code_sections'])}")
        print(f"Strings encontradas: {len(analysis['strings'])}")

        # Busca strings específicas
        print("\n=== STRINGS ENCONTRADAS ===")
        strings = emulator.find_strings(min_length=4)
        for i, string_info in enumerate(strings[:10]):  # Mostra apenas as 10 primeiras
            print(f"{i+1}. Offset {hex(string_info['offset'])}: '{string_info['text']}'")

        # Exporta análise
        emulator.export_analysis("rom_analysis.json")

        # Exemplo de patch simples
        patches = [
            {
                'offset': 0x1000,
                'data': b'PATCHED!'
            }
        ]

        if emulator.patch_rom(patches):
            emulator.save_rom("sonic_patched.sms")

    else:
        print("Erro ao carregar ROM")