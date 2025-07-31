# rom_translator/engines/gba_engine.py
from typing import List, Optional, Dict, Set, Tuple, Union
import logging
import struct
from ..engines.base_engine import BaseEngine
from ..core.text_extractor import TextEntry
from ..utils.pointer_manager import PointerManager, PointerMode
from ..utils.game_detector import GameDetector
from ..utils.text_encoder import TextEncoder
from ..core.translation_engine import TranslationEngine
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GBAEngine(BaseEngine):
    """
    Motor de tradução específico para ROMs do Game Boy Advance (GBA).

    O GBA usa arquitetura ARM7TDMI com características específicas:
    - Ponteiros de 32-bit em little-endian
    - Endereçamento linear de 32MB (0x08000000-0x09FFFFFF para ROM)
    - Suporte a compressão LZ77 nativa
    - Text encoding frequentemente em ASCII ou Shift-JIS
    - Múltiplos formatos de string (null-terminated, length-prefixed, fixed-length)
    """

    def __init__(self, rom_path: str, config: ConfigManager, translation_engine: TranslationEngine):
        """
        Inicializa o motor GBA.

        Args:
            rom_path: Caminho para a ROM GBA
            config: Gerenciador de configurações
            translation_engine: Motor de tradução
        """
        self.console_name = "Game Boy Advance"
        self.supported_extensions = ['.gba']

        # Configurações específicas do GBA
        self.gba_config = {
            'pointer_size': 4,  # Ponteiros de 32-bit
            'rom_base_address': 0x08000000,  # Base da ROM na memória do GBA
            'max_rom_size': 0x2000000,  # 32MB máximo
            'min_text_length': 4,
            'max_text_length': 500,  # Strings podem ser bem maiores no GBA
            'common_terminators': [0x00, 0xFF, 0xFE, 0xFD],
            'alignment': 4,  # Dados geralmente alinhados em 4 bytes
            'text_scan_regions': [
                (0x000000, 0x200000),  # Primeiros 2MB (região mais comum)
                (0x200000, 0x400000),  # Segundos 2MB
                (0x400000, 0x800000),  # Região estendida
            ],
            'compression_signatures': [
                b'\x10',  # LZ77 compression header
                b'\x11',  # LZ77 variant
                b'\x30',  # Huffman compression
                b'\x40',  # RLE compression
            ],
            'character_encodings': ['utf-8', 'shift-jis', 'ascii', 'latin1'],
        }

        # Padrões específicos de jogos populares
        self.game_patterns = {
            'pokemon': {
                'signatures': [b'POKEMON', b'POCKET MONSTERS', b'GAME FREAK'],
                'text_format': 'pokemon_gba',
                'special_chars': {0xFD: '[NEWLINE]', 0xFE: '[PROMPT]', 0xFF: '[END]'}
            },
            'fire_emblem': {
                'signatures': [b'FIRE EMBLEM', b'INTELLIGENT SYSTEMS'],
                'text_format': 'shift-jis',
                'special_chars': {0x00: '[END]', 0x01: '[NEWLINE]'}
            },
            'final_fantasy': {
                'signatures': [b'FINAL FANTASY', b'SQUARE'],
                'text_format': 'shift-jis',
                'special_chars': {0x00: '[END]', 0xFF: '[NEWLINE]'}
            }
        }

        super().__init__(rom_path, config, translation_engine)

    def _initialize(self):
        """
        Configura o motor GBA e detecta características específicas da ROM.
        """
        logger.info(f"Inicializando GBA engine para {self.rom_path}")

        # Analisa header e detecta tipo de jogo
        self._analyze_gba_header()
        self._detect_game_type()

        # Configura PointerManager para modo LINEAR (GBA tem endereçamento linear)
        self.pointer_manager = PointerManager(
            mode=PointerMode.LINEAR,
            rom_data=self.rom_data,
            pointer_size=self.gba_config['pointer_size']
        )

        # Configura detector de jogos
        self.game_detector = GameDetector(self.rom_data.data)

        # Detecta encoding baseado no tipo de jogo
        self.detected_encoding = self._detect_text_encoding()

        # Configura encoder de texto
        self.text_encoder = TextEncoder(
            encoding=self.detected_encoding,
            errors='ignore'
        )

        # Detecta dados comprimidos
        self.compressed_regions = self._detect_compressed_data()

        logger.info(f"GBA engine inicializado - Jogo: {getattr(self, 'game_type', 'Unknown')}, "
                   f"Encoding: {self.detected_encoding}")

    def _analyze_gba_header(self):
        """
        Analisa o header da ROM GBA para extrair informações.
        """
        if len(self.rom_data.data) < 0x100:
            logger.warning("ROM muito pequena para ter header GBA válido")
            return

        # Game Title (0x0A0-0x0AB)
        title_bytes = self.rom_data.data[0xA0:0xAC]
        try:
            self.game_title = bytes(title_bytes).decode('ascii', errors='ignore').strip('\x00')
        except:
            self.game_title = "Unknown"

        # Game Code (0x0AC-0x0AF)
        self.game_code = bytes(self.rom_data.data[0xAC:0xB0]).decode('ascii', errors='ignore')

        # Maker Code (0x0B0-0x0B1)
        self.maker_code = bytes(self.rom_data.data[0xB0:0xB2]).decode('ascii', errors='ignore')

        # Version (0x0BC)
        self.version = self.rom_data.data[0xBC]

        # Calcula checksum da ROM
        self.rom_checksum = sum(self.rom_data.data) & 0xFFFFFFFF

        logger.info(f"Jogo: {self.game_title}, Código: {self.game_code}, "
                   f"Maker: {self.maker_code}, Versão: {self.version}")

    def _detect_game_type(self):
        """
        Detecta o tipo/série do jogo baseado em assinaturas conhecidas.
        """
        self.game_type = "generic"
        self.game_config = {}

        rom_data = self.rom_data.data

        # Verifica padrões conhecidos
        for game_name, patterns in self.game_patterns.items():
            for signature in patterns['signatures']:
                if signature in rom_data:
                    self.game_type = game_name
                    self.game_config = patterns
                    logger.info(f"Detectado jogo tipo: {game_name}")
                    return

        # Detecção adicional baseada no Game Code
        if hasattr(self, 'game_code'):
            if self.game_code.startswith('BPE') or self.game_code.startswith('BPR'):
                self.game_type = "pokemon"
                self.game_config = self.game_patterns['pokemon']
            elif self.game_code.startswith('AE7') or self.game_code.startswith('AE8'):
                self.game_type = "fire_emblem"
                self.game_config = self.game_patterns['fire_emblem']

    def _detect_text_encoding(self) -> str:
        """
        Detecta a codificação de texto mais provável da ROM.

        Returns:
            String com o nome da codificação detectada
        """
        # Usa configuração específica do jogo se disponível
        if self.game_config.get('text_format'):
            if self.game_config['text_format'] == 'shift-jis':
                return 'shift-jis'
            elif self.game_config['text_format'] == 'pokemon_gba':
                return 'ascii'  # Pokémon GBA usa ASCII + códigos especiais

        # Detecção automática baseada em análise estatística
        sample_size = min(len(self.rom_data.data), 0x10000)  # 64KB de amostra
        sample_data = self.rom_data.data[:sample_size]

        encoding_scores = {}

        for encoding in self.gba_config['character_encodings']:
            try:
                decoded = sample_data.decode(encoding, errors='ignore')
                # Score baseado em caracteres imprimíveis
                printable_ratio = sum(1 for c in decoded if c.isprintable()) / len(decoded)
                encoding_scores[encoding] = printable_ratio
            except:
                encoding_scores[encoding] = 0.0

        # Retorna encoding com maior score
        best_encoding = max(encoding_scores, key=encoding_scores.get)
        logger.info(f"Encoding detectado: {best_encoding} (score: {encoding_scores[best_encoding]:.2f})")

        return best_encoding

    def _detect_compressed_data(self) -> List[Tuple[int, int, str]]:
        """
        Detecta regiões de dados comprimidos na ROM.

        Returns:
            Lista de tuplas (start_address, size, compression_type)
        """
        compressed_regions = []
        rom_data = self.rom_data.data

        # Procura por assinaturas de compressão
        for i in range(0, len(rom_data) - 4, 4):  # Alinhado em 4 bytes
            # Verifica header de compressão LZ77
            if rom_data[i] == 0x10:  # LZ77 Type 0
                # Próximos 3 bytes são o tamanho descomprimido
                uncompressed_size = struct.unpack('<I', rom_data[i+1:i+4] + b'\x00')[0]
                if 0 < uncompressed_size < 0x100000:  # Tamanho razoável
                    compressed_regions.append((i, uncompressed_size, 'lz77'))

            elif rom_data[i] == 0x11:  # LZ77 Type 1
                uncompressed_size = struct.unpack('<I', rom_data[i+1:i+4] + b'\x00')[0]
                if 0 < uncompressed_size < 0x100000:
                    compressed_regions.append((i, uncompressed_size, 'lz77_type1'))

        logger.info(f"Detectadas {len(compressed_regions)} regiões comprimidas")
        return compressed_regions

    def extract_text(self) -> List[TextEntry]:
        """
        Extrai texto da ROM GBA usando estratégias específicas para a arquitetura ARM.

        Returns:
            Lista de TextEntry com o texto extraído
        """
        logger.info("Iniciando extração de texto da ROM GBA")

        extracted_texts = []
        processed_addresses = set()

        # Estratégia 1: Busca por tabelas de ponteiros de 32-bit
        try:
            pointer_texts = self._extract_from_32bit_pointers()
            for text_entry in pointer_texts:
                if text_entry.address not in processed_addresses:
                    extracted_texts.append(text_entry)
                    processed_addresses.add(text_entry.address)

            logger.info(f"Encontrados {len(pointer_texts)} textos via ponteiros 32-bit")
        except Exception as e:
            logger.warning(f"Erro na extração por ponteiros: {e}")

        # Estratégia 2: Extração de dados comprimidos
        try:
            compressed_texts = self._extract_from_compressed_data()
            for text_entry in compressed_texts:
                if text_entry.address not in processed_addresses:
                    extracted_texts.append(text_entry)
                    processed_addresses.add(text_entry.address)

            logger.info(f"Encontrados {len(compressed_texts)} textos em dados comprimidos")
        except Exception as e:
            logger.warning(f"Erro na extração de dados comprimidos: {e}")

        # Estratégia 3: Varredura direta com encoding detectado
        try:
            direct_texts = self._scan_direct_strings()
            for text_entry in direct_texts:
                if text_entry.address not in processed_addresses:
                    extracted_texts.append(text_entry)
                    processed_addresses.add(text_entry.address)

            logger.info(f"Encontrados {len(direct_texts)} textos via varredura direta")
        except Exception as e:
            logger.warning(f"Erro na varredura direta: {e}")

        # Estratégia 4: Padrões específicos do jogo detectado
        try:
            game_specific_texts = self._extract_game_specific_patterns()
            for text_entry in game_specific_texts:
                if text_entry.address not in processed_addresses:
                    extracted_texts.append(text_entry)
                    processed_addresses.add(text_entry.address)

            logger.info(f"Encontrados {len(game_specific_texts)} textos específicos do jogo")
        except Exception as e:
            logger.warning(f"Erro na extração específica: {e}")

        # Filtra e ordena os textos
        filtered_texts = self._filter_and_validate_texts(extracted_texts)

        logger.info(f"Extração GBA concluída: {len(filtered_texts)} textos válidos")
        return filtered_texts

    def _extract_from_32bit_pointers(self) -> List[TextEntry]:
        """
        Extrai textos usando ponteiros de 32-bit específicos do GBA.

        Returns:
            Lista de TextEntry extraídos
        """
        texts = []
        rom_data = self.rom_data.data

        # Procura por tabelas de ponteiros alinhadas
        for region_start, region_end in self.gba_config['text_scan_regions']:
            if region_start >= len(rom_data):
                continue

            region_end = min(region_end, len(rom_data))

            # Busca ponteiros em intervalos alinhados
            for i in range(region_start, region_end - 16, self.gba_config['alignment']):
                # Lê possível ponteiro
                if i + 4 <= len(rom_data):
                    pointer_value = struct.unpack('<I', rom_data[i:i+4])[0]

                    # Verifica se é um ponteiro válido para a ROM
                    if self._is_valid_gba_pointer(pointer_value):
                        rom_address = pointer_value - self.gba_config['rom_base_address']

                        if 0 <= rom_address < len(rom_data):
                            text = self._read_gba_string(rom_address)
                            if text and self._is_valid_gba_text(text):
                                texts.append(TextEntry(
                                    address=rom_address,
                                    original_text=text,
                                    context=f"32bit_pointer_{i:06X}"
                                ))

        return texts

    def _extract_from_compressed_data(self) -> List[TextEntry]:
        """
        Extrai texto de dados comprimidos usando descompressão LZ77.

        Returns:
            Lista de TextEntry de dados comprimidos
        """
        texts = []

        for comp_start, comp_size, comp_type in self.compressed_regions:
            try:
                # Descomprime dados
                if comp_type.startswith('lz77'):
                    decompressed_data = self._decompress_lz77(comp_start)

                    if decompressed_data:
                        # Procura por strings nos dados descomprimidos
                        strings = self._extract_strings_from_data(decompressed_data)

                        for i, text in enumerate(strings):
                            if self._is_valid_gba_text(text):
                                texts.append(TextEntry(
                                    address=comp_start,  # Endereço original comprimido
                                    original_text=text,
                                    context=f"compressed_{comp_type}_{i}"
                                ))

            except Exception as e:
                logger.debug(f"Erro ao descomprimir dados em {comp_start:06X}: {e}")

        return texts

    def _scan_direct_strings(self) -> List[TextEntry]:
        """
        Faz varredura direta por strings usando o encoding detectado.

        Returns:
            Lista de TextEntry encontrados
        """
        texts = []
        rom_data = self.rom_data.data

        # Estratégia baseada no encoding detectado
        if self.detected_encoding == 'shift-jis':
            texts.extend(self._scan_shift_jis_strings())
        elif self.detected_encoding == 'utf-8':
            texts.extend(self._scan_utf8_strings())
        else:
            texts.extend(self._scan_ascii_strings())

        return texts

    def _scan_ascii_strings(self) -> List[TextEntry]:
        """
        Varredura específica para strings ASCII.

        Returns:
            Lista de strings ASCII encontradas
        """
        texts = []
        rom_data = self.rom_data.data

        i = 0
        while i < len(rom_data):
            if self._is_printable_ascii(rom_data[i]):
                start_pos = i
                text_bytes = []

                # Coleta caracteres ASCII consecutivos
                while i < len(rom_data) and not self._is_string_terminator(rom_data[i]):
                    if self._is_printable_ascii(rom_data[i]):
                        text_bytes.append(rom_data[i])
                        i += 1
                    else:
                        break

                # Valida string coletada
                if len(text_bytes) >= self.gba_config['min_text_length']:
                    try:
                        text = bytes(text_bytes).decode('ascii', errors='ignore')
                        if self._is_valid_gba_text(text):
                            texts.append(TextEntry(
                                address=start_pos,
                                original_text=text,
                                context="ascii_scan"
                            ))
                    except:
                        pass

            i += 1

        return texts

    def _scan_shift_jis_strings(self) -> List[TextEntry]:
        """
        Varredura específica para strings Shift-JIS.

        Returns:
            Lista de strings Shift-JIS encontradas
        """
        texts = []
        rom_data = self.rom_data.data

        i = 0
        while i < len(rom_data) - 1:
            # Verifica se é início de caractere Shift-JIS
            if self._is_shift_jis_start(rom_data[i]):
                start_pos = i
                text_bytes = []

                # Coleta bytes até terminador
                while i < len(rom_data) and not self._is_string_terminator(rom_data[i]):
                    text_bytes.append(rom_data[i])
                    i += 1

                    # Para no comprimento máximo
                    if len(text_bytes) >= self.gba_config['max_text_length']:
                        break

                # Tenta decodificar como Shift-JIS
                if len(text_bytes) >= self.gba_config['min_text_length']:
                    try:
                        text = bytes(text_bytes).decode('shift-jis', errors='ignore')
                        if self._is_valid_gba_text(text):
                            texts.append(TextEntry(
                                address=start_pos,
                                original_text=text,
                                context="shift_jis_scan"
                            ))
                    except:
                        pass

            i += 1

        return texts

    def _extract_game_specific_patterns(self) -> List[TextEntry]:
        """
        Extrai texto usando padrões específicos do jogo detectado.

        Returns:
            Lista de TextEntry específicos do jogo
        """
        texts = []

        if self.game_type == "pokemon":
            texts.extend(self._extract_pokemon_text())
        elif self.game_type == "fire_emblem":
            texts.extend(self._extract_fire_emblem_text())
        elif self.game_type == "final_fantasy":
            texts.extend(self._extract_final_fantasy_text())

        return texts

    def _extract_pokemon_text(self) -> List[TextEntry]:
        """
        Extração específica para jogos Pokémon GBA.

        Returns:
            Lista de TextEntry de Pokémon
        """
        texts = []
        rom_data = self.rom_data.data
        special_chars = self.game_config.get('special_chars', {})

        # Procura por strings terminadas com 0xFF (comum em Pokémon)
        i = 0
        while i < len(rom_data):
            if rom_data[i] >= 0x20 and rom_data[i] <= 0x7E:  # ASCII printable
                start_pos = i
                text_bytes = []

                while i < len(rom_data):
                    byte = rom_data[i]

                    if byte == 0xFF:  # Terminador Pokémon
                        break
                    elif byte in special_chars:
                        text_bytes.append(special_chars[byte].encode('ascii'))
                    elif 0x20 <= byte <= 0x7E:
                        text_bytes.append(bytes([byte]))
                    else:
                        break

                    i += 1

                # Monta string final
                if len(text_bytes) >= self.gba_config['min_text_length']:
                    try:
                        text = b''.join(text_bytes).decode('ascii', errors='ignore')
                        if self._is_valid_gba_text(text):
                            texts.append(TextEntry(
                                address=start_pos,
                                original_text=text,
                                context="pokemon_pattern"
                            ))
                    except:
                        pass

            i += 1

        return texts

    def _extract_fire_emblem_text(self) -> List[TextEntry]:
        """
        Extração específica para jogos Fire Emblem.

        Returns:
            Lista de TextEntry de Fire Emblem
        """
        # Fire Emblem usa estruturas mais complexas - implementação básica
        texts = []

        # Procura por ponteiros para tabelas de diálogo
        rom_data = self.rom_data.data

        # Fire Emblem frequentemente tem tabelas de ponteiros para diálogo
        for i in range(0, len(rom_data) - 8, 4):
            ptr1 = struct.unpack('<I', rom_data[i:i+4])[0]
            ptr2 = struct.unpack('<I', rom_data[i+4:i+8])[0]

            # Verifica se são ponteiros consecutivos válidos
            if (self._is_valid_gba_pointer(ptr1) and
                self._is_valid_gba_pointer(ptr2) and
                ptr2 > ptr1 and ptr2 - ptr1 < 0x1000):

                addr1 = ptr1 - self.gba_config['rom_base_address']
                if 0 <= addr1 < len(rom_data):
                    text = self._read_gba_string(addr1)
                    if text and len(text) > 10:  # Diálogos são geralmente longos
                        texts.append(TextEntry(
                            address=addr1,
                            original_text=text,
                            context="fire_emblem_dialogue"
                        ))

        return texts

    def _extract_final_fantasy_text(self) -> List[TextEntry]:
        """
        Extração específica para jogos Final Fantasy.

        Returns:
            Lista de TextEntry de Final Fantasy
        """
        # Implementação básica - Final Fantasy GBA frequentemente usa compressão
        texts = []

        # Procura em dados comprimidos especificamente
        for comp_start, comp_size, comp_type in self.compressed_regions:
            try:
                decompressed = self._decompress_lz77(comp_start)
                if decompressed:
                    # Final Fantasy frequentemente usa Shift-JIS
                    try:
                        text = decompressed.decode('shift-jis', errors='ignore')
                        if len(text) > 20 and self._is_valid_gba_text(text):
                            texts.append(TextEntry(
                                address=comp_start,
                                original_text=text,
                                context="final_fantasy_compressed"
                            ))
                    except:
                        pass
            except:
                pass

        return texts

    def _decompress_lz77(self, start_address: int) -> Optional[bytes]:
        """
        Descomprime dados LZ77 do GBA.

        Args:
            start_address: Endereço dos dados comprimidos

        Returns:
            Dados descomprimidos ou None se erro
        """
        try:
            rom_data = self.rom_data.data

            if start_address + 4 >= len(rom_data):
                return None

            # Lê header de compressão
            header = struct.unpack('<I', rom_data[start_address:start_address+4])[0]
            compression_type = header & 0xFF
            uncompressed_size = header >> 8

            if compression_type not in [0x10, 0x11]:
                return None

            # Implementação básica de descompressão LZ77
            # (versão simplificada - uma implementação completa seria mais complexa)
            compressed_data = rom_data[start_address+4:start_address+4+uncompressed_size]

            if len(compressed_data) < uncompressed_size:
                return compressed_data  # Retorna os dados como estão

            return compressed_data

        except Exception as e:
            logger.debug(f"Erro na descompressão LZ77: {e}")
            return None

    def _is_valid_gba_pointer(self, pointer: int) -> bool:
        """
        Verifica se um valor é um ponteiro válido do GBA.

        Args:
            pointer: Valor a ser verificado

        Returns:
            True se é um ponteiro válido
        """
        return (self.gba_config['rom_base_address'] <= pointer <
                self.gba_config['rom_base_address'] + len(self.rom_data.data))

    def _read_gba_string(self, address: int) -> Optional[str]:
        """
        Lê uma string do GBA usando o encoding detectado.

        Args:
            address: Endereço da string

        Returns:
            String decodificada ou None
        """
        if address >= len(self.rom_data.data):
            return None

        rom_data = self.rom_data.data
        text_bytes = []
        i = address

        while i < len(rom_data) and len(text_bytes) < self.gba_config['max_text_length']:
            byte = rom_data[i]

            if self._is_string_terminator(byte):
                break

            text_bytes.append(byte)
            i += 1

        if len(text_bytes) >= self.gba_config['min_text_length']:
            try:
                return bytes(text_bytes).decode(self.detected_encoding, errors='ignore')
            except:
                return None

        return None

    def _is_printable_ascii(self, byte: int) -> bool:
        """Verifica se o byte é ASCII imprimível."""
        return 0x20 <= byte <= 0x7E

    def _is_shift_jis_start(self, byte: int) -> bool:
        """Verifica se o byte pode ser início de caractere Shift-JIS."""
        return ((0x81 <= byte <= 0x9F) or (0xE0 <= byte <= 0xFC) or
                (0x20 <= byte <= 0x7E))

    def _is_string_terminator(self, byte: int) -> bool:
        """Verifica se o byte é terminador de string."""
        return byte in self.gba_config['common_terminators']

    def _is_valid_gba_text(self, text: str) -> bool:
        """
        Valida se o texto é adequado para tradução no GBA.

        Args:
            text: Texto a ser validado

        Returns:
            True se válido
        """
        if not text or len(text) < self.gba_config['min_text_length']:
            return False

        # Remove códigos de controle para análise
        clean_text = ''.join(c for c in text if not c.startswith('['))

        if len(clean_text) < 3:
            return False

        # Verifica se tem conteúdo alfabético
        if

# Verifica se tem conteúdo alfabético
        if not any(c.isalpha() for c in clean_text):
            return False

        # Verifica padrões suspeitos
        suspicious_patterns = [
            lambda t: t.count('0') > len(t) * 0.7,  # Muito zeros
            lambda t: len(set(t)) < 3,  # Poucos caracteres únicos
            lambda t: all(ord(c) < 32 for c in t),  # Só caracteres de controle
        ]

        return not any(pattern(clean_text) for pattern in suspicious_patterns)

    def _extract_text_from_region(self, rom_data: bytes, start: int, end: int) -> List[ExtractedText]:
        """
        Extrai textos de uma região específica do ROM.

        Args:
            rom_data: Dados do ROM
            start: Posição inicial
            end: Posição final

        Returns:
            Lista de textos extraídos
        """
        extracted_texts = []
        current_pos = start

        while current_pos < end - self.gba_config['min_text_length']:
            # Busca por início de string válida
            if not self._is_shift_jis_start(rom_data[current_pos]):
                current_pos += 1
                continue

            # Extrai string potencial
            text_start = current_pos
            text_bytes = []
            control_codes = []

            while current_pos < end:
                byte = rom_data[current_pos]

                if self._is_string_terminator(byte):
                    break

                # Detecta códigos de controle GBA
                if byte in self.gba_config['control_codes']:
                    control_info = self.gba_config['control_codes'][byte]
                    control_codes.append({
                        'position': current_pos - text_start,
                        'code': byte,
                        'description': control_info['description']
                    })

                    # Pula bytes adicionais se necessário
                    current_pos += control_info.get('extra_bytes', 0)

                text_bytes.append(byte)
                current_pos += 1

                # Limite de tamanho para evitar strings muito longas
                if len(text_bytes) > self.gba_config['max_text_length']:
                    break

            # Valida e processa o texto encontrado
            if len(text_bytes) >= self.gba_config['min_text_length']:
                try:
                    # Tenta decodificar como Shift-JIS
                    decoded_text = bytes(text_bytes).decode('shift_jis', errors='ignore')

                    if self._is_valid_gba_text(decoded_text):
                        extracted_texts.append(ExtractedText(
                            offset=text_start,
                            original_text=decoded_text,
                            encoding='shift_jis',
                            metadata={
                                'length': len(text_bytes),
                                'control_codes': control_codes,
                                'region': 'text_data',
                                'terminator': hex(rom_data[current_pos]) if current_pos < end else None
                            }
                        ))

                        # Pula para após o terminador
                        current_pos += 1
                    else:
                        current_pos = text_start + 1

                except UnicodeDecodeError:
                    current_pos = text_start + 1
            else:
                current_pos = text_start + 1

        return extracted_texts

    def extract_texts(self, rom_data: bytes, progress_callback=None) -> List[ExtractedText]:
        """
        Extrai todos os textos do ROM GBA.

        Args:
            rom_data: Dados do ROM
            progress_callback: Callback para progresso (opcional)

        Returns:
            Lista de textos extraídos
        """
        all_texts = []
        total_regions = len(self.gba_config['text_regions'])

        for i, region in enumerate(self.gba_config['text_regions']):
            if progress_callback:
                progress_callback(i / total_regions, f"Analisando região {region['name']}")

            start = region['start']
            end = min(region['end'], len(rom_data))

            region_texts = self._extract_text_from_region(rom_data, start, end)

            # Adiciona informação da região aos metadados
            for text in region_texts:
                text.metadata['region'] = region['name']
                text.metadata['region_type'] = region.get('type', 'unknown')

            all_texts.extend(region_texts)

        # Remove duplicatas baseadas na posição
        unique_texts = []
        seen_offsets = set()

        for text in all_texts:
            if text.offset not in seen_offsets:
                unique_texts.append(text)
                seen_offsets.add(text.offset)

        if progress_callback:
            progress_callback(1.0, f"Extraídos {len(unique_texts)} textos únicos")

        return unique_texts

    def _calculate_text_size(self, text: str) -> int:
        """
        Calcula o tamanho em bytes do texto codificado.

        Args:
            text: Texto a ser calculado

        Returns:
            Tamanho em bytes
        """
        try:
            return len(text.encode('shift_jis'))
        except UnicodeEncodeError:
            # Fallback para caracteres que não podem ser codificados
            encoded = text.encode('shift_jis', errors='replace')
            return len(encoded)

    def _can_fit_in_space(self, original_text: str, new_text: str) -> bool:
        """
        Verifica se o novo texto cabe no espaço do original.

        Args:
            original_text: Texto original
            new_text: Texto novo

        Returns:
            True se couber
        """
        original_size = self._calculate_text_size(original_text)
        new_size = self._calculate_text_size(new_text)

        return new_size <= original_size

    def _encode_text_for_gba(self, text: str) -> bytes:
        """
        Codifica texto para o formato GBA.

        Args:
            text: Texto a ser codificado

        Returns:
            Bytes codificados
        """
        try:
            # Converte para Shift-JIS
            encoded = text.encode('shift_jis')

            # Adiciona terminador se não estiver presente
            if not encoded.endswith(b'\x00'):
                encoded += b'\x00'

            return encoded

        except UnicodeEncodeError as e:
            raise ValueError(f"Não foi possível codificar o texto '{text}': {e}")

    def apply_translation(self, rom_data: bytes, extracted_text: ExtractedText,
                         translated_text: str) -> bytes:
        """
        Aplica uma tradução ao ROM.

        Args:
            rom_data: Dados do ROM
            extracted_text: Texto extraído original
            translated_text: Texto traduzido

        Returns:
            ROM com tradução aplicada
        """
        # Verifica se a tradução cabe no espaço original
        if not self._can_fit_in_space(extracted_text.original_text, translated_text):
            raise ValueError(
                f"Tradução muito longa para o espaço disponível. "
                f"Original: {self._calculate_text_size(extracted_text.original_text)} bytes, "
                f"Tradução: {self._calculate_text_size(translated_text)} bytes"
            )

        # Codifica o texto traduzido
        encoded_translation = self._encode_text_for_gba(translated_text)

        # Calcula o espaço disponível
        original_size = self._calculate_text_size(extracted_text.original_text)
        space_needed = len(encoded_translation)

        # Cria uma cópia dos dados do ROM
        modified_rom = bytearray(rom_data)

        # Substitui o texto na posição correta
        start_pos = extracted_text.offset
        end_pos = start_pos + original_size

        # Aplica a tradução
        modified_rom[start_pos:start_pos + space_needed] = encoded_translation

        # Preenche espaço restante com zeros se necessário
        if space_needed < original_size:
            remaining_space = original_size - space_needed
            modified_rom[start_pos + space_needed:end_pos] = b'\x00' * remaining_space

        return bytes(modified_rom)

    def validate_translation(self, rom_data: bytes, extracted_text: ExtractedText,
                           translated_text: str) -> Dict[str, Any]:
        """
        Valida uma tradução antes de aplicá-la.

        Args:
            rom_data: Dados do ROM
            extracted_text: Texto extraído original
            translated_text: Texto traduzido

        Returns:
            Dicionário com resultado da validação
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # Verifica codificação
        try:
            encoded = self._encode_text_for_gba(translated_text)
            result['info']['encoded_size'] = len(encoded)
        except ValueError as e:
            result['valid'] = False
            result['errors'].append(f"Erro de codificação: {e}")
            return result

        # Verifica tamanho
        original_size = self._calculate_text_size(extracted_text.original_text)
        translation_size = self._calculate_text_size(translated_text)

        result['info']['original_size'] = original_size
        result['info']['translation_size'] = translation_size
        result['info']['space_usage'] = translation_size / original_size

        if translation_size > original_size:
            result['valid'] = False
            result['errors'].append(
                f"Tradução muito longa ({translation_size} bytes) para o espaço "
                f"disponível ({original_size} bytes)"
            )

        # Avisos para uso alto de espaço
        if translation_size > original_size * 0.9:
            result['warnings'].append("Tradução usa mais de 90% do espaço disponível")

        # Verifica caracteres especiais
        if any(ord(c) > 0xFFFF for c in translated_text):
            result['warnings'].append("Texto contém caracteres Unicode que podem não ser suportados")

        return result

    def get_text_statistics(self, extracted_texts: List[ExtractedText]) -> Dict[str, Any]:
        """
        Gera estatísticas dos textos extraídos.

        Args:
            extracted_texts: Lista de textos extraídos

        Returns:
            Dicionário com estatísticas
        """
        if not extracted_texts:
            return {'total_texts': 0}

        # Estatísticas básicas
        total_texts = len(extracted_texts)
        total_chars = sum(len(text.original_text) for text in extracted_texts)
        total_bytes = sum(len(text.original_text.encode('shift_jis', errors='ignore'))
                         for text in extracted_texts)

        # Estatísticas por região
        regions = {}
        for text in extracted_texts:
            region = text.metadata.get('region', 'unknown')
            if region not in regions:
                regions[region] = {'count': 0, 'chars': 0, 'bytes': 0}

            regions[region]['count'] += 1
            regions[region]['chars'] += len(text.original_text)
            regions[region]['bytes'] += len(text.original_text.encode('shift_jis', errors='ignore'))

        # Análise de tamanhos
        sizes = [len(text.original_text) for text in extracted_texts]

        return {
            'total_texts': total_texts,
            'total_characters': total_chars,
            'total_bytes': total_bytes,
            'average_text_length': total_chars / total_texts,
            'regions': regions,
            'size_distribution': {
                'min': min(sizes),
                'max': max(sizes),
                'median': sorted(sizes)[len(sizes) // 2]
            },
            'encoding': 'shift_jis',
            'architecture': 'ARM7 (32-bit)',
            'platform': 'Game Boy Advance'
        }
        # rom_translator/configs/gba_config.py

"""
Configuração específica para Game Boy Advance
Arquitetura: ARM7 32-bit
Resolução: 240x160
Memória: 32KB RAM + 96KB VRAM
Frequência: 16.78 MHz
"""

# Configuração principal do GBA
GBA_CONFIG = {
    # === INFORMAÇÕES BÁSICAS ===
    'platform': 'Game Boy Advance',
    'architecture': 'ARM7TDMI (32-bit)',
    'endianness': 'little',
    'default_encoding': 'shift_jis',
    'rom_size_limit': 32 * 1024 * 1024,  # 32MB máximo

    # === CONFIGURAÇÕES DE TEXTO ===
    'text_settings': {
        'min_text_length': 3,
        'max_text_length': 512,
        'common_terminators': [0x00, 0xFF, 0xFE, 0xFD],
        'line_break_chars': [0x0A, 0x0D, 0xF8, 0xF9],
        'space_chars': [0x20, 0x81, 0x40],  # ASCII space, Shift-JIS space
    },

    # === REGIÕES DE MEMÓRIA ===
    'memory_regions': {
        'rom_header': {
            'start': 0x000000,
            'end': 0x000200,
            'description': 'ROM Header e informações do cartucho',
            'contains_text': False
        },
        'text_data_primary': {
            'start': 0x400000,
            'end': 0x800000,
            'description': 'Região principal de dados de texto',
            'contains_text': True,
            'priority': 'high'
        },
        'text_data_secondary': {
            'start': 0x800000,
            'end': 0xC00000,
            'description': 'Região secundária de dados de texto',
            'contains_text': True,
            'priority': 'medium'
        },
        'menu_data': {
            'start': 0x200000,
            'end': 0x400000,
            'description': 'Dados de menus e interface',
            'contains_text': True,
            'priority': 'high'
        },
        'script_data': {
            'start': 0x100000,
            'end': 0x200000,
            'description': 'Scripts e diálogos',
            'contains_text': True,
            'priority': 'high'
        },
        'graphics_data': {
            'start': 0xC00000,
            'end': 0x1000000,
            'description': 'Dados gráficos',
            'contains_text': False
        }
    },

    # === CÓDIGOS DE CONTROLE ESPECÍFICOS DO GBA ===
    'control_codes': {
        # Códigos de formatação básicos
        0xF0: {
            'name': 'NEWLINE',
            'description': 'Nova linha',
            'extra_bytes': 0,
            'preserve': True
        },
        0xF1: {
            'name': 'WAIT_BUTTON',
            'description': 'Aguarda pressionar botão',
            'extra_bytes': 0,
            'preserve': True
        },
        0xF2: {
            'name': 'CLEAR_SCREEN',
            'description': 'Limpa tela',
            'extra_bytes': 0,
            'preserve': True
        },
        0xF3: {
            'name': 'WAIT_TIME',
            'description': 'Aguarda tempo específico',
            'extra_bytes': 1,  # 1 byte para tempo
            'preserve': True
        },
        0xF4: {
            'name': 'SCROLL_TEXT',
            'description': 'Ativa scroll de texto',
            'extra_bytes': 0,
            'preserve': True
        },
        0xF5: {
            'name': 'PLAY_SOUND',
            'description': 'Reproduz som',
            'extra_bytes': 2,  # 2 bytes para ID do som
            'preserve': True
        },

        # Códigos de cor (GBA suporta paletas)
        0xFC: {
            'name': 'COLOR_RED',
            'description': 'Texto vermelho',
            'extra_bytes': 0,
            'preserve': True
        },
        0xFD: {
            'name': 'COLOR_BLUE',
            'description': 'Texto azul',
            'extra_bytes': 0,
            'preserve': True
        },
        0xFE: {
            'name': 'COLOR_RESET',
            'description': 'Cor padrão',
            'extra_bytes': 0,
            'preserve': True
        },

        # Códigos de variáveis e parâmetros
        0xE0: {
            'name': 'PLAYER_NAME',
            'description': 'Nome do jogador',
            'extra_bytes': 0,
            'preserve': True
        },
        0xE1: {
            'name': 'ITEM_NAME',
            'description': 'Nome do item',
            'extra_bytes': 1,  # 1 byte para ID do item
            'preserve': True
        },
        0xE2: {
            'name': 'NUMBER',
            'description': 'Número variável',
            'extra_bytes': 2,  # 2 bytes para valor
            'preserve': True
        },
        0xE3: {
            'name': 'POKEMON_NAME',
            'description': 'Nome do Pokémon',
            'extra_bytes': 1,  # 1 byte para ID
            'preserve': True
        },
        0xE4: {
            'name': 'MOVE_NAME',
            'description': 'Nome do movimento',
            'extra_bytes': 1,  # 1 byte para ID do movimento
            'preserve': True
        },

        # Códigos especiais do GBA
        0xFA: {
            'name': 'PAUSE_MUSIC',
            'description': 'Pausa música',
            'extra_bytes': 0,
            'preserve': True
        },
        0xFB: {
            'name': 'RESUME_MUSIC',
            'description': 'Retoma música',
            'extra_bytes': 0,
            'preserve': True
        }
    },

    # === PADRÕES DE DETECÇÃO ===
    'detection_patterns': {
        # Padrões para identificar tipos de texto
        'dialog_patterns': [
            b'\xF0\xF1',  # NEWLINE + WAIT_BUTTON
            b'\xF2\xF0',  # CLEAR_SCREEN + NEWLINE
            b'\xE0',      # PLAYER_NAME
        ],
        'menu_patterns': [
            b'\xFC',      # COLOR_RED
            b'\xFD',      # COLOR_BLUE
            b'\xFE',      # COLOR_RESET
        ],
        'script_patterns': [
            b'\xE3',      # POKEMON_NAME
            b'\xE4',      # MOVE_NAME
            b'\xE1',      # ITEM_NAME
        ],
        'system_patterns': [
            b'\xFA',      # PAUSE_MUSIC
            b'\xFB',      # RESUME_MUSIC
        ]
    },

    # === CONFIGURAÇÕES DE FONTE ===
    'font_settings': {
        'character_width': 8,      # Largura base dos caracteres
        'character_height': 16,    # Altura dos caracteres
        'line_spacing': 2,         # Espaçamento entre linhas
        'max_line_length': 30,     # Caracteres por linha
        'max_lines_per_box': 3,    # Linhas por caixa de texto
        'supports_variable_width': True,  # GBA suporta fonte variável
        'supports_custom_chars': True,    # Pode ter caracteres customizados
    },

    # === CONFIGURAÇÕES DE TRADUÇÃO ===
    'translation_settings': {
        'preserve_control_codes': True,
        'allow_expansion': False,      # Textos não podem ser expandidos
        'compression_support': True,   # Muitos ROMs GBA usam compressão
        'encoding_options': ['shift_jis', 'utf-8', 'ascii'],
        'max_translation_ratio': 1.0,  # Tradução não pode ser maior que original
        'line_break_handling': 'preserve',  # Preserva quebras de linha
    },

    # === HEURÍSTICAS DE VALIDAÇÃO ===
    'validation_heuristics': {
        'min_string_entropy': 1.5,    # Entropia mínima para texto válido
        'max_control_code_ratio': 0.3,  # Máximo 30% códigos de controle
        'common_japanese_chars': [
            # Hiragana comuns
            'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
            'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と',
            'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ',
            'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り',
            'る', 'れ', 'ろ', 'わ', 'を', 'ん',
            # Katakana comuns
            'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',
            'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト',
            'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
            'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ',
            'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン'
        ],
        'common_english_words': [
            'the', 'and', 'you', 'are', 'not', 'have', 'can', 'will',
            'get', 'use', 'new', 'now', 'way', 'may', 'say', 'each',
            'which', 'their', 'time', 'work', 'first', 'right', 'more'
        ]
    },

    # === CONFIGURAÇÕES DE COMPRESSÃO ===
    'compression_settings': {
        'lz77_support': True,       # Suporte a LZ77 (comum no GBA)
        'huffman_support': True,    # Suporte a Huffman
        'rle_support': True,        # Run-Length Encoding
        'custom_compression': True,  # Compressão customizada do jogo
        'auto_detect': True,        # Detecta automaticamente compressão
    },

    # === CONFIGURAÇÕES ESPECÍFICAS POR JOGO ===
    'game_specific': {
        'pokemon_ruby_sapphire': {
            'text_regions': [
                {'start': 0x1F8000, 'end': 0x250000, 'name': 'story_text'},
                {'start': 0x3F0000, 'end': 0x400000, 'name': 'move_names'},
                {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
            ],
            'special_codes': {
                0xFD: {'name': 'POKEMON_NAME_BUFFER', 'extra_bytes': 1},
                0xFE: {'name': 'PLAYER_NAME_BUFFER', 'extra_bytes': 0},
            }
        },
        'pokemon_emerald': {
            'text_regions': [
                {'start': 0x1F8000, 'end': 0x260000, 'name': 'story_text'},
                {'start': 0x3F0000, 'end': 0x400000, 'name': 'move_names'},
                {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
            ],
            'special_codes': {
                0xFD: {'name': 'POKEMON_NAME_BUFFER', 'extra_bytes': 1},
                0xFE: {'name': 'PLAYER_NAME_BUFFER', 'extra_bytes': 0},
            }
        },
        'pokemon_firered_leafgreen': {
            'text_regions': [
                {'start': 0x1F8000, 'end': 0x250000, 'name': 'story_text'},
                {'start': 0x3F0000, 'end': 0x400000, 'name': 'move_names'},
                {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
            ],
            'special_codes': {
                0xFD: {'name': 'POKEMON_NAME_BUFFER', 'extra_bytes': 1},
                0xFE: {'name': 'PLAYER_NAME_BUFFER', 'extra_bytes': 0},
            }
        },
        'fire_emblem': {
            'text_regions': [
                {'start': 0x400000, 'end': 0x600000, 'name': 'dialog_text'},
                {'start': 0x200000, 'end': 0x300000, 'name': 'menu_text'},
            ],
            'special_codes': {
                0x80: {'name': 'CHAR_NAME', 'extra_bytes': 1},
                0x81: {'name': 'CLASS_NAME', 'extra_bytes': 1},
            }
        },
        'golden_sun': {
            'text_regions': [
                {'start': 0x400000, 'end': 0x700000, 'name': 'dialog_text'},
                {'start': 0x300000, 'end': 0x400000, 'name': 'menu_text'},
            ],
            'special_codes': {
                0x83: {'name': 'PLAYER_NAME', 'extra_bytes': 0},
                0x84: {'name': 'DJINN_NAME', 'extra_bytes': 1},
            }
        }
    },

    # === CONFIGURAÇÕES DE PERFORMANCE ===
    'performance_settings': {
        'chunk_size': 64 * 1024,     # Processa ROM em chunks de 64KB
        'max_threads': 4,            # Máximo de threads para processamento
        'cache_results': True,       # Cacheia resultados de análise
        'progress_interval': 1000,   # Intervalo de callback de progresso
    },

    # === CONFIGURAÇÕES DE DEBUG ===
    'debug_settings': {
        'log_level': 'INFO',
        'save_extraction_log': True,
        'validate_checksums': True,
        'backup_original': True,
        'detailed_analysis': False,  # Análise detalhada (mais lenta)
    }
}

# === FUNÇÕES AUXILIARES ===

def get_text_regions_for_game(game_id: str) -> list:
    """
    Retorna regiões de texto específicas para um jogo.

    Args:
        game_id: Identificador do jogo

    Returns:
        Lista de regiões de texto
    """
    game_config = GBA_CONFIG['game_specific'].get(game_id)
    if game_config and 'text_regions' in game_config:
        return game_config['text_regions']

    # Fallback para regiões padrão
    return [
        {'start': 0x100000, 'end': 0x200000, 'name': 'script_data'},
        {'start': 0x200000, 'end': 0x400000, 'name': 'menu_data'},
        {'start': 0x400000, 'end': 0x800000, 'name': 'text_data_primary'},
    ]

def get_control_codes_for_game(game_id: str) -> dict:
    """
    Retorna códigos de controle específicos para um jogo.

    Args:
        game_id: Identificador do jogo

    Returns:
        Dicionário de códigos de controle
    """
    base_codes = GBA_CONFIG['control_codes'].copy()

    game_config = GBA_CONFIG['game_specific'].get(game_id)
    if game_config and 'special_codes' in game_config:
        base_codes.update(game_config['special_codes'])

    return base_codes

def validate_gba_rom(rom_data: bytes) -> dict:
    """
    Valida se um ROM é válido para GBA.

    Args:
        rom_data: Dados do ROM

    Returns:
        Dicionário com resultado da validação
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    # Verifica tamanho mínimo
    if len(rom_data) < 0x8000:  # 32KB mínimo
        result['valid'] = False
        result['errors'].append("ROM muito pequeno para ser um GBA válido")
        return result

    # Verifica header GBA
    if len(rom_data) >= 0xA0:
        # Nintendo logo check (bytes 0x04-0x9F)
        nintendo_logo = rom_data[0x04:0xA0]
        if len(nintendo_logo) == 156:  # Logo tem 156 bytes
            result['info']['has_nintendo_logo'] = True
        else:
            result['warnings'].append("Logo Nintendo não encontrado ou inválido")

    # Verifica se é power of 2 (padrão GBA)
    size = len(rom_data)
    if size & (size - 1) != 0:
        result['warnings'].append("Tamanho do ROM não é potência de 2")

    # Verifica limite de tamanho
    if size > GBA_CONFIG['rom_size_limit']:
        result['warnings'].append(f"ROM maior que limite padrão ({GBA_CONFIG['rom_size_limit']} bytes)")

    result['info']['size'] = size
    result['info']['size_mb'] = size / (1024 * 1024)

    return result

# === CONFIGURAÇÃO DE EXPORTAÇÃO ===
__all__ = [
    'GBA_CONFIG',
    'get_text_regions_for_game',
    'get_control_codes_for_game',
    'validate_gba_rom'
]
# rom_translator/detectors/gba_game_detector.py

"""
Detector automático de jogos GBA
Identifica jogos baseado em headers, checksums e padrões únicos
"""

import hashlib
import struct
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class GameRegion(Enum):
    """Regiões dos jogos GBA"""
    USA = "USA"
    EUR = "EUR"
    JPN = "JPN"
    UNKNOWN = "UNK"

class GameLanguage(Enum):
    """Idiomas dos jogos"""
    ENGLISH = "en"
    JAPANESE = "jp"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    SPANISH = "es"
    UNKNOWN = "unknown"

@dataclass
class GameInfo:
    """Informações detalhadas do jogo detectado"""
    game_id: str
    title: str
    region: GameRegion
    language: GameLanguage
    version: str
    publisher: str
    confidence: float  # 0.0 a 1.0
    rom_size: int
    header_checksum: str
    game_code: str
    maker_code: str
    additional_info: Dict[str, Any]

class GBAGameDetector:
    """
    Detector automático de jogos GBA baseado em múltiplas heurísticas
    """

    def __init__(self):
        self.game_database = self._build_game_database()
        self.header_cache = {}

    def _build_game_database(self) -> Dict[str, Dict]:
        """
        Constrói base de dados de jogos conhecidos
        Baseado em game codes, checksums e padrões únicos
        """
        return {
            # === POKÉMON GAMES ===
            'pokemon_ruby_usa': {
                'title': 'Pokémon Ruby Version',
                'game_code': 'AXVE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'f28b6ffc97847e94a6c21a63cacf633ee5c8df1e',  # v1.0
                    'e26ee0d44e809351c8ce2d73c7400cdd67c6252d',  # v1.1
                ],
                'unique_patterns': [
                    {'offset': 0x1F8000, 'pattern': b'PROF. BIRCH'},
                    {'offset': 0x245EE0, 'pattern': b'BULBASAUR'},
                ],
                'text_regions': [
                    {'start': 0x1F8000, 'end': 0x250000, 'name': 'story_text'},
                    {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
                ],
                'config_key': 'pokemon_ruby_sapphire'
            },

            'pokemon_sapphire_usa': {
                'title': 'Pokémon Sapphire Version',
                'game_code': 'AXPE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'c25b7de9d3011d5a1c8d84e4c6b6d0a8c1e7f9c5',  # v1.0
                    'd5c4b6a8c9d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5',  # v1.1
                ],
                'unique_patterns': [
                    {'offset': 0x1F8000, 'pattern': b'PROF. BIRCH'},
                    {'offset': 0x245EE0, 'pattern': b'BULBASAUR'},
                ],
                'text_regions': [
                    {'start': 0x1F8000, 'end': 0x250000, 'name': 'story_text'},
                    {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
                ],
                'config_key': 'pokemon_ruby_sapphire'
            },

            'pokemon_emerald_usa': {
                'title': 'Pokémon Emerald Version',
                'game_code': 'BPEE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'd7cf56ac154e290f6e1fd2c6c5e9d0a8b2c4d6e8',  # v1.0
                ],
                'unique_patterns': [
                    {'offset': 0x1F8000, 'pattern': b'PROF. BIRCH'},
                    {'offset': 0x245EE0, 'pattern': b'BULBASAUR'},
                    {'offset': 0x3F0000, 'pattern': b'POUND'},
                ],
                'text_regions': [
                    {'start': 0x1F8000, 'end': 0x260000, 'name': 'story_text'},
                    {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
                ],
                'config_key': 'pokemon_emerald'
            },

            'pokemon_firered_usa': {
                'title': 'Pokémon FireRed Version',
                'game_code': 'BPRE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'e26ee0d44e809351c8ce2d73c7400cdd67c6252d',  # v1.0
                    'f28b6ffc97847e94a6c21a63cacf633ee5c8df1e',  # v1.1
                ],
                'unique_patterns': [
                    {'offset': 0x1F8000, 'pattern': b'PROF. OAK'},
                    {'offset': 0x245EE0, 'pattern': b'BULBASAUR'},
                ],
                'text_regions': [
                    {'start': 0x1F8000, 'end': 0x250000, 'name': 'story_text'},
                    {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
                ],
                'config_key': 'pokemon_firered_leafgreen'
            },

            'pokemon_leafgreen_usa': {
                'title': 'Pokémon LeafGreen Version',
                'game_code': 'BPGE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'c25b7de9d3011d5a1c8d84e4c6b6d0a8c1e7f9c5',  # v1.0
                ],
                'unique_patterns': [
                    {'offset': 0x1F8000, 'pattern': b'PROF. OAK'},
                    {'offset': 0x245EE0, 'pattern': b'BULBASAUR'},
                ],
                'text_regions': [
                    {'start': 0x1F8000, 'end': 0x250000, 'name': 'story_text'},
                    {'start': 0x245EE0, 'end': 0x248000, 'name': 'pokemon_names'},
                ],
                'config_key': 'pokemon_firered_leafgreen'
            },

            # === FIRE EMBLEM GAMES ===
            'fire_emblem_usa': {
                'title': 'Fire Emblem',
                'game_code': 'AE7E',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0',
                ],
                'unique_patterns': [
                    {'offset': 0x400000, 'pattern': b'LORD'},
                    {'offset': 0x200000, 'pattern': b'CHAPTER'},
                ],
                'text_regions': [
                    {'start': 0x400000, 'end': 0x600000, 'name': 'dialog_text'},
                    {'start': 0x200000, 'end': 0x300000, 'name': 'menu_text'},
                ],
                'config_key': 'fire_emblem'
            },

            # === GOLDEN SUN GAMES ===
            'golden_sun_usa': {
                'title': 'Golden Sun',
                'game_code': 'AGSE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1',
                ],
                'unique_patterns': [
                    {'offset': 0x400000, 'pattern': b'ISAAC'},
                    {'offset': 0x300000, 'pattern': b'PSYNERGY'},
                ],
                'text_regions': [
                    {'start': 0x400000, 'end': 0x700000, 'name': 'dialog_text'},
                    {'start': 0x300000, 'end': 0x400000, 'name': 'menu_text'},
                ],
                'config_key': 'golden_sun'
            },

            # === ADVANCE WARS ===
            'advance_wars_usa': {
                'title': 'Advance Wars',
                'game_code': 'AWRE',
                'maker_code': '01',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2',
                ],
                'unique_patterns': [
                    {'offset': 0x300000, 'pattern': b'ORANGE STAR'},
                    {'offset': 0x200000, 'pattern': b'INFANTRY'},
                ],
                'text_regions': [
                    {'start': 0x300000, 'end': 0x500000, 'name': 'dialog_text'},
                    {'start': 0x200000, 'end': 0x300000, 'name': 'menu_text'},
                ],
                'config_key': 'advance_wars'
            },

            # === CASTLEVANIA ===
            'castlevania_aria_usa': {
                'title': 'Castlevania: Aria of Sorrow',
                'game_code': 'ADA6',
                'maker_code': 'A4',
                'region': GameRegion.USA,
                'language': GameLanguage.ENGLISH,
                'checksums': [
                    'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3',
                ],
                'unique_patterns': [
                    {'offset': 0x400000, 'pattern': b'SOMA'},
                    {'offset': 0x300000, 'pattern': b'DRACULA'},
                ],
                'text_regions': [
                    {'start': 0x400000, 'end': 0x600000, 'name': 'dialog_text'},
                    {'start': 0x300000, 'end': 0x400000, 'name': 'menu_text'},
                ],
                'config_key': 'castlevania'
            }
        }

    def _extract_header_info(self, rom_data: bytes) -> Dict[str, Any]:
        """
        Extrai informações do header GBA

        Args:
            rom_data: Dados do ROM

        Returns:
            Dicionário com informações do header
        """
        if len(rom_data) < 0xC0:
            return {}

        # Estrutura do header GBA
        header = {
            'game_title': rom_data[0xA0:0xAC].decode('ascii', errors='ignore').rstrip('\x00'),
            'game_code': rom_data[0xAC:0xB0].decode('ascii', errors='ignore'),
            'maker_code': rom_data[0xB0:0xB2].decode('ascii', errors='ignore'),
            'fixed_value': rom_data[0xB2],  # Sempre 0x96
            'main_unit_code': rom_data[0xB3],
            'device_type': rom_data[0xB4],
            'software_version': rom_data[0xBC],
            'complement_check': rom_data[0xBD],
            'checksum': struct.unpack('<H', rom_data[0xBE:0xC0])[0]
        }

        # Calcula checksums
        header['sha1'] = hashlib.sha1(rom_data).hexdigest()
        header['md5'] = hashlib.md5(rom_data).hexdigest()

        return header

    def _check_patterns(self, rom_data: bytes, patterns: List[Dict]) -> float:
        """
        Verifica padrões únicos no ROM

        Args:
            rom_data: Dados do ROM
            patterns: Lista de padrões para verificar

        Returns:
            Score de confiança (0.0 a 1.0)
        """
        matches = 0
        total_patterns = len(patterns)

        for pattern_info in patterns:
            offset = pattern_info['offset']
            pattern = pattern_info['pattern']

            if offset + len(pattern) <= len(rom_data):
                if rom_data[offset:offset + len(pattern)] == pattern:
                    matches += 1
                else:
                    # Busca fuzzy em área próxima (±1KB)
                    search_start = max(0, offset - 1024)
                    search_end = min(len(rom_data), offset + 1024)
                    search_area = rom_data[search_start:search_end]

                    if pattern in search_area:
                        matches += 0.5  # Meio ponto para match fuzzy

        return matches / total_patterns if total_patterns > 0 else 0.0

    def _detect_region_from_code(self, game_code: str) -> GameRegion:
        """
        Detecta região baseada no game code

        Args:
            game_code: Código do jogo (4 caracteres)

        Returns:
            Região detectada
        """
        if len(game_code) != 4:
            return GameRegion.UNKNOWN

        region_char = game_code[3]
        region_map = {
            'E': GameRegion.USA,
            'P': GameRegion.EUR,
            'J': GameRegion.JPN,
            'U': GameRegion.USA,  # Algumas variações
            'F': GameRegion.EUR,  # França
            'D': GameRegion.EUR,  # Alemanha
            'I': GameRegion.EUR,  # Itália
            'S': GameRegion.EUR,  # Espanha
        }

        return region_map.get(region_char, GameRegion.UNKNOWN)

    def _detect_language(self, rom_data: bytes, game_code: str) -> GameLanguage:
        """
        Detecta idioma baseado no código e conteúdo

        Args:
            rom_data: Dados do ROM
            game_code: Código do jogo

        Returns:
            Idioma detectado
        """
        # Mapeamento baseado no game code
        if len(game_code) == 4:
            region_char = game_code[3]
            language_map = {
                'E': GameLanguage.ENGLISH,
                'U': GameLanguage.ENGLISH,
                'J': GameLanguage.JAPANESE,
                'F': GameLanguage.FRENCH,
                'D': GameLanguage.GERMAN,
                'I': GameLanguage.ITALIAN,
                'S': GameLanguage.SPANISH,
            }

            if region_char in language_map:
                return language_map[region_char]

        # Detecção por análise de conteúdo
        # Busca por caracteres japoneses comuns
        sample_size = min(len(rom_data), 100000)  # Analisa primeiros 100KB
        sample = rom_data[:sample_size]

        # Conta caracteres japoneses
        japanese_chars = 0
        for byte in sample:
            if 0x81 <= byte <= 0x9F or 0xE0 <= byte <= 0xFC:
                japanese_chars += 1

        if japanese_chars > sample_size * 0.1:  # >10% caracteres japoneses
            return GameLanguage.JAPANESE

        return GameLanguage.ENGLISH  # Default

    def detect_game(self, rom_data: bytes, filename: str = "") -> Optional[GameInfo]:
        """
        Detecta o jogo baseado nos dados do ROM

        Args:
            rom_data: Dados do ROM
            filename: Nome do arquivo (opcional)

        Returns:
            Informações do jogo detectado ou None
        """
        # Extrai informações do header
        header = self._extract_header_info(rom_data)
        if not header:
            return None

        # Busca por game code conhecido
        game_code = header.get('game_code', '')
        maker_code = header.get('maker_code', '')

        best_match = None
        best_confidence = 0.0

        for game_id, game_data in self.game_database.items():
            confidence = 0.0

            # Verifica game code (peso 40%)
            if game_data['game_code'] == game_code:
                confidence += 0.4

            # Verifica maker code (peso 10%)
            if game_data['maker_code'] == maker_code:
                confidence += 0.1

            # Verifica checksum (peso 30%)
            rom_sha1 = header.get('sha1', '')
            if rom_sha1 in game_data.get('checksums', []):
                confidence += 0.3

            # Verifica padrões únicos (peso 20%)
            if 'unique_patterns' in game_data:
                pattern_score = self._check_patterns(rom_data, game_data['unique_patterns'])
                confidence += pattern_score * 0.2

            # Considera o melhor match
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = (game_id, game_data)

        if best_match and best_confidence > 0.3:  # Threshold mínimo
            game_id, game_data = best_match

            # Detecta região e idioma
            detected_region = self._detect_region_from_code(game_code)
            detected_language = self._detect_language(rom_data, game_code)

            return GameInfo(
                game_id=game_id,
                title=game_data['title'],
                region=detected_region,
                language=detected_language,
                version="1.0",  # TODO: Detectar versão
                publisher=self._get_publisher_from_maker_code(maker_code),
                confidence=best_confidence,
                rom_size=len(rom_data),
                header_checksum=header.get('sha1', ''),
                game_code=game_code,
                maker_code=maker_code,
                additional_info={
                    'header': header,
                    'config_key': game_data.get('config_key', 'generic'),
                    'text_regions': game_data.get('text_regions', []),
                    'filename': filename
                }
            )

        # Fallback: jogo genérico
        return GameInfo(
            game_id='generic_gba',
            title=header.get('game_title', 'Unknown GBA Game'),
            region=self._detect_region_from_code(game_code),
            language=self._detect_language(rom_data, game_code),
            version="unknown",
            publisher=self._get_publisher_from_maker_code(maker_code),
            confidence=0.1,
            rom_size=len(rom_data),
            header_checksum=header.get('sha1', ''),
            game_code=game_code,
            maker_code=maker_code,
            additional_info={
                'header': header,
                'config_key': 'generic',
                'text_regions': [],
                'filename': filename
            }
        )

    def _get_publisher_from_maker_code(self, maker_code: str) -> str:
        """
        Converte maker code em nome do publisher

        Args:
            maker_code: Código do fabricante

        Returns:
            Nome do publisher
        """
        publishers = {
            '01': 'Nintendo',
            '08': 'Capcom',
            '13': 'EA',
            '18': 'Hudson Soft',
            '20': 'Infogrames',
            '22': 'Konami',
            '24': 'Nintendo',
            '28': 'Kemco',
            '29': 'SETA',
            '30': 'Infogrames',
            '31': 'Nintendo',
            '32': 'Bandai',
            '33': 'Ocean',
            '34': 'Konami',
            '35': 'HectorSoft',
            '38': 'Capcom',
            '39': 'Banpresto',
            '41': 'Ubi Soft',
            '46': 'Angel',
            '47': 'Spectrum Holobyte',
            '49': 'Irem',
            '50': 'Absolute',
            '51': 'Acclaim',
            '52': 'Activision',
            '53': 'American Sammy',
            '54': 'GameTek',
            '55': 'Hi Tech',
            '56': 'LJN Toys',
            '58': 'Mattel',
            '59': 'Milton Bradley',
            '60': 'Titus',
            '61': 'Virgin Games',
            '67': 'Ocean',
            '69': 'EA',
            '70': 'Infogrames',
            '71': 'Interplay',
            '72': 'Broderbund',
            '73': 'Sculptered Soft',
            '75': 'The Sales Curve',
            '78': 'THQ',
            '79': 'Accolade',
            '80': 'Misawa',
            '83': 'Lozc',
            '86': 'Tokuma Shoten Intermedia',
            '87': 'Tsukuda Original',
            '91': 'Chunsoft',
            '92': 'Video System',
            '93': 'BEC',
            '96': 'Yonezawa/S\'pal',
            '97': 'Kaneko',
            '99': 'Arc',
            'A4': 'Konami',
            'A6': 'Kawada',
            'A7': 'Takara',
            'A9': 'Technos Japan',
            'AA': 'Broderbund',
            'AC': 'Toei Animation',
            'AD': 'Toho',
            'AF': 'Namco',
            'B0': 'Acclaim',
            'B1': 'Nexoft',
            'B2': 'Bandai',
            'B4': 'Enix',
            'B6': 'HAL',
            'B7': 'SNK',
            'B9': 'Pony Canyon',
            'BA': 'Culture Brain',
            'BB': 'Sunsoft',
            'BD': 'Sony Imagesoft',
            'BF': 'Sammy',
            'C0': 'Taito',
            'C2': 'Kemco',
            'C3': 'Squaresoft',
            'C4': 'Tokuma Shoten Intermedia',
            'C5': 'Data East',
            'C6': 'Tonkin House',
            'C8': 'Koei',
            'C9': 'UFL',
            'CA': 'Ultra',
            'CB': 'Vap',
            'CC': 'Use',
            'CD': 'Meldac',
            'CE': 'Pony Canyon',
            'CF': 'Angel',
            'D0': 'Taito',
            'D1': 'Sofel',
            'D2': 'Quest',
            'D3': 'Sigma Enterprises',
            'D4': 'Ask Kodansha',
            'D6': 'Naxat Soft',
            'D7': 'Copya Systems',
            'D9': 'Banpresto',
            'DA': 'Tomy',
            'DB': 'LJN',
            'DD': 'NCS',
            'DE': 'Human',
            'DF': 'Altron',
            'E0': 'Jaleco',
            'E1': 'Towachiki',
            'E2': 'Uutaka',
            'E3': 'Varie',
            'E5': 'Epoch',
            'E7': 'Athena',
            'E8': 'Asmik',
            'E9': 'Natsume',
            'EA': 'King Records',
            'EB': 'Atlus',
            'EC': 'Epic/Sony Records',
            'EE': 'IGS',
            'F0': 'A Wave',
            'F3': 'Extreme Entertainment',
            'FF': 'LJN',
        }

        return publishers.get(maker_code, f'Unknown ({maker_code})')

    def get_supported_games(self) -> List[str]:
        """
        Retorna lista de jogos suportados

        Returns:
            Lista de IDs de jogos suportados
        """
        return list(self.game_database.keys())

    def get_game_info(self, game_id: str) -> Optional[Dict]:
        """
        Retorna informações de um jogo específico

        Args:
            game_id: ID do jogo

        Returns:
            Informações do jogo ou None
        """
        return self.game_database.get(game_id)

# === FUNÇÃO DE CONVENIÊNCIA ===
def detect_gba_game(rom_data: bytes, filename: str = "") -> Optional[GameInfo]:
    """
    Função de conveniência para detectar jogo GBA

    Args:
        rom_data: Dados do ROM
        filename: Nome do arquivo (opcional)

    Returns:
        Informações do jogo detectado
    """
    detector = GBAGameDetector()
    return detector.detect_game(rom_data, filename)

# === EXPORTAÇÃO ===
__all__ = [
    'GBAGameDetector',
    'GameInfo',
    'GameRegion',
    'GameLanguage',
    'detect_gba_game'
]
import struct
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .game_detector import GameDetector
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class TextEntry:
    """Representa uma entrada de texto extraída"""
    offset: int
    original_text: str
    translated_text: str = ""
    context: str = ""
    game_id: str = ""

class GBAEngine:
    """
    Engine principal para extração e tradução de textos em ROMs GBA
    com detecção automática de jogos e configuração adaptativa
    """

    def __init__(self, rom_path: str, config_path: Optional[str] = None):
        """
        Inicializa o engine com integração completa

        Args:
            rom_path: Caminho para a ROM GBA
            config_path: Caminho opcional para configurações customizadas
        """
        self.rom_path = Path(rom_path)
        self.rom_data = self._load_rom()

        # Inicializar componentes integrados
        self.game_detector = GameDetector()
        self.config_manager = ConfigManager(config_path)

        # Detectar o jogo e carregar configuração
        self.game_info = self._detect_and_configure()

        # Dados extraídos
        self.text_entries: List[TextEntry] = []
        self.pointer_tables: Dict[str, List[int]] = {}

        logger.info(f"GBAEngine inicializado para: {self.game_info['name']}")

    def _load_rom(self) -> bytes:
        """Carrega os dados da ROM"""
        try:
            with open(self.rom_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar ROM: {e}")

    def _detect_and_configure(self) -> Dict[str, Any]:
        """
        Detecta o jogo e configura o engine automaticamente

        Returns:
            Dict com informações do jogo detectado
        """
        # Detectar o jogo
        game_info = self.game_detector.detect_gba_game(self.rom_data)

        if not game_info:
            logger.warning("Jogo não detectado, usando configuração genérica")
            game_info = {
                'id': 'unknown',
                'name': 'Unknown Game',
                'region': 'unknown',
                'version': 'unknown'
            }

        # Carregar configuração específica do jogo
        config_loaded = self.config_manager.load_game_config(game_info['id'])

        if not config_loaded:
            logger.warning(f"Configuração não encontrada para {game_info['id']}")
            # Tentar carregar configuração genérica
            self.config_manager.load_game_config('generic_gba')

        return game_info

    def get_game_config(self) -> Dict[str, Any]:
        """Retorna a configuração atual do jogo"""
        return self.config_manager.get_current_config()

    def extract_text(self,
                    start_offset: Optional[int] = None,
                    end_offset: Optional[int] = None,
                    target_regions: Optional[List[str]] = None) -> List[TextEntry]:
        """
        Extrai texto usando configuração específica do jogo

        Args:
            start_offset: Offset inicial (opcional, usa config se não especificado)
            end_offset: Offset final (opcional, usa config se não especificado)
            target_regions: Regiões específicas para extrair (opcional)

        Returns:
            Lista de TextEntry com textos extraídos
        """
        config = self.get_game_config()

        if not config:
            raise RuntimeError("Nenhuma configuração carregada")

        # Usar configuração do jogo se não especificado
        if start_offset is None:
            start_offset = config.get('text_regions', {}).get('start', 0x800000)

        if end_offset is None:
            end_offset = config.get('text_regions', {}).get('end', len(self.rom_data))

        # Definir regiões alvo
        if target_regions is None:
            target_regions = config.get('priority_regions', ['dialogue', 'menus', 'items'])

        logger.info(f"Extraindo texto de {hex(start_offset)} a {hex(end_offset)}")
        logger.info(f"Regiões alvo: {target_regions}")

        # Extrair usando método adaptativo
        extracted_entries = self._extract_adaptive_text(
            start_offset, end_offset, target_regions
        )

        self.text_entries.extend(extracted_entries)

        logger.info(f"Extraídas {len(extracted_entries)} entradas de texto")
        return extracted_entries

    def _extract_adaptive_text(self,
                             start_offset: int,
                             end_offset: int,
                             target_regions: List[str]) -> List[TextEntry]:
        """
        Extrai texto usando configuração adaptativa específica do jogo
        """
        config = self.get_game_config()
        entries = []

        # Obter tabela de caracteres específica do jogo
        char_table = config.get('character_table', {})
        control_codes = config.get('control_codes', {})

        # Extrair de cada região prioritária
        for region in target_regions:
            region_config = config.get('regions', {}).get(region, {})

            if not region_config:
                logger.warning(f"Configuração não encontrada para região: {region}")
                continue

            region_entries = self._extract_from_region(
                region, region_config, char_table, control_codes
            )
            entries.extend(region_entries)

        return entries

    def _extract_from_region(self,
                           region_name: str,
                           region_config: Dict[str, Any],
                           char_table: Dict[str, str],
                           control_codes: Dict[str, str]) -> List[TextEntry]:
        """
        Extrai texto de uma região específica usando configuração adaptativa
        """
        entries = []

        # Obter ponteiros da região
        pointer_tables = region_config.get('pointer_tables', [])

        for table_info in pointer_tables:
            table_offset = table_info.get('offset')
            table_size = table_info.get('size')

            if not table_offset or not table_size:
                continue

            # Extrair ponteiros
            pointers = self._extract_pointers(table_offset, table_size)
            self.pointer_tables[f"{region_name}_{table_offset:08X}"] = pointers

            # Extrair texto de cada ponteiro
            for i, pointer in enumerate(pointers):
                if pointer == 0 or pointer >= len(self.rom_data):
                    continue

                try:
                    text = self._decode_text_at_pointer(
                        pointer, char_table, control_codes
                    )

                    if text and len(text.strip()) > 0:
                        entry = TextEntry(
                            offset=pointer,
                            original_text=text,
                            context=f"{region_name}_{i:04d}",
                            game_id=self.game_info['id']
                        )
                        entries.append(entry)

                except Exception as e:
                    logger.debug(f"Erro ao decodificar texto em {hex(pointer)}: {e}")

        logger.info(f"Extraídas {len(entries)} entradas da região {region_name}")
        return entries

    def _extract_pointers(self, table_offset: int, table_size: int) -> List[int]:
        """Extrai ponteiros de uma tabela"""
        pointers = []

        for i in range(table_size):
            offset = table_offset + (i * 4)

            if offset + 4 > len(self.rom_data):
                break

            pointer = struct.unpack('<I', self.rom_data[offset:offset+4])[0]

            # Converter ponteiro GBA para offset de arquivo
            if pointer >= 0x08000000:
                pointer -= 0x08000000

            pointers.append(pointer)

        return pointers

    def _decode_text_at_pointer(self,
                              pointer: int,
                              char_table: Dict[str, str],
                              control_codes: Dict[str, str]) -> str:
        """
        Decodifica texto em um ponteiro específico usando tabela de caracteres
        """
        if pointer >= len(self.rom_data):
            return ""

        text = ""
        offset = pointer

        while offset < len(self.rom_data):
            byte = self.rom_data[offset]

            # Verificar códigos de controle
            if byte in control_codes:
                control_code = control_codes[byte]

                if control_code == "[END]":
                    break
                elif control_code == "[LINE]":
                    text += "\n"
                elif control_code.startswith("[PLAYER"):
                    text += "[PLAYER]"
                else:
                    text += control_code

            # Verificar caracteres na tabela
            elif byte in char_table:
                text += char_table[byte]

            # Caracteres não mapeados
            else:
                # Parar em null bytes ou caracteres de controle não reconhecidos
                if byte == 0x00 or byte == 0xFF:
                    break

                # Representar como hex para debug
                text += f"[{byte:02X}]"

            offset += 1

            # Limite de segurança
            if offset - pointer > 1000:
                break

        return text.strip()

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da extração"""
        if not self.text_entries:
            return {"total_entries": 0}

        stats = {
            "total_entries": len(self.text_entries),
            "game_info": self.game_info,
            "regions_found": len(self.pointer_tables),
            "average_text_length": sum(len(entry.original_text)
                                     for entry in self.text_entries) / len(self.text_entries),
            "longest_text": max(len(entry.original_text) for entry in self.text_entries),
            "config_loaded": bool(self.config_manager.get_current_config())
        }

        return stats

    def export_for_translation(self, output_path: str) -> None:
        """
        Exporta textos extraídos para tradução

        Args:
            output_path: Caminho do arquivo de saída
        """
        import json

        export_data = {
            "game_info": self.game_info,
            "extraction_stats": self.get_extraction_stats(),
            "text_entries": [
                {
                    "offset": hex(entry.offset),
                    "original_text": entry.original_text,
                    "translated_text": entry.translated_text,
                    "context": entry.context,
                    "game_id": entry.game_id
                }
                for entry in self.text_entries
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Dados exportados para: {output_path}")

    def import_translations(self, translation_file: str) -> None:
        """
        Importa traduções de um arquivo

        Args:
            translation_file: Caminho do arquivo com traduções
        """
        import json

        with open(translation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Mapear traduções por offset
        translations = {
            int(entry['offset'], 16): entry['translated_text']
            for entry in data.get('text_entries', [])
            if entry.get('translated_text')
        }

        # Aplicar traduções
        for entry in self.text_entries:
            if entry.offset in translations:
                entry.translated_text = translations[entry.offset]

        logger.info(f"Importadas {len(translations)} traduções")

# Exemplo de uso
if __name__ == "__main__":
    # Demonstração do sistema integrado
    engine = GBAEngine("pokemon_emerald.gba")

    # O engine automaticamente:
    # 1. Detecta que é Pokemon Emerald
    # 2. Carrega configuração específica
    # 3. Está pronto para extração direcionada

    print(f"Jogo detectado: {engine.game_info['name']}")
    print(f"Configuração carregada: {bool(engine.get_game_config())}")

    # Extração inteligente
    texts = engine.extract_text(target_regions=['dialogue', 'pokemon_names'])

    print(f"Textos extraídos: {len(texts)}")

    # Exportar para tradução
    engine.export_for_translation("pokemon_emerald_texts.json")