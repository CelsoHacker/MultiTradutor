# rom_translator/engines/nes_engine.py
from typing import List, Optional, Set
import logging
from ..engines.base_engine import BaseEngine
from ..core.text_extractor import TextEntry
from ..utils.pointer_manager import PointerManager, PointerMode
from ..utils.game_detector import GameDetector
from ..utils.text_encoder import TextEncoder
from ..core.translation_engine import TranslationEngine
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class NESEngine(BaseEngine):
    """
    Motor de tradução específico para ROMs do Nintendo Entertainment System (NES).

    O NES usa uma arquitetura de 8-bit com endereçamento de 16-bit, onde:
    - Ponteiros são geralmente de 16-bit (2 bytes)
    - Texto pode estar em formato ASCII simples ou usando tabelas customizadas
    - Muitas ROMs usam terminadores nulos (0x00) para strings
    - Algumas ROMs mais complexas podem usar tabelas de ponteiros
    """

    def __init__(self, rom_path: str, config: ConfigManager, translation_engine: TranslationEngine):
        """
        Inicializa o motor NES.

        Args:
            rom_path: Caminho para a ROM NES
            config: Gerenciador de configurações
            translation_engine: Motor de tradução
        """
        self.console_name = "NES"
        self.supported_extensions = ['.nes']

        # Configurações específicas do NES
        self.nes_config = {
            'pointer_size': 2,  # Ponteiros de 16-bit
            'min_text_length': 4,  # Strings mínimas de 4 caracteres
            'max_text_length': 255,  # Limite prático para strings
            'common_terminators': [0x00, 0xFF],  # Terminadores comuns
            'ascii_range': (0x20, 0x7E),  # Range ASCII imprimível
            'scan_chunk_size': 0x1000,  # Chunks de 4KB para varredura
        }

        super().__init__(rom_path, config, translation_engine)

    def _initialize(self):
        """
        Configura o PointerManager e outras configurações específicas do NES.
        O NES tradicionalmente usa modo LINEAR devido à sua arquitetura simples.
        """
        logger.info(f"Inicializando NES engine para {self.rom_path}")

        # Configura o PointerManager para modo LINEAR (padrão do NES)
        self.pointer_manager = PointerManager(
            mode=PointerMode.LINEAR,
            rom_data=self.rom_data,
            pointer_size=self.nes_config['pointer_size']
        )

        # Configura o detector de jogos para procurar padrões específicos do NES
        self.game_detector = GameDetector(self.rom_data.data)

        # Configura o encoder de texto (padrão ASCII para começar)
        self.text_encoder = TextEncoder(encoding='ascii', errors='ignore')

        logger.info("NES engine inicializado com sucesso")

    def extract_text(self) -> List[TextEntry]:
        """
        Extrai texto da ROM NES usando múltiplas estratégias:
        1. Detecção automática de tabelas de ponteiros
        2. Varredura por strings ASCII terminadas em nulo
        3. Busca por padrões de texto específicos do NES

        Returns:
            Lista de TextEntry com o texto extraído
        """
        logger.info("Iniciando extração de texto da ROM NES")

        extracted_texts = []
        processed_addresses = set()

        # Estratégia 1: Usar GameDetector para encontrar tabelas de ponteiros
        try:
            pointer_tables = self.game_detector.find_pointer_tables(
                pointer_size=self.nes_config['pointer_size']
            )

            logger.info(f"Encontradas {len(pointer_tables)} tabelas de ponteiros")

            for table in pointer_tables:
                texts_from_table = self._extract_from_pointer_table(table)
                for text_entry in texts_from_table:
                    if text_entry.address not in processed_addresses:
                        extracted_texts.append(text_entry)
                        processed_addresses.add(text_entry.address)

        except Exception as e:
            logger.warning(f"Erro na detecção automática de ponteiros: {e}")

        # Estratégia 2: Varredura por strings ASCII simples
        try:
            ascii_texts = self._scan_ascii_strings()

            logger.info(f"Encontradas {len(ascii_texts)} strings ASCII")

            for text_entry in ascii_texts:
                if text_entry.address not in processed_addresses:
                    extracted_texts.append(text_entry)
                    processed_addresses.add(text_entry.address)

        except Exception as e:
            logger.warning(f"Erro na varredura ASCII: {e}")

        # Estratégia 3: Busca por padrões específicos do NES
        try:
            pattern_texts = self._scan_nes_patterns()

            logger.info(f"Encontradas {len(pattern_texts)} strings por padrões")

            for text_entry in pattern_texts:
                if text_entry.address not in processed_addresses:
                    extracted_texts.append(text_entry)
                    processed_addresses.add(text_entry.address)

        except Exception as e:
            logger.warning(f"Erro na busca por padrões: {e}")

        # Filtra e ordena os textos extraídos
        filtered_texts = self._filter_and_validate_texts(extracted_texts)

        logger.info(f"Extração concluída: {len(filtered_texts)} textos válidos encontrados")

        return filtered_texts

    def _extract_from_pointer_table(self, table_info: dict) -> List[TextEntry]:
        """
        Extrai textos de uma tabela de ponteiros específica.

        Args:
            table_info: Informações da tabela (start_address, count, etc.)

        Returns:
            Lista de TextEntry extraídos da tabela
        """
        texts = []
        table_start = table_info['start_address']
        pointer_count = table_info['count']

        logger.debug(f"Extraindo de tabela de ponteiros em 0x{table_start:04X} ({pointer_count} ponteiros)")

        for i in range(pointer_count):
            pointer_address = table_start + (i * self.nes_config['pointer_size'])

            # Lê o ponteiro
            if pointer_address + 1 < len(self.rom_data.data):
                # NES usa little-endian
                pointer_value = (self.rom_data.data[pointer_address + 1] << 8) | self.rom_data.data[pointer_address]

                # Converte para endereço de ROM se necessário
                rom_address = self._cpu_to_rom_address(pointer_value)

                if rom_address is not None and rom_address < len(self.rom_data.data):
                    text = self._read_null_terminated_string(rom_address)
                    if text and self._is_valid_text(text):
                        texts.append(TextEntry(
                            address=rom_address,
                            original_text=text,
                            context=f"pointer_table_{table_start:04X}_{i}"
                        ))

        return texts

    def _scan_ascii_strings(self) -> List[TextEntry]:
        """
        Faz varredura por strings ASCII terminadas em nulo na ROM.

        Returns:
            Lista de TextEntry com strings ASCII encontradas
        """
        texts = []
        rom_data = self.rom_data.data

        i = 0
        while i < len(rom_data):
            # Procura por início de string ASCII válida
            if self._is_ascii_char(rom_data[i]):
                start_pos = i
                text_bytes = []

                # Lê até encontrar terminador ou caractere inválido
                while i < len(rom_data) and not self._is_string_terminator(rom_data[i]):
                    if self._is_ascii_char(rom_data[i]):
                        text_bytes.append(rom_data[i])
                        i += 1
                    else:
                        break

                # Verifica se encontrou uma string válida
                if len(text_bytes) >= self.nes_config['min_text_length']:
                    try:
                        text = bytes(text_bytes).decode('ascii', errors='ignore')
                        if self._is_valid_text(text):
                            texts.append(TextEntry(
                                address=start_pos,
                                original_text=text,
                                context="ascii_scan"
                            ))
                    except UnicodeDecodeError:
                        pass

            i += 1

        return texts

    def _scan_nes_patterns(self) -> List[TextEntry]:
        """
        Busca por padrões específicos comuns em ROMs NES.

        Returns:
            Lista de TextEntry com textos encontrados por padrões
        """
        texts = []
        rom_data = self.rom_data.data

        # Padrão 1: Busca por strings precedidas por byte de tamanho
        for i in range(len(rom_data) - 1):
            if rom_data[i] > 0 and rom_data[i] < 128:  # Possível byte de tamanho
                string_length = rom_data[i]
                start_pos = i + 1

                if start_pos + string_length < len(rom_data):
                    text_bytes = rom_data[start_pos:start_pos + string_length]

                    # Verifica se são caracteres válidos
                    if all(self._is_ascii_char(b) for b in text_bytes):
                        try:
                            text = bytes(text_bytes).decode('ascii', errors='ignore')
                            if self._is_valid_text(text):
                                texts.append(TextEntry(
                                    address=start_pos,
                                    original_text=text,
                                    context="length_prefixed"
                                ))
                        except UnicodeDecodeError:
                            pass

        # Padrão 2: Busca por sequências de controle específicas do NES
        # (Este é um exemplo - pode ser expandido baseado em jogos específicos)
        control_sequences = [0xFE, 0xFD, 0xFC]  # Códigos de controle comuns

        for control_byte in control_sequences:
            i = 0
            while i < len(rom_data):
                if rom_data[i] == control_byte and i + 1 < len(rom_data):
                    # Procura por texto após o código de controle
                    start_pos = i + 1
                    text = self._read_null_terminated_string(start_pos)

                    if text and self._is_valid_text(text):
                        texts.append(TextEntry(
                            address=start_pos,
                            original_text=text,
                            context=f"control_sequence_{control_byte:02X}"
                        ))

                i += 1

        return texts

    def _read_null_terminated_string(self, start_address: int) -> Optional[str]:
        """
        Lê uma string terminada em nulo a partir do endereço especificado.

        Args:
            start_address: Endereço inicial

        Returns:
            String lida ou None se inválida
        """
        if start_address >= len(self.rom_data.data):
            return None

        text_bytes = []
        i = start_address

        while i < len(self.rom_data.data) and i < start_address + self.nes_config['max_text_length']:
            byte = self.rom_data.data[i]

            if self._is_string_terminator(byte):
                break
            elif self._is_ascii_char(byte):
                text_bytes.append(byte)
            else:
                # Caractere inválido - interrompe a leitura
                break

            i += 1

        if len(text_bytes) >= self.nes_config['min_text_length']:
            try:
                return bytes(text_bytes).decode('ascii', errors='ignore')
            except UnicodeDecodeError:
                return None

        return None

    def _is_ascii_char(self, byte: int) -> bool:
        """Verifica se o byte é um caractere ASCII imprimível."""
        return self.nes_config['ascii_range'][0] <= byte <= self.nes_config['ascii_range'][1]

    def _is_string_terminator(self, byte: int) -> bool:
        """Verifica se o byte é um terminador de string."""
        return byte in self.nes_config['common_terminators']

    def _is_valid_text(self, text: str) -> bool:
        """
        Valida se o texto extraído é realmente texto traduzível.

        Args:
            text: Texto a ser validado

        Returns:
            True se o texto é válido para tradução
        """
        if not text or len(text) < self.nes_config['min_text_length']:
            return False

        # Verifica se contém caracteres alfabéticos
        if not any(c.isalpha() for c in text):
            return False

        # Verifica se não é só números ou símbolos
        if text.isdigit() or all(not c.isalnum() for c in text):
            return False

        # Verifica se não contém muitos caracteres de controle
        control_chars = sum(1 for c in text if ord(c) < 32)
        if control_chars > len(text) * 0.1:  # Mais de 10% são caracteres de controle
            return False

        return True

    def _cpu_to_rom_address(self, cpu_address: int) -> Optional[int]:
        """
        Converte endereço de CPU para endereço de ROM.

        O NES usa mapeamento de memória complexo, mas para simplicidade
        inicial, assumimos mapeamento linear básico.

        Args:
            cpu_address: Endereço da CPU (16-bit)

        Returns:
            Endereço correspondente na ROM ou None se inválido
        """
        # Mapeamento básico - pode ser refinado para diferentes mappers
        if 0x8000 <= cpu_address <= 0xFFFF:
            # Área típica de ROM
            rom_address = cpu_address - 0x8000
            if rom_address < len(self.rom_data.data):
                return rom_address

        return None

    def _filter_and_validate_texts(self, texts: List[TextEntry]) -> List[TextEntry]:
        """
        Filtra e valida a lista final de textos extraídos.

        Args:
            texts: Lista de textos extraídos

        Returns:
            Lista filtrada e validada
        """
        # Remove duplicatas baseadas no endereço
        unique_texts = {}
        for text in texts:
            if text.address not in unique_texts:
                unique_texts[text.address] = text

        # Ordena por endereço
        sorted_texts = sorted(unique_texts.values(), key=lambda x: x.address)

        # Aplica filtros adicionais
        filtered_texts = []
        for text in sorted_texts:
            if self._is_valid_text(text.original_text):
                filtered_texts.append(text)

        return filtered_texts

    def get_console_info(self) -> dict:
        """
        Retorna informações específicas sobre o console NES.

        Returns:
            Dicionário com informações do console
        """
        return {
            'name': self.console_name,
            'architecture': '8-bit',
            'pointer_size': self.nes_config['pointer_size'],
            'supported_extensions': self.supported_extensions,
            'memory_model': 'banked',
            'typical_encoding': 'ASCII',
            'common_terminators': self.nes_config['common_terminators']
        }