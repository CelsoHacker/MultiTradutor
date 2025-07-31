import logging
from pathlib import Path
from typing import List, Optional

try:
    from .base_engine import EnhancedBaseEngine, EnhancedTextEntry
except ImportError:
    # Ajuste para testes locais se necessário
    from base_engine import EnhancedBaseEngine, EnhancedTextEntry

logger = logging.getLogger(__name__)

class DreamcastEngine(EnhancedBaseEngine):
    CONSOLE_NAME = "Sega Dreamcast"
    SUPPORTED_EXTENSIONS = [".gdi", ".cdi", ".bin", ".iso"]

    def __init__(self):
        super().__init__("dreamcast")

    def get_rom_info(self) -> dict:
        return {
            "console": self.CONSOLE_NAME,
            "extensões": self.SUPPORTED_EXTENSIONS,
            "suporte_texto": True,
            "observações": "Este módulo é genérico para ROMs de Dreamcast. Pode exigir ajustes por jogo."
        }

    def load_rom(self, file_path: str) -> bool:
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Arquivo não encontrado: {file_path}")
                return False

            with open(file_path, "rb") as f:
                self.rom_data = f.read()

            logger.info(f"ROM Dreamcast carregada: {path.name} ({len(self.rom_data)} bytes)")
            return True
        except Exception as e:
            logger.exception(f"Erro ao carregar a ROM: {e}")
            return False

    def extract_text_regions(self) -> List[EnhancedTextEntry]:
        logger.info("Iniciando extração de texto da ROM Dreamcast...")
        entries = []

        # TODO: Implementar análise real para o jogo-alvo
        # Aqui é um exemplo de placeholder
        example_offset = 0x10000
        example_length = 64

        text_chunk = self.rom_data[example_offset:example_offset + example_length]
        decoded = self.decode_text(text_chunk)

        entry = EnhancedTextEntry(
            offset=example_offset,
            original_text=decoded,
            translated_text="",
            notes="Texto de exemplo (ajuste manual por jogo)"
        )
        entries.append(entry)

        logger.info(f"{len(entries)} regiões de texto extraídas (exemplo).")
        return entries

    def insert_translated_text(self, entries: List[EnhancedTextEntry]) -> bool:
        logger.info("Inserindo textos traduzidos na ROM Dreamcast...")
        rom_bytes = bytearray(self.rom_data)

        for entry in entries:
            encoded = self.encode_text(entry.translated_text)
            offset = entry.offset

            # Evitar sobrescrever além do tamanho original
            max_len = len(entry.original_text.encode("utf-8"))
            encoded = encoded[:max_len]

            rom_bytes[offset:offset + len(encoded)] = encoded

        self.rom_data = bytes(rom_bytes)
        logger.info("Textos inseridos com sucesso.")
        return True

    def save_rom(self, output_path: str) -> bool:
        try:
            with open(output_path, "wb") as f:
                f.write(self.rom_data)
            logger.info(f"ROM Dreamcast salva em: {output_path}")
            return True
        except Exception as e:
            logger.exception(f"Erro ao salvar ROM: {e}")
            return False

    def decode_text(self, data: bytes) -> str:
        try:
            # Exemplo genérico — cada jogo pode ter codificação diferente
            return data.decode("ascii", errors="ignore")
        except Exception as e:
            logger.warning(f"Falha na decodificação: {e}")
            return ""

    def encode_text(self, text: str) -> bytes:
        try:
            return text.encode("ascii", errors="replace")
        except Exception as e:
            logger.warning(f"Falha na codificação: {e}")
            return b""
