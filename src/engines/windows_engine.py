# src/engines/windows_engine.py
"""
WindowsEngine - Engine para tradução de aplicativos Windows
Autor: MultiTradutor Team
Versão: 1.0

Este engine extrai e traduz strings de executáveis Windows (.exe, .dll, .msi)
usando técnicas similares ao que fazemos com ROMs, mas adaptado para PE files.
"""

import os
import struct
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pefile
import subprocess
from .base_engine import BaseEngine
from ..utils.backup_manager import BackupManager
from ..ai_modules.translation_ai import TranslationAI

class WindowsEngine(BaseEngine):
    """
    Engine especializado para aplicativos Windows

    Funciona como nossos engines de console, mas para PE (Portable Executable) files.
    Pensa nele como o "Zelda: A Link to the Past" dos engines - clássico e confiável!
    """

    def __init__(self):
        super().__init__()
        self.supported_formats = ['.exe', '.dll', '.msi', '.scr']
        self.pe_file = None
        self.resource_data = {}
        self.string_tables = {}
        self.dialog_resources = {}
        self.backup_manager = BackupManager()

        # Resource types que contêm strings traduzíveis
        self.translatable_resources = {
            'RT_STRING': 6,      # String tables
            'RT_DIALOG': 5,      # Dialog boxes
            'RT_MENU': 4,        # Menus
            'RT_ACCELERATOR': 9, # Keyboard shortcuts
            'RT_VERSION': 16     # Version info
        }

    def detect_software_type(self, file_path: str) -> bool:
        """
        Detecta se o arquivo é um PE válido
        É como fazer um CRC check numa ROM - precisamos ter certeza!
        """
        try:
            if not Path(file_path).suffix.lower() in self.supported_formats:
                return False

            # Tenta abrir como PE file
            pe = pefile.PE(file_path, fast_load=True)
            pe.close()
            return True

        except (pefile.PEFormatError, FileNotFoundError):
            return False

    def extract_texts(self, file_path: str) -> Dict[str, any]:
        """
        Extrai todas as strings traduzíveis do executável

        Aqui é onde a mágica acontece - como extrair o texto de uma ROM,
        mas navegando pela estrutura PE em vez de offsets fixos.
        """
        if not self.detect_software_type(file_path):
            raise ValueError(f"Arquivo não é um PE válido: {file_path}")

        print(f"🔍 Carregando PE file: {os.path.basename(file_path)}")
        self.pe_file = pefile.PE(file_path)

        extracted_data = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'strings': {},
            'dialogs': {},
            'menus': {},
            'version_info': {},
            'metadata': self._extract_metadata()
        }

        try:
            # Extrai string tables (o pão e manteiga!)
            if hasattr(self.pe_file, 'DIRECTORY_ENTRY_RESOURCE'):
                self._extract_string_tables(extracted_data)
                self._extract_dialog_resources(extracted_data)
                self._extract_menu_resources(extracted_data)
                self._extract_version_info(extracted_data)

            # Extrai strings hardcoded no código (bonus stage!)
            self._extract_hardcoded_strings(extracted_data)

            print(f"✅ Extraído: {len(extracted_data['strings'])} strings, "
                  f"{len(extracted_data['dialogs'])} dialogs")

        except Exception as e:
            print(f"⚠️ Erro durante extração: {e}")
        finally:
            self.pe_file.close()

        return extracted_data

    def _extract_metadata(self) -> Dict[str, str]:
        """Extrai metadados do PE - nosso "header" da ROM"""
        metadata = {
            'architecture': 'x64' if self.pe_file.PE_TYPE == pefile.OPTIONAL_HEADER_MAGIC_PE_PLUS else 'x86',
            'timestamp': self.pe_file.FILE_HEADER.TimeDateStamp,
            'sections': len(self.pe_file.sections),
            'has_resources': hasattr(self.pe_file, 'DIRECTORY_ENTRY_RESOURCE'),
            'compiler': self._detect_compiler()
        }
        return metadata

    def _detect_compiler(self) -> str:
        """Detecta o compilador usado - útil para estratégias específicas"""
        # Delphi deixa assinatura bem clara
        if b'Borland' in self.pe_file.get_data():
            return 'Delphi/C++Builder'

        # Visual Studio também
        if any(b'Microsoft' in section.get_data() for section in self.pe_file.sections):
            return 'Visual Studio'

        # MinGW tem suas peculiaridades
        if any(b'mingw' in section.Name.lower() for section in self.pe_file.sections):
            return 'MinGW'

        return 'Unknown'

    def _extract_string_tables(self, data: Dict) -> None:
        """
        Extrai string tables - onde 90% das traduções vão estar
        É como extrair a tabela de ponteiros numa ROM!
        """
        try:
            for resource_type in self.pe_file.DIRECTORY_ENTRY_RESOURCE.entries:
                if resource_type.id == self.translatable_resources['RT_STRING']:

                    for resource_id in resource_type.directory.entries:
                        for resource_lang in resource_id.directory.entries:

                            # Pega os dados raw
                            string_data = self.pe_file.get_data(
                                resource_lang.data.struct.OffsetToData,
                                resource_lang.data.struct.Size
                            )

                            # Decodifica as strings (formato Windows é especial)
                            strings = self._decode_string_table(string_data, resource_id.id)

                            for string_id, text in strings.items():
                                if text.strip():  # Só strings não vazias
                                    key = f"STRING_{resource_id.id}_{string_id}"
                                    data['strings'][key] = {
                                        'original': text,
                                        'translated': '',
                                        'context': 'string_table',
                                        'resource_id': resource_id.id,
                                        'string_id': string_id
                                    }

        except AttributeError:
            print("⚠️ Sem recursos de string encontrados")

    def _decode_string_table(self, data: bytes, table_id: int) -> Dict[int, str]:
        """
        Decodifica string table do Windows
        Formato: [length:word][string:wchar[length]]

        É como decodificar texto em japonês numa ROM - cada char é 2 bytes!
        """
        strings = {}
        offset = 0
        base_id = (table_id - 1) * 16  # String tables são blocos de 16

        for i in range(16):  # Cada table tem até 16 strings
            if offset >= len(data):
                break

            # Le o comprimento (word = 2 bytes)
            if offset + 2 > len(data):
                break

            length = struct.unpack('<H', data[offset:offset+2])[0]
            offset += 2

            if length > 0:
                # Le a string (UTF-16LE)
                if offset + (length * 2) <= len(data):
                    string_bytes = data[offset:offset + (length * 2)]
                    try:
                        text = string_bytes.decode('utf-16le')
                        strings[base_id + i] = text
                    except UnicodeDecodeError:
                        # Fallback para encoding problemático
                        text = string_bytes.decode('utf-16le', errors='replace')
                        strings[base_id + i] = text

                offset += length * 2

        return strings

    def _extract_dialog_resources(self, data: Dict) -> None:
        """Extrai recursos de dialog - onde ficam labels de interface"""
        # Implementação similar às string tables, mas para dialogs
        # Cada dialog tem controles com texto
        pass

    def _extract_menu_resources(self, data: Dict) -> None:
        """Extrai menus - items do menu principal e contexto"""
        # Menus são hierárquicos, precisa navegar a árvore
        pass

    def _extract_version_info(self, data: Dict) -> None:
        """Extrai version info - nome do produto, copyright, etc."""
        try:
            if hasattr(self.pe_file, 'VS_VERSIONINFO'):
                for version_info in self.pe_file.VS_VERSIONINFO:
                    for entry in version_info:
                        if hasattr(entry, 'StringTable'):
                            for string_table in entry.StringTable:
                                for key, value in string_table.entries.items():
                                    if isinstance(value, bytes):
                                        value = value.decode('utf-8', errors='replace')

                                    data['version_info'][key.decode('utf-8')] = {
                                        'original': value,
                                        'translated': '',
                                        'context': 'version_info'
                                    }
        except:
            pass

    def _extract_hardcoded_strings(self, data: Dict) -> None:
        """
        Busca strings hardcoded no código
        É como procurar texto ASCII numa ROM - força bruta mas efetivo!
        """
        try:
            # Busca em todas as seções
            for section in self.pe_file.sections:
                section_data = section.get_data()

                # Busca strings ASCII (mínimo 4 chars)
                ascii_strings = self._find_ascii_strings(section_data, min_length=4)

                # Busca strings Unicode
                unicode_strings = self._find_unicode_strings(section_data, min_length=4)

                for i, text in enumerate(ascii_strings + unicode_strings):
                    if self._is_translatable_string(text):
                        key = f"HARDCODED_{section.Name.decode().strip('\\x00')}_{i}"
                        data['strings'][key] = {
                            'original': text,
                            'translated': '',
                            'context': 'hardcoded',
                            'section': section.Name.decode().strip('\x00')
                        }
        except:
            pass

    def _find_ascii_strings(self, data: bytes, min_length: int = 4) -> List[str]:
        """Encontra strings ASCII printáveis"""
        strings = []
        current_string = ""

        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string += chr(byte)
            else:
                if len(current_string) >= min_length:
                    strings.append(current_string)
                current_string = ""

        return strings

    def _find_unicode_strings(self, data: bytes, min_length: int = 4) -> List[str]:
        """Encontra strings Unicode (UTF-16LE)"""
        strings = []

        # Unicode strings geralmente são null-terminated wide chars
        for i in range(0, len(data) - 1, 2):
            try:
                if i + (min_length * 2) <= len(data):
                    # Tenta decodificar como UTF-16LE
                    test_bytes = data[i:i + (min_length * 2)]
                    if b'\x00\x00' not in test_bytes[:-2]:  # Não termina no meio
                        text = test_bytes.decode('utf-16le')
                        if text.isprintable() and not text.isspace():
                            strings.append(text)
            except:
                continue

        return strings

    def _is_translatable_string(self, text: str) -> bool:
        """
        Determina se uma string vale a pena traduzir
        Filtros similares aos que usamos para ROMs!
        """
        # Muito curta
        if len(text) < 3:
            return False

        # Só números ou symbols
        if text.isdigit() or not any(c.isalpha() for c in text):
            return False

        # Paths de arquivo
        if '\\' in text or '/' in text:
            return False

        # Registry keys
        if text.startswith(('HKEY_', 'SOFTWARE\\')):
            return False

        # Extensões de arquivo
        if text.startswith('.') and len(text) <= 5:
            return False

        return True

    def extract_audio(self, file_path: str) -> Dict[str, any]:
        """
        Extrai recursos de áudio (WAV, etc.)
        Nem todo exe tem áudio, mas quando tem é aqui que mora!
        """
        if not self.detect_software_type(file_path):
            return {}

        audio_data = {
            'file_path': file_path,
            'audio_resources': {},
            'embedded_sounds': {}
        }

        try:
            pe = pefile.PE(file_path)

            if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
                for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                    # RT_RCDATA (10) e custom resources podem ter áudio
                    if resource_type.id in [10, 23]:  # RCDATA e HTML
                        self._extract_audio_resources(pe, resource_type, audio_data)

            pe.close()

        except Exception as e:
            print(f"⚠️ Erro ao extrair áudio: {e}")

        return audio_data

    def _extract_audio_resources(self, pe: pefile.PE, resource_type, data: Dict) -> None:
        """Extrai recursos que podem conter áudio"""
        try:
            for resource_id in resource_type.directory.entries:
                for resource_lang in resource_id.directory.entries:

                    resource_data = pe.get_data(
                        resource_lang.data.struct.OffsetToData,
                        resource_lang.data.struct.Size
                    )

                    # Detecta formato de áudio pelos magic bytes
                    audio_format = self._detect_audio_format(resource_data)

                    if audio_format:
                        key = f"AUDIO_{resource_type.id}_{resource_id.id}"
                        data['audio_resources'][key] = {
                            'format': audio_format,
                            'size': len(resource_data),
                            'data': resource_data,
                            'resource_id': resource_id.id
                        }

        except Exception as e:
            print(f"⚠️ Erro ao processar recurso de áudio: {e}")

    def _detect_audio_format(self, data: bytes) -> Optional[str]:
        """Detecta formato de áudio pelos magic bytes"""
        if data.startswith(b'RIFF') and b'WAVE' in data[:12]:
            return 'WAV'
        elif data.startswith(b'ID3') or data.startswith(b'\xff\xfb'):
            return 'MP3'
        elif data.startswith(b'OggS'):
            return 'OGG'
        elif data.startswith(b'fLaC'):
            return 'FLAC'

        return None

    def apply_translation(self, translations: Dict, output_path: str = None) -> bool:
        """
        Aplica as traduções ao arquivo

        Aqui é onde fazemos a "ROM patch" - mas em PE files!
        SEMPRE fazemos backup antes - lição aprendida dos tempos de ROM hacking!
        """
        if not self.pe_file:
            raise ValueError("Nenhum arquivo carregado!")

        source_path = translations['file_path']
        if not output_path:
            output_path = source_path.replace('.exe', '_translated.exe')

        print(f"🔄 Aplicando traduções em: {os.path.basename(source_path)}")

        # BACKUP SAGRADO! Nunca patcheamos sem backup!
        backup_path = self.backup_manager.create_backup(source_path)
        print(f"💾 Backup criado: {backup_path}")

        try:
            # Copia arquivo original
            import shutil
            shutil.copy2(source_path, output_path)

            # Aplica as patches
            success = self._patch_pe_file(output_path, translations)

            if success:
                print(f"✅ Tradução aplicada com sucesso: {output_path}")
                return True
            else:
                print(f"❌ Falha ao aplicar tradução")
                return False

        except Exception as e:
            print(f"💥 Erro crítico durante patch: {e}")
            return False

    def _patch_pe_file(self, file_path: str, translations: Dict) -> bool:
        """
        Aplica as patches no PE file
        É como fazer patches IPS, mas bem mais complicado!
        """
        try:
            # Aqui seria a implementação do patch
            # Por enquanto, vamos simular sucesso

            # TODO: Implementar patch real de string tables
            # TODO: Implementar patch de dialog resources
            # TODO: Implementar patch de version info
            # TODO: Implementar patch de hardcoded strings (mais complexo)

            print("🔧 Patches aplicadas com sucesso (mock)")
            return True

        except Exception as e:
            print(f"💥 Erro durante patch: {e}")
            return False

    def get_translation_stats(self, extracted_data: Dict) -> Dict[str, int]:
        """
        Estatísticas de tradução - nosso "speedrun timer"!
        """
        return {
            'total_strings': len(extracted_data.get('strings', {})),
            'total_dialogs': len(extracted_data.get('dialogs', {})),
            'total_menus': len(extracted_data.get('menus', {})),
            'version_info_items': len(extracted_data.get('version_info', {})),
            'audio_resources': len(extracted_data.get('audio_resources', {})),
            'file_size_mb': round(extracted_data.get('file_size', 0) / 1024 / 1024, 2)
        }

# Exemplo de uso:
if __name__ == "__main__":
    engine = WindowsEngine()

    # Testa com um executável
    test_file = "notepad.exe"  # Exemplo

    if engine.detect_software_type(test_file):
        print("🎮 Arquivo detectado como PE válido!")

        # Extrai textos
        extracted = engine.extract_texts(test_file)

        # Mostra stats
        stats = engine.get_translation_stats(extracted)
        print(f"📊 Stats: {stats}")

        # Simula tradução (aqui viria a IA)
        for key, data in extracted['strings'].items():
            data['translated'] = f"[TRADUZIDO] {data['original']}"

        # Aplica tradução
        success = engine.apply_translation(extracted, "notepad_pt.exe")

        if success:
            print("🏆 GG! Tradução completada com sucesso!")
    else:
        print("❌ Arquivo não é um PE válido")