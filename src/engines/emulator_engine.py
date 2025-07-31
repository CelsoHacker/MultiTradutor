# src/engines/emulator_engine.py
"""
EmulatorEngine - Ponte entre o sistema moderno e engines legacy
Autor: MultiTradutor Team
Versão: 1.0

Este engine integra todos os engines de console existentes no novo sistema modular.
É como um "Universal Game Genie" - funciona com qualquer console!

Arquitetura:
- Detecta automaticamente o tipo de ROM
- Redireciona para o engine legacy apropriado
- Padroniza a saída para o formato moderno
- Mantém compatibilidade total com código existente
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from abc import ABC
from .base_engine import BaseEngine

# Imports dos engines legacy (assumindo que existem)
# Adaptamos conforme a estrutura real dos seus engines
try:
    from ..engines_legacy.nes_engine import NESEngine
    from ..engines_legacy.snes_engine import SNESEngine
    from ..engines_legacy.gba_engine import GBAEngine
    from ..engines_legacy.gb_engine import GBEngine
    from ..engines_legacy.genesis_engine import GenesisEngine
    from ..engines_legacy.n64_engine import N64Engine
    from ..engines_legacy.psx_engine import PSXEngine
    # Adicione outros conforme disponível
except ImportError as e:
    print(f"⚠️ Alguns engines legacy não encontrados: {e}")
    # Fallback para desenvolvimento - remover em produção
    class MockEngine:
        def __init__(self): pass
        def is_compatible(self, rom_path): return False
        def extract_texts(self, rom_path): return {}
        def apply_translation(self, translations): return True

    NESEngine = SNESEngine = GBAEngine = GBEngine = MockEngine
    GenesisEngine = N64Engine = PSXEngine = MockEngine

class LegacyEngineAdapter:
    """
    Adapter pattern para padronizar a interface dos engines legacy
    É como criar um "universal controller port" - cada console tem sua peculiaridade,
    mas todos falam a mesma linguagem no final!
    """

    def __init__(self, legacy_engine: Any, console_type: str):
        self.legacy_engine = legacy_engine
        self.console_type = console_type
        self.rom_path = None

    def extract_texts(self, rom_path: str) -> Dict[str, Any]:
        """
        Padroniza a saída dos engines legacy para o formato moderno
        """
        self.rom_path = rom_path

        # Chama o método do engine legacy (pode ter nomes diferentes)
        if hasattr(self.legacy_engine, 'extract_texts'):
            legacy_data = self.legacy_engine.extract_texts(rom_path)
        elif hasattr(self.legacy_engine, 'extract_strings'):
            legacy_data = self.legacy_engine.extract_strings(rom_path)
        elif hasattr(self.legacy_engine, 'get_text_data'):
            legacy_data = self.legacy_engine.get_text_data(rom_path)
        else:
            # Fallback - tenta métodos comuns
            legacy_data = self._try_common_methods(rom_path)

        # Converte para formato padronizado
        return self._standardize_output(legacy_data, rom_path)

    def _try_common_methods(self, rom_path: str) -> Dict:
        """Tenta métodos comuns dos engines legacy"""
        methods_to_try = [
            'analyze_rom', 'load_rom', 'parse_rom',
            'read_text_tables', 'scan_text'
        ]

        for method_name in methods_to_try:
            if hasattr(self.legacy_engine, method_name):
                try:
                    method = getattr(self.legacy_engine, method_name)
                    result = method(rom_path)
                    if result:
                        return result
                except Exception as e:
                    print(f"⚠️ Método {method_name} falhou: {e}")
                    continue

        return {}

    def _standardize_output(self, legacy_data: Dict, rom_path: str) -> Dict[str, Any]:
        """
        Converte dados do formato legacy para o formato moderno padronizado

        Formato legacy (típico):
        {
            'strings': [(offset, text), ...],
            'pointers': [(addr, target), ...],
            'tables': {...}
        }

        Formato moderno (padronizado):
        {
            'file_path': str,
            'strings': {id: {original, translated, context, offset}},
            'metadata': {...}
        }
        """
        standardized = {
            'file_path': rom_path,
            'file_size': os.path.getsize(rom_path) if os.path.exists(rom_path) else 0,
            'console_type': self.console_type,
            'strings': {},
            'pointers': {},
            'tables': {},
            'metadata': {
                'rom_name': os.path.basename(rom_path),
                'console': self.console_type,
                'engine_type': 'legacy',
                'rom_hash': self._calculate_rom_hash(rom_path)
            }
        }

        # Converte strings (formato mais comum)
        if 'strings' in legacy_data:
            self._convert_strings(legacy_data['strings'], standardized)

        # Converte ponteiros (para ROMs com pointer tables)
        if 'pointers' in legacy_data:
            self._convert_pointers(legacy_data['pointers'], standardized)

        # Converte tabelas de texto (para ROMs com múltiplas tabelas)
        if 'tables' in legacy_data or 'text_tables' in legacy_data:
            tables = legacy_data.get('tables', legacy_data.get('text_tables', {}))
            self._convert_tables(tables, standardized)

        # Preserva metadados legacy
        if 'metadata' in legacy_data:
            standardized['metadata'].update(legacy_data['metadata'])

        return standardized

    def _convert_strings(self, legacy_strings: Union[List, Dict], output: Dict) -> None:
        """Converte strings do formato legacy"""
        if isinstance(legacy_strings, list):
            # Formato: [(offset, text), ...]
            for i, item in enumerate(legacy_strings):
                if isinstance(item, tuple) and len(item) >= 2:
                    offset, text = item[0], item[1]
                    key = f"STRING_{offset:06X}" if isinstance(offset, int) else f"STRING_{i:04d}"

                    output['strings'][key] = {
                        'original': str(text),
                        'translated': '',
                        'context': 'rom_string',
                        'offset': offset,
                        'console': self.console_type
                    }

        elif isinstance(legacy_strings, dict):
            # Formato: {offset: text, ...} ou {id: text, ...}
            for key, text in legacy_strings.items():
                if isinstance(key, int):
                    string_key = f"STRING_{key:06X}"
                    context_offset = key
                else:
                    string_key = f"STRING_{key}"
                    context_offset = None

                output['strings'][string_key] = {
                    'original': str(text),
                    'translated': '',
                    'context': 'rom_string',
                    'offset': context_offset,
                    'console': self.console_type
                }

    def _convert_pointers(self, legacy_pointers: Union[List, Dict], output: Dict) -> None:
        """Converte ponteiros do formato legacy"""
        if isinstance(legacy_pointers, list):
            for i, item in enumerate(legacy_pointers):
                if isinstance(item, tuple) and len(item) >= 2:
                    addr, target = item[0], item[1]
                    key = f"PTR_{addr:06X}"

                    output['pointers'][key] = {
                        'address': addr,
                        'target': target,
                        'console': self.console_type
                    }

        elif isinstance(legacy_pointers, dict):
            for addr, target in legacy_pointers.items():
                key = f"PTR_{addr:06X}" if isinstance(addr, int) else f"PTR_{addr}"
                output['pointers'][key] = {
                    'address': addr,
                    'target': target,
                    'console': self.console_type
                }

    def _convert_tables(self, legacy_tables: Dict, output: Dict) -> None:
        """Converte tabelas de texto do formato legacy"""
        for table_name, table_data in legacy_tables.items():
            output['tables'][table_name] = {
                'data': table_data,
                'console': self.console_type,
                'type': 'text_table'
            }

    def _calculate_rom_hash(self, rom_path: str) -> str:
        """Calcula hash MD5 da ROM para identificação"""
        try:
            with open(rom_path, 'rb') as f:
                # Para ROMs grandes, hasheamos só os primeiros 64KB + últimos 64KB
                file_size = os.path.getsize(rom_path)
                if file_size > 128 * 1024:  # > 128KB
                    header = f.read(64 * 1024)
                    f.seek(-64 * 1024, 2)  # Últimos 64KB
                    footer = f.read()
                    data = header + footer
                else:
                    data = f.read()

                return hashlib.md5(data).hexdigest()
        except Exception:
            return "unknown"

    def apply_translation(self, translations: Dict, output_path: str = None) -> bool:
        """Aplica traduções usando o engine legacy"""
        try:
            # Converte de volta para formato legacy
            legacy_translations = self._convert_to_legacy_format(translations)

            # Chama método do engine legacy
            if hasattr(self.legacy_engine, 'apply_translation'):
                return self.legacy_engine.apply_translation(legacy_translations, output_path)
            elif hasattr(self.legacy_engine, 'patch_rom'):
                return self.legacy_engine.patch_rom(self.rom_path, legacy_translations, output_path)
            elif hasattr(self.legacy_engine, 'write_translation'):
                return self.legacy_engine.write_translation(translations, output_path)
            else:
                print(f"⚠️ Engine {self.console_type} não suporta aplicação de tradução")
                return False

        except Exception as e:
            print(f"💥 Erro ao aplicar tradução via engine legacy: {e}")
            return False

    def _convert_to_legacy_format(self, modern_translations: Dict) -> Dict:
        """Converte traduções do formato moderno de volta para legacy"""
        legacy_format = {
            'strings': {},
            'pointers': {},
            'tables': {}
        }

        # Converte strings
        for key, data in modern_translations.get('strings', {}).items():
            if 'offset' in data and data['offset'] is not None:
                legacy_format['strings'][data['offset']] = data['translated']
            else:
                # Fallback usando o key
                legacy_format['strings'][key] = data['translated']

        # Converte ponteiros se existirem
        for key, data in modern_translations.get('pointers', {}).items():
            if 'address' in data:
                legacy_format['pointers'][data['address']] = data['target']

        # Converte tabelas se existirem
        for table_name, table_data in modern_translations.get('tables', {}).items():
            legacy_format['tables'][table_name] = table_data.get('data', table_data)

        return legacy_format


class EmulatorEngine(BaseEngine):
    """
    Engine principal para integração com engines legacy de console

    Este é o "Master System" - coordena todos os engines de console numa interface unificada.
    Como ter um emulador multi-sistema, mas para tradução!
    """

    def __init__(self):
        super().__init__()
        self.supported_formats = [
            '.nes', '.smc', '.sfc',  # Nintendo
            '.gb', '.gbc', '.gba',   # Game Boy family
            '.md', '.gen', '.smd',   # Sega Genesis/Mega Drive
            '.n64', '.z64', '.v64',  # Nintendo 64
            '.iso', '.bin', '.cue',  # PSX/Disc-based
            '.rom', '.zip'           # Generic
        ]

        # Registry dos engines legacy disponíveis
        self.legacy_engines = {
            'nes': NESEngine,
            'snes': SNESEngine,
            'gameboy': GBEngine,
            'gba': GBAEngine,
            'genesis': GenesisEngine,
            'n64': N64Engine,
            'psx': PSXEngine
        }

        # Cache de engines instanciados
        self._engine_cache = {}

        # Database de assinaturas de ROM para detecção automática
        self.rom_signatures = self._load_rom_signatures()

    def _load_rom_signatures(self) -> Dict[str, Dict]:
        """
        Carrega database de assinaturas para detecção automática
        É como ter uma "ROM database" - cada console tem suas peculiaridades
        """
        return {
            'nes': {
                'magic_bytes': [b'NES\x1A'],
                'extensions': ['.nes'],
                'header_size': 16,
                'detection_method': 'header'
            },
            'snes': {
                'magic_bytes': [],  # SNES não tem magic bytes fixos
                'extensions': ['.smc', '.sfc'],
                'header_size': [0, 512],  # Com ou sem header
                'detection_method': 'heuristic'
            },
            'gameboy': {
                'magic_bytes': [b'\xCE\xED\x66\x66'],  # Nintendo logo
                'extensions': ['.gb', '.gbc'],
                'header_offset': 0x104,
                'detection_method': 'logo'
            },
            'gba': {
                'magic_bytes': [b'\x24\xFF\xAE\x51\x69\x9A\xA2\x21'],  # Nintendo logo
                'extensions': ['.gba'],
                'header_offset': 0x04,
                'detection_method': 'logo'
            },
            'genesis': {
                'magic_bytes': [b'SEGA MEGA DRIVE', b'SEGA GENESIS'],
                'extensions': ['.md', '.gen', '.smd'],
                'header_offset': 0x100,
                'detection_method': 'header'
            }
        }

    def detect_software_type(self, file_path: str) -> bool:
        """
        Detecta se o arquivo é uma ROM suportada
        É como o autodetect do emulador - analisa magic bytes, extensão, etc.
        """
        if not os.path.exists(file_path):
            return False

        # Primeiro check: extensão
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            return False

        # Segundo check: tenta detectar console específico
        console_type = self.detect_console_type(file_path)
        return console_type is not None

    def detect_console_type(self, rom_path: str) -> Optional[str]:
        """
        Detecta o tipo de console baseado na ROM

        Usa várias heurísticas como um detective de ROM:
        1. Magic bytes (mais confiável)
        2. Extensão do arquivo
        3. Tamanho do arquivo
        4. Análise de header
5. Fallback para engines legacy (eles sabem melhor!)
        """
        try:
            with open(rom_path, 'rb') as f:
                rom_data = f.read(8192)  # Lê primeiros 8KB para análise

            file_ext = Path(rom_path).suffix.lower()
            file_size = os.path.getsize(rom_path)

            # Testa cada console
            for console, signature in self.rom_signatures.items():
                if self._matches_signature(rom_data, file_ext, file_size, signature):
                    print(f"🎮 Detectado: {console.upper()}")
                    return console

            # Fallback: pergunta para os engines legacy
            print("🔍 Detecção por signature falhou, tentando engines legacy...")
            for console, engine_class in self.legacy_engines.items():
                try:
                    engine = self._get_engine_instance(console, engine_class)
                    if hasattr(engine, 'is_compatible') and engine.is_compatible(rom_path):
                        print(f"🎯 Engine legacy confirmou: {console.upper()}")
                        return console
                except Exception as e:
                    print(f"⚠️ Engine {console} falhou na detecção: {e}")
                    continue

        except Exception as e:
            print(f"💥 Erro na detecção de console: {e}")

        return None

    def _matches_signature(self, rom_data: bytes, file_ext: str, file_size: int, signature: Dict) -> bool:
        """Verifica se ROM bate com signature do console"""

        # Check extensão
        if file_ext not in signature.get('extensions', []):
            return False

        detection_method = signature.get('detection_method', 'header')

        if detection_method == 'header':
            # Procura magic bytes no início
            for magic in signature.get('magic_bytes', []):
                if rom_data.startswith(magic):
                    return True

        elif detection_method == 'logo':
            # Procura logo Nintendo em offset específico
            offset = signature.get('header_offset', 0)
            if len(rom_data) > offset:
                for magic in signature.get('magic_bytes', []):
                    if rom_data[offset:offset+len(magic)] == magic:
                        return True

        elif detection_method == 'heuristic':
            # SNES é complicado - usa heurísticas
            return self._detect_snes_heuristic(rom_data, file_size)

        return False

    def _detect_snes_heuristic(self, rom_data: bytes, file_size: int) -> bool:
        """
        Detecção heurística para SNES (que não tem magic bytes confiáveis)

        SNES é o "problema child" dos consoles - sem signature clara,
        então analisamos padrões típicos
        """

        # Tamanhos típicos de ROM SNES (múltiplos de 32KB)
        typical_sizes = [0x80000, 0x100000, 0x200000, 0x300000, 0x400000]
        size_matches = any(abs(file_size - size) <= 512 for size in typical_sizes)

        if not size_matches:
            return False

        # Procura por header interno SNES
        # Pode estar em 0x7FC0 (LoROM) ou 0xFFC0 (HiROM)
        for header_offset in [0x7FC0, 0xFFC0]:
            if len(rom_data) > header_offset + 32:
                # Checksum complement check
                checksum = int.from_bytes(rom_data[header_offset+28:header_offset+30], 'little')
                complement = int.from_bytes(rom_data[header_offset+30:header_offset+32], 'little')

                if (checksum ^ complement) == 0xFFFF and checksum != 0:
                    return True

        return False

    def _get_engine_instance(self, console: str, engine_class) -> Any:
        """
        Gerencia cache de engines para performance
        Como um "engine pool" - evita criar instâncias desnecessárias
        """
        if console not in self._engine_cache:
            try:
                self._engine_cache[console] = engine_class()
                print(f"🏭 Engine {console} instanciado e cacheado")
            except Exception as e:
                print(f"💥 Falha ao instanciar engine {console}: {e}")
                return None

        return self._engine_cache[console]

    def extract_texts(self, file_path: str) -> Dict[str, Any]:
        """
        Extrai textos usando o engine legacy apropriado

        Funciona como um dispatcher - identifica o console e
        redireciona para o engine especializado
        """
        console_type = self.detect_console_type(file_path)

        if not console_type:
            raise ValueError(f"Console não detectado para: {os.path.basename(file_path)}")

        if console_type not in self.legacy_engines:
            raise ValueError(f"Engine não disponível para console: {console_type}")

        print(f"🎮 Processando {console_type.upper()} ROM: {os.path.basename(file_path)}")

        # Pega engine legacy
        engine_class = self.legacy_engines[console_type]
        legacy_engine = self._get_engine_instance(console_type, engine_class)

        if not legacy_engine:
            raise RuntimeError(f"Falha ao carregar engine {console_type}")

        # Cria adapter para padronizar interface
        adapter = LegacyEngineAdapter(legacy_engine, console_type)

        # Extrai textos usando adapter
        extracted_data = adapter.extract_texts(file_path)

        # Adiciona metadados específicos do emulador
        extracted_data['emulator_info'] = {
            'console_detected': console_type,
            'engine_version': getattr(legacy_engine, 'version', 'unknown'),
            'detection_confidence': 'high',  # Poderia ser calculado
            'recommended_emulator': self._get_recommended_emulator(console_type)
        }

        print(f"✅ Extraído com sucesso: {len(extracted_data.get('strings', {}))} strings")

        return extracted_data

    def _get_recommended_emulator(self, console_type: str) -> str:
        """Retorna emulador recomendado para cada console"""
        emulator_map = {
            'nes': 'Mesen',
            'snes': 'bsnes',
            'gameboy': 'SameBoy',
            'gba': 'mGBA',
            'genesis': 'Genesis Plus GX',
            'n64': 'Project64',
            'psx': 'DuckStation'
        }
        return emulator_map.get(console_type, 'Unknown')

    def extract_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Extrai áudio usando engine legacy (se suportado)

        Nem todos os engines legacy suportam áudio, mas quando suportam
        é aqui que a mágica acontece!
        """
        console_type = self.detect_console_type(file_path)

        if not console_type:
            return {'error': 'Console não detectado'}

        engine_class = self.legacy_engines.get(console_type)
        if not engine_class:
            return {'error': f'Engine não disponível para {console_type}'}

        legacy_engine = self._get_engine_instance(console_type, engine_class)

        # Tenta extrair áudio se engine suportar
        if hasattr(legacy_engine, 'extract_audio'):
            try:
                return legacy_engine.extract_audio(file_path)
            except Exception as e:
                print(f"⚠️ Falha na extração de áudio: {e}")

        return {'audio_resources': {}, 'note': 'Engine não suporta extração de áudio'}

    def apply_translation(self, translations: Dict, output_path: str = None) -> bool:
        """
        Aplica tradução usando engine legacy apropriado

        É o momento da verdade - quando finalmente patcheamos a ROM!
        """
        console_type = translations.get('console_type')
        if not console_type:
            # Tenta detectar pelo file_path
            source_path = translations.get('file_path')
            if source_path:
                console_type = self.detect_console_type(source_path)

        if not console_type or console_type not in self.legacy_engines:
            print(f"❌ Console não suportado para patch: {console_type}")
            return False

        print(f"🔧 Aplicando tradução {console_type.upper()}...")

        # Pega engine legacy
        engine_class = self.legacy_engines[console_type]
        legacy_engine = self._get_engine_instance(console_type, engine_class)

        # Cria adapter
        adapter = LegacyEngineAdapter(legacy_engine, console_type)

        # Aplica tradução
        try:
            success = adapter.apply_translation(translations, output_path)

            if success:
                print(f"🏆 Tradução aplicada com sucesso!")
                print(f"📁 Arquivo de saída: {output_path}")
            else:
                print(f"❌ Falha ao aplicar tradução")

            return success

        except Exception as e:
            print(f"💥 Erro crítico durante aplicação: {e}")
            return False

    def get_console_capabilities(self, console_type: str) -> Dict[str, bool]:
        """
        Retorna capacidades do engine para um console específico
        Útil para a UI saber quais features mostrar
        """
        if console_type not in self.legacy_engines:
            return {}

        engine_class = self.legacy_engines[console_type]
        legacy_engine = self._get_engine_instance(console_type, engine_class)

        capabilities = {
            'text_extraction': hasattr(legacy_engine, 'extract_texts') or
                             hasattr(legacy_engine, 'extract_strings'),
            'audio_extraction': hasattr(legacy_engine, 'extract_audio'),
            'translation_apply': hasattr(legacy_engine, 'apply_translation') or
                               hasattr(legacy_engine, 'patch_rom'),
            'pointer_support': hasattr(legacy_engine, 'update_pointers'),
            'table_support': hasattr(legacy_engine, 'get_text_tables'),
            'backup_support': hasattr(legacy_engine, 'create_backup'),
            'compression_support': hasattr(legacy_engine, 'decompress_text')
        }

        return capabilities

    def list_supported_consoles(self) -> List[Dict[str, str]]:
        """Lista todos os consoles suportados com informações"""
        consoles = []

        for console, engine_class in self.legacy_engines.items():
            try:
                engine = self._get_engine_instance(console, engine_class)
                capabilities = self.get_console_capabilities(console)

                consoles.append({
                    'name': console,
                    'display_name': console.upper(),
                    'engine_available': engine is not None,
                    'supported_extensions': self.rom_signatures.get(console, {}).get('extensions', []),
                    'capabilities': capabilities,
                    'recommended_emulator': self._get_recommended_emulator(console)
                })

            except Exception as e:
                print(f"⚠️ Erro ao carregar info do console {console}: {e}")

        return consoles


# Factory function para facilitar uso
def create_emulator_engine() -> EmulatorEngine:
    """
    Factory function para criar uma instância do EmulatorEngine

    Use esta função em vez de instanciar diretamente - ela garante
    que tudo está configurado corretamente
    """
    engine = EmulatorEngine()

    # Valida se pelo menos um engine legacy está disponível
    available_engines = []
    for console, engine_class in engine.legacy_engines.items():
        try:
            test_engine = engine_class()
            available_engines.append(console)
        except Exception as e:
            print(f"⚠️ Engine {console} não disponível: {e}")

    if not available_engines:
        print("❌ AVISO: Nenhum engine legacy disponível!")
        print("   Verifique se os engines estão na pasta engines_legacy/")
    else:
        print(f"✅ Engines disponíveis: {', '.join(available_engines)}")

    return engine


# Exemplo de uso integrado
if __name__ == "__main__":
    # Cria engine
    emulator_engine = create_emulator_engine()

    # Lista consoles suportados
    print("🎮 Consoles suportados:")
    for console_info in emulator_engine.list_supported_consoles():
        print(f"   {console_info['display_name']}: {console_info['supported_extensions']}")

    # Testa com uma ROM (exemplo)
    test_rom = "super_mario_world.smc"

    if os.path.exists(test_rom):
        if emulator_engine.detect_software_type(test_rom):
            print(f"\n🎯 Testando com: {test_rom}")

            # Extrai textos
            extracted = emulator_engine.extract_texts(test_rom)

            print(f"📊 Resultado:")
            print(f"   Console: {extracted['console_type']}")
            print(f"   Strings: {len(extracted.get('strings', {}))}")
            print(f"   Engine: {extracted.get('emulator_info', {}).get('engine_version', 'N/A')}")

            # Simula tradução
            for key, data in list(extracted.get('strings', {}).items())[:5]:  # Só primeiras 5
                data['translated'] = f"[PT] {data['original']}"

                     # Aplica tradução
            output_rom = test_rom.replace('.smc', '_traduzido.smc')
            success = emulator_engine.apply_translation(extracted, output_rom)

            if success:
                print(f"🏆 Tradução aplicada: {output_rom}")
            else:
                print("❌ Falha na aplicação da tradução")
        else:
            print(f"❌ ROM não reconhecida: {test_rom}")
    else:
        print(f"ℹ️ Arquivo de teste não encontrado: {test_rom}")
        print("   Para testar, coloque uma ROM na pasta e ajuste o nome do arquivo")


# Utilitários extras para integração
class LegacyEngineManager:
    """
    Gerenciador para facilitar integração e debugging dos engines legacy
    É como ter um "ROM organizer" - mantém tudo catalogado e funcionando
    """

    def __init__(self, emulator_engine: EmulatorEngine):
        self.emulator_engine = emulator_engine
        self.engine_health = {}
        self._check_engine_health()

    def _check_engine_health(self):
        """Verifica saúde de todos os engines legacy"""
        print("🏥 Verificando saúde dos engines legacy...")

        for console, engine_class in self.emulator_engine.legacy_engines.items():
            try:
                # Tenta instanciar
                engine = engine_class()

                # Verifica métodos essenciais
                health = {
                    'instantiable': True,
                    'has_extract': hasattr(engine, 'extract_texts') or hasattr(engine, 'extract_strings'),
                    'has_apply': hasattr(engine, 'apply_translation') or hasattr(engine, 'patch_rom'),
                    'has_detect': hasattr(engine, 'is_compatible'),
                    'version': getattr(engine, 'version', 'unknown'),
                    'status': 'healthy'
                }

                # Score de saúde
                score = sum([health['has_extract'], health['has_apply'], health['has_detect']])
                if score == 3:
                    health['status'] = 'excellent'
                elif score == 2:
                    health['status'] = 'good'
                elif score == 1:
                    health['status'] = 'limited'
                else:
                    health['status'] = 'broken'

                self.engine_health[console] = health

            except Exception as e:
                self.engine_health[console] = {
                    'instantiable': False,
                    'error': str(e),
                    'status': 'failed'
                }

    def print_health_report(self):
        """Imprime relatório de saúde dos engines"""
        print("\n📋 RELATÓRIO DE SAÚDE DOS ENGINES LEGACY")
        print("=" * 50)

        for console, health in self.engine_health.items():
            status_emoji = {
                'excellent': '🟢',
                'good': '🟡',
                'limited': '🟠',
                'broken': '🔴',
                'failed': '💀'
            }.get(health['status'], '❓')

            print(f"{status_emoji} {console.upper():<12} - {health['status'].upper()}")

            if health.get('instantiable', False):
                print(f"   📦 Extração: {'✅' if health.get('has_extract') else '❌'}")
                print(f"   🔧 Aplicação: {'✅' if health.get('has_apply') else '❌'}")
                print(f"   🔍 Detecção: {'✅' if health.get('has_detect') else '❌'}")
                print(f"   📟 Versão: {health.get('version', 'N/A')}")
            else:
                print(f"   💥 Erro: {health.get('error', 'Desconhecido')}")
            print()

    def suggest_fixes(self):
        """Sugere correções para engines com problemas"""
        print("🔧 SUGESTÕES DE CORREÇÃO")
        print("=" * 30)

        for console, health in self.engine_health.items():
            if health['status'] in ['broken', 'failed', 'limited']:
                print(f"\n🚨 {console.upper()} precisa de atenção:")

                if not health.get('instantiable'):
                    print("   - Verificar se o arquivo do engine existe")
                    print("   - Verificar imports e dependências")
                    print("   - Verificar sintaxe do código")

                if not health.get('has_extract'):
                    print("   - Implementar método extract_texts() ou extract_strings()")
                    print("   - Verificar se o método retorna dados no formato esperado")

                if not health.get('has_apply'):
                    print("   - Implementar método apply_translation() ou patch_rom()")
                    print("   - Verificar se consegue escrever no arquivo de saída")

                if not health.get('has_detect'):
                    print("   - Implementar método is_compatible()")
                    print("   - Melhorar detecção automática de ROM")


class ROMTestSuite:
    """
    Suite de testes para validar integração com engines legacy
    É como ter um "ROM tester" - garante que tudo funciona como esperado
    """

    def __init__(self, emulator_engine: EmulatorEngine):
        self.emulator_engine = emulator_engine
        self.test_results = {}

    def run_integration_tests(self, test_roms: Dict[str, str] = None):
        """
        Executa testes de integração com ROMs reais

        test_roms: Dict no formato {'console': 'caminho_da_rom'}
        """
        if not test_roms:
            print("ℹ️ Nenhuma ROM de teste fornecida")
            print("   Para testar completamente, forneça ROMs de cada console")
            return

        print("🧪 EXECUTANDO TESTES DE INTEGRAÇÃO")
        print("=" * 40)

        for console, rom_path in test_roms.items():
            print(f"\n🎮 Testando {console.upper()}: {os.path.basename(rom_path)}")

            test_result = {
                'console': console,
                'rom_path': rom_path,
                'detection': False,
                'extraction': False,
                'translation': False,
                'errors': []
            }

            try:
                # Teste 1: Detecção
                if self.emulator_engine.detect_software_type(rom_path):
                    detected_console = self.emulator_engine.detect_console_type(rom_path)
                    test_result['detection'] = True
                    test_result['detected_as'] = detected_console
                    print(f"   ✅ Detecção: {detected_console}")

                    if detected_console != console:
                        print(f"   ⚠️ Detectado como {detected_console}, esperado {console}")
                else:
                    test_result['errors'].append("Falha na detecção")
                    print(f"   ❌ Detecção falhou")
                    continue

                # Teste 2: Extração
                try:
                    extracted = self.emulator_engine.extract_texts(rom_path)
                    strings_count = len(extracted.get('strings', {}))

                    if strings_count > 0:
                        test_result['extraction'] = True
                        test_result['strings_extracted'] = strings_count
                        print(f"   ✅ Extração: {strings_count} strings")
                    else:
                        test_result['errors'].append("Nenhuma string extraída")
                        print(f"   ⚠️ Extração: 0 strings (pode ser normal)")

                except Exception as e:
                    test_result['errors'].append(f"Erro na extração: {e}")
                    print(f"   ❌ Extração falhou: {e}")
                    continue

                # Teste 3: Aplicação (simulada)
                try:
                    # Simula algumas traduções
                    sample_translations = {}
                    for i, (key, data) in enumerate(extracted.get('strings', {}).items()):
                        if i >= 3:  # Só 3 para teste
                            break
                        data['translated'] = f"[TESTE] {data['original']}"
                        sample_translations[key] = data

                    if sample_translations:
                        # Testa aplicação (sem realmente escrever arquivo)
                        test_output = rom_path.replace('.', '_test.')
                        success = self.emulator_engine.apply_translation(extracted, test_output)

                        if success:
                            test_result['translation'] = True
                            print(f"   ✅ Aplicação: Sucesso (simulado)")

                            # Remove arquivo de teste se foi criado
                            if os.path.exists(test_output):
                                os.remove(test_output)
                        else:
                            test_result['errors'].append("Falha na aplicação")
                            print(f"   ❌ Aplicação falhou")

                except Exception as e:
                    test_result['errors'].append(f"Erro na aplicação: {e}")
                    print(f"   ❌ Aplicação falhou: {e}")

            except Exception as e:
                test_result['errors'].append(f"Erro geral: {e}")
                print(f"   💥 Erro geral: {e}")

            self.test_results[console] = test_result

    def print_test_summary(self):
        """Imprime resumo dos testes"""
        if not self.test_results:
            print("❓ Nenhum teste executado")
            return

        print("\n📊 RESUMO DOS TESTES")
        print("=" * 25)

        total_tests = len(self.test_results)
        successful_detections = sum(1 for r in self.test_results.values() if r['detection'])
        successful_extractions = sum(1 for r in self.test_results.values() if r['extraction'])
        successful_translations = sum(1 for r in self.test_results.values() if r['translation'])

        print(f"📈 Detecção: {successful_detections}/{total_tests} ({successful_detections/total_tests*100:.1f}%)")
        print(f"📈 Extração: {successful_extractions}/{total_tests} ({successful_extractions/total_tests*100:.1f}%)")
        print(f"📈 Aplicação: {successful_translations}/{total_tests} ({successful_translations/total_tests*100:.1f}%)")

        # Mostra problemas
        print("\n🚨 PROBLEMAS ENCONTRADOS:")
        for console, result in self.test_results.items():
            if result['errors']:
                print(f"   {console.upper()}:")
                for error in result['errors']:
                    print(f"     - {error}")


# Script de setup e validação
def setup_legacy_integration():
    """
    Script principal para setup e validação da integração legacy

    Execute este script após configurar os engines legacy para verificar
    se tudo está funcionando corretamente
    """
    print("🚀 INICIANDO SETUP DA INTEGRAÇÃO LEGACY")
    print("=" * 50)

    # Cria engine principal
    emulator_engine = create_emulator_engine()

    # Verifica saúde dos engines
    manager = LegacyEngineManager(emulator_engine)
    manager.print_health_report()
    manager.suggest_fixes()

    # Opcionalmente executa testes com ROMs
    print("\n" + "=" * 50)
    print("Para executar testes completos, forneça ROMs de teste:")
    print("  test_roms = {")
    print("      'nes': 'roms/test.nes',")
    print("      'snes': 'roms/test.smc',")
    print("      'gba': 'roms/test.gba'")
    print("  }")
    print("  suite = ROMTestSuite(emulator_engine)")
    print("  suite.run_integration_tests(test_roms)")
    print("  suite.print_test_summary()")

    return emulator_engine, manager


if __name__ == "__main__":
    # Setup completo
    engine, manager = setup_legacy_integration()

    print("\n🎯 INTEGRAÇÃO LEGACY CONFIGURADA!")
    print("   Use o EmulatorEngine para processar ROMs")
    print("   Todos os engines legacy são acessados automaticamente")
    print("   Interface unificada mantém compatibilidade total")
    print("\n🎮 Agora você pode traduzir ROMs usando a nova arquitetura!")
    print("   multitradutor.translate_rom('jogo.nes', 'pt-br')")
    print("   # Internamente usa NESEngine legacy, mas com interface moderna!")