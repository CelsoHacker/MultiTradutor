import struct
import logging
import os
import re
import json
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import codecs
import hashlib
from collections import defaultdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class TextFormat(Enum):
    """Formatos de texto suportados"""
    NULL_TERMINATED = "null_terminated"
    LENGTH_PREFIXED = "length_prefixed"
    FIXED_LENGTH = "fixed_length"
    PASCAL_STRING = "pascal_string"
    WORD_PREFIXED = "word_prefixed"

class CompressionType(Enum):
    """Tipos de compressão suportados"""
    NONE = "none"
    LZSS = "lzss"
    RLE = "rle"
    HUFFMAN = "huffman"

@dataclass
class PointerInfo:
    """Informações sobre um ponteiro detectado"""
    offset: int
    target_offset: int
    size: int  # 2 ou 4 bytes
    is_relative: bool = False
    confidence: float = 0.0
    context: str = ""

@dataclass
class TextCluster:
    """Cluster de textos relacionados"""
    entries: List['PSXTextEntry']
    base_offset: int
    total_size: int
    pointer_table_offset: Optional[int] = None
    compression: CompressionType = CompressionType.NONE

@dataclass
class PSXTextEntry:
    """Representa uma entrada de texto extraída do PS1"""
    file_path: str
    offset: int
    original_text: str
    translated_text: str = ""
    context: str = ""
    game_id: str = ""
    encoding: str = "shift-jis"
    checksum: str = ""
    text_type: str = ""
    character_limit: int = 0
    format_type: TextFormat = TextFormat.NULL_TERMINATED
    pointer_info: Optional[PointerInfo] = None
    cluster_id: Optional[str] = None

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.md5(self.original_text.encode()).hexdigest()

@dataclass
class PSXFileInfo:
    """Informações sobre um arquivo dentro da ISO"""
    name: str
    path: str
    size: int
    offset: int
    is_text_file: bool = False
    encoding: Optional[str] = None
    file_type: str = ""
    compression: Optional[CompressionType] = None
    text_clusters: List[TextCluster] = field(default_factory=list)

class AdvancedPS1Engine:
    """
    Motor avançado de tradução para ROMs de PS1
    Inclui detecção automática de ponteiros, análise de clusters e compressão
    """

    def __init__(self, game_path: str, config_path: Optional[str] = None):
        self.game_path = Path(game_path)
        self.config_path = config_path
        self.is_iso = self.game_path.suffix.lower() in ['.iso', '.bin']
        self.is_directory = self.game_path.is_dir()

        # Dados do jogo
        self.game_data = self._load_game_data()
        self.file_system: Dict[str, PSXFileInfo] = {}
        self.text_entries: List[PSXTextEntry] = []
        self.pointer_tables: Dict[int, List[PointerInfo]] = {}
        self.text_clusters: Dict[str, TextCluster] = {}

        # Cache otimizado
        self._file_cache: Dict[str, bytes] = {}
        self._pointer_cache: Dict[int, List[PointerInfo]] = {}
        self._analysis_cache: Dict[str, Any] = {}

        # Threading
        self._max_workers = min(4, os.cpu_count() or 1)
        self._thread_lock = threading.Lock()

        # Configurações
        self.config = self._load_config()

        # Indexar arquivos
        self._index_files()

        logger.info(f"Motor avançado PS1 inicializado - Processadores: {self._max_workers}")

    def _load_game_data(self) -> Union[bytes, Path]:
        """Carrega dados do jogo com verificação de integridade"""
        if self.is_iso:
            with open(self.game_path, 'rb') as f:
                data = f.read()
            return data
        return self.game_path

    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações do jogo"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _index_files(self) -> None:
        """Indexa arquivos da ISO/diretório"""
        if self.is_iso:
            self._index_iso_files()
        else:
            self._index_directory_files()

    def _index_iso_files(self) -> None:
        """Indexa arquivos da ISO usando estrutura CD-ROM"""
        # Implementação simplificada - em produção usaria parser ISO9660 completo
        pass

    def _index_directory_files(self) -> None:
        """Indexa arquivos do diretório"""
        for root, dirs, files in os.walk(self.game_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.game_path)

                file_info = PSXFileInfo(
                    name=file,
                    path=str(relative_path),
                    size=file_path.stat().st_size,
                    offset=0
                )

                self.file_system[str(relative_path)] = file_info

    # ================================
    # MELHORIA 1: DETECÇÃO AUTOMÁTICA DE PONTEIROS
    # ================================

    def detect_pointer_table(self, data: bytes, base_offset: int = 0) -> List[PointerInfo]:
        """
        Detecta tabela de ponteiros com análise heurística avançada

        Args:
            data: Dados binários para análise
            base_offset: Offset base para cálculo de ponteiros relativos

        Returns:
            Lista de informações sobre ponteiros detectados
        """
        cache_key = f"ptr_{hashlib.md5(data[:1024]).hexdigest()}_{base_offset}"

        if cache_key in self._pointer_cache:
            return self._pointer_cache[cache_key]

        pointers = []

        # Análise em paralelo para diferentes tamanhos de ponteiro
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_4byte = executor.submit(self._detect_pointers_4byte, data, base_offset)
            future_2byte = executor.submit(self._detect_pointers_2byte, data, base_offset)

            pointers.extend(future_4byte.result())
            pointers.extend(future_2byte.result())

        # Filtrar ponteiros por confiança
        high_confidence_pointers = [p for p in pointers if p.confidence > 0.7]

        # Cache resultado
        self._pointer_cache[cache_key] = high_confidence_pointers

        logger.info(f"Detectados {len(high_confidence_pointers)} ponteiros com alta confiança")
        return high_confidence_pointers

    def _detect_pointers_4byte(self, data: bytes, base_offset: int) -> List[PointerInfo]:
        """Detecta ponteiros de 4 bytes"""
        pointers = []

        for i in range(0, len(data) - 4, 4):  # Alinhamento de 4 bytes
            try:
                ptr_value = struct.unpack('<I', data[i:i+4])[0]

                # Validar range do ponteiro
                if self._is_valid_pointer_value(ptr_value, len(data), base_offset):
                    target_offset = ptr_value - base_offset if ptr_value > base_offset else ptr_value

                    # Calcular confiança baseada em contexto
                    confidence = self._calculate_pointer_confidence(
                        data, i, target_offset, ptr_value, 4
                    )

                    if confidence > 0.5:  # Threshold mínimo
                        pointer_info = PointerInfo(
                            offset=i,
                            target_offset=target_offset,
                            size=4,
                            is_relative=ptr_value > base_offset,
                            confidence=confidence,
                            context=f"4byte_ptr_{i:06X}"
                        )
                        pointers.append(pointer_info)

            except struct.error:
                continue

        return pointers

    def _detect_pointers_2byte(self, data: bytes, base_offset: int) -> List[PointerInfo]:
        """Detecta ponteiros de 2 bytes (offsets relativos)"""
        pointers = []

        for i in range(0, len(data) - 2, 2):  # Alinhamento de 2 bytes
            try:
                ptr_value = struct.unpack('<H', data[i:i+2])[0]

                # Ponteiros de 2 bytes são tipicamente relativos
                if 0x100 < ptr_value < 0x8000:  # Range típico para offsets relativos
                    target_offset = ptr_value

                    confidence = self._calculate_pointer_confidence(
                        data, i, target_offset, ptr_value, 2
                    )

                    if confidence > 0.6:  # Threshold mais alto para 2 bytes
                        pointer_info = PointerInfo(
                            offset=i,
                            target_offset=target_offset,
                            size=2,
                            is_relative=True,
                            confidence=confidence,
                            context=f"2byte_ptr_{i:06X}"
                        )
                        pointers.append(pointer_info)

            except struct.error:
                continue

        return pointers

    def _is_valid_pointer_value(self, ptr_value: int, data_size: int, base_offset: int) -> bool:
        """Valida se um valor pode ser um ponteiro válido"""
        # Checks básicos
        if ptr_value == 0 or ptr_value == 0xFFFFFFFF:
            return False

        # Range check para ponteiros absolutos
        if ptr_value > base_offset:
            adjusted_value = ptr_value - base_offset
            return 0 < adjusted_value < data_size

        # Range check para ponteiros relativos
        return 0 < ptr_value < data_size

    def _calculate_pointer_confidence(self, data: bytes, ptr_offset: int,
                                    target_offset: int, ptr_value: int, size: int) -> float:
        """
        Calcula confiança de que um valor é realmente um ponteiro

        Fatores considerados:
        - Sequência de ponteiros adjacentes
        - Validade do target (texto ou dados estruturados)
        - Alinhamento
        - Padrões conhecidos
        """
        confidence = 0.0

        # Fator 1: Sequência de ponteiros (25% do peso)
        sequence_score = self._check_pointer_sequence(data, ptr_offset, size)
        confidence += sequence_score * 0.25

        # Fator 2: Validade do target (35% do peso)
        if target_offset < len(data):
            target_score = self._check_pointer_target(data, target_offset)
            confidence += target_score * 0.35

        # Fator 3: Alinhamento (15% do peso)
        alignment_score = 1.0 if ptr_offset % size == 0 else 0.5
        confidence += alignment_score * 0.15

        # Fator 4: Range realístico (15% do peso)
        range_score = self._check_pointer_range(ptr_value, len(data))
        confidence += range_score * 0.15

        # Fator 5: Padrões conhecidos (10% do peso)
        pattern_score = self._check_known_patterns(data, ptr_offset, ptr_value)
        confidence += pattern_score * 0.10

        return min(confidence, 1.0)

    def _check_pointer_sequence(self, data: bytes, offset: int, size: int) -> float:
        """Verifica se há sequência de ponteiros adjacentes"""
        sequence_count = 1

        # Verificar ponteiros anteriores
        for i in range(offset - size, -1, -size):
            if i < 0:
                break

            try:
                if size == 4:
                    value = struct.unpack('<I', data[i:i+4])[0]
                else:
                    value = struct.unpack('<H', data[i:i+2])[0]

                if self._is_valid_pointer_value(value, len(data), 0):
                    sequence_count += 1
                else:
                    break
            except struct.error:
                break

        # Verificar ponteiros posteriores
        for i in range(offset + size, len(data), size):
            if i + size > len(data):
                break

            try:
                if size == 4:
                    value = struct.unpack('<I', data[i:i+4])[0]
                else:
                    value = struct.unpack('<H', data[i:i+2])[0]

                if self._is_valid_pointer_value(value, len(data), 0):
                    sequence_count += 1
                else:
                    break
            except struct.error:
                break

        # Normalizar score (mais ponteiros = maior confiança)
        return min(sequence_count / 10.0, 1.0)

    def _check_pointer_target(self, data: bytes, target_offset: int) -> float:
        """Verifica se o target do ponteiro contém dados válidos"""
        if target_offset >= len(data):
            return 0.0

        # Verificar se aponta para texto
        text_score = self._analyze_text_at_offset(data, target_offset)
        if text_score > 0.7:
            return 1.0

        # Verificar se aponta para dados estruturados
        structure_score = self._analyze_structure_at_offset(data, target_offset)
        if structure_score > 0.6:
            return 0.8

        return 0.3  # Score baixo para dados não identificados

    def _analyze_text_at_offset(self, data: bytes, offset: int) -> float:
        """Analisa se há texto válido no offset"""
        if offset >= len(data):
            return 0.0

        sample_size = min(32, len(data) - offset)
        sample = data[offset:offset + sample_size]

        # Contar caracteres imprimíveis
        printable_count = sum(1 for b in sample if 0x20 <= b <= 0x7E)

        # Contar caracteres japoneses (Shift-JIS)
        japanese_count = sum(1 for b in sample if 0x81 <= b <= 0x9F or 0xE0 <= b <= 0xFC)

        # Verificar terminadores
        has_terminator = 0x00 in sample or 0xFF in sample

        total_valid = printable_count + japanese_count
        ratio = total_valid / len(sample)

        # Bonus para terminadores
        if has_terminator:
            ratio += 0.2

        return min(ratio, 1.0)

    def _analyze_structure_at_offset(self, data: bytes, offset: int) -> float:
        """Analisa se há dados estruturados no offset"""
        if offset >= len(data):
            return 0.0

        sample_size = min(64, len(data) - offset)
        sample = data[offset:offset + sample_size]

        # Verificar padrões estruturados
        null_ratio = sample.count(0) / len(sample)

        # Estruturas tendem a ter muitos nulls (padding)
        if 0.3 < null_ratio < 0.7:
            return 0.8

        # Verificar alinhamento de dados
        if len(sample) >= 8:
            alignment_score = self._check_data_alignment(sample)
            return alignment_score

        return 0.2

    def _check_data_alignment(self, data: bytes) -> float:
        """Verifica alinhamento de dados estruturados"""
        # Verificar se há padrões de alinhamento de 4 bytes
        aligned_nulls = 0
        for i in range(0, len(data), 4):
            if i + 3 < len(data) and data[i+3] == 0:
                aligned_nulls += 1

        return min(aligned_nulls / (len(data) // 4), 1.0)

    def _check_pointer_range(self, ptr_value: int, data_size: int) -> float:
        """Verifica se o ponteiro está em range realístico"""
        # Ponteiros muito baixos ou muito altos são suspeitos
        if ptr_value < 0x1000:
            return 0.2
        if ptr_value > 0x80000000:  # Limite superior para PS1
            return 0.1

        # Range típico para PS1 (RAM)
        if 0x80000000 <= ptr_value <= 0x80200000:
            return 1.0

        # Range para offsets relativos
        if ptr_value < data_size:
            return 0.9

        return 0.5

    def _check_known_patterns(self, data: bytes, offset: int, ptr_value: int) -> float:
        """Verifica padrões conhecidos de ponteiros"""
        # Verificar assinaturas conhecidas antes do ponteiro
        if offset >= 4:
            prev_bytes = data[offset-4:offset]

            # Padrões comuns em jogos PS1
            known_patterns = [
                b'\x00\x00\x00\x00',  # Null padding
                b'\xFF\xFF\xFF\xFF',  # Padding
                b'\x00\x00\x00\x01',  # Counter
            ]

            if prev_bytes in known_patterns:
                return 0.8

        return 0.5

    # ================================
    # MELHORIA 2: ANÁLISE DE CLUSTERS
    # ================================

    def analyze_text_clusters(self, file_data: bytes, file_info: PSXFileInfo) -> List[TextCluster]:
        """
        Analisa clusters de texto relacionados

        Args:
            file_data: Dados do arquivo
            file_info: Informações do arquivo

        Returns:
            Lista de clusters de texto identificados
        """
        clusters = []

        # Detectar ponteiros primeiro
        pointers = self.detect_pointer_table(file_data)

        # Agrupar ponteiros por proximidade
        pointer_groups = self._group_pointers_by_proximity(pointers)

        # Analisar cada grupo
        for group in pointer_groups:
            cluster = self._analyze_pointer_group(file_data, group, file_info)
            if cluster:
                clusters.append(cluster)

        # Atualizar informações do arquivo
        file_info.text_clusters = clusters

        logger.info(f"Encontrados {len(clusters)} clusters de texto em {file_info.name}")
        return clusters

    def _group_pointers_by_proximity(self, pointers: List[PointerInfo]) -> List[List[PointerInfo]]:
        """Agrupa ponteiros por proximidade física"""
        if not pointers:
            return []

        # Ordenar por offset
        sorted_pointers = sorted(pointers, key=lambda p: p.offset)

        groups = []
        current_group = [sorted_pointers[0]]

        for i in range(1, len(sorted_pointers)):
            prev_ptr = sorted_pointers[i-1]
            curr_ptr = sorted_pointers[i]

            # Agrupar se estão próximos (dentro de 64 bytes)
            if curr_ptr.offset - prev_ptr.offset <= 64:
                current_group.append(curr_ptr)
            else:
                # Finalizar grupo atual e iniciar novo
                if len(current_group) >= 3:  # Mínimo para ser considerado tabela
                    groups.append(current_group)
                current_group = [curr_ptr]

        # Adicionar último grupo
        if len(current_group) >= 3:
            groups.append(current_group)

        return groups

    def _analyze_pointer_group(self, file_data: bytes, pointers: List[PointerInfo],
                             file_info: PSXFileInfo) -> Optional[TextCluster]:
        """Analisa um grupo de ponteiros para formar cluster"""
        if not pointers:
            return None

        text_entries = []

        # Extrair texto de cada ponteiro
        for pointer in pointers:
            try:
                text = self._extract_text_from_pointer(file_data, pointer)
                if text:
                    entry = PSXTextEntry(
                        file_path=file_info.path,
                        offset=pointer.target_offset,
                        original_text=text,
                        context=f"cluster_{pointer.context}",
                        pointer_info=pointer
                    )
                    text_entries.append(entry)
            except Exception as e:
                logger.debug(f"Erro ao extrair texto do ponteiro {pointer.offset:X}: {e}")
                continue

        if not text_entries:
            return None

        # Calcular estatísticas do cluster
        base_offset = min(p.offset for p in pointers)
        targets = [p.target_offset for p in pointers]
        total_size = max(targets) - min(targets) if targets else 0

        # Detectar compressão
        compression = self._detect_compression(file_data, min(targets), total_size)

        cluster = TextCluster(
            entries=text_entries,
            base_offset=base_offset,
            total_size=total_size,
            pointer_table_offset=base_offset,
            compression=compression
        )

        # Gerar ID único para o cluster
        cluster_id = f"cluster_{file_info.name}_{base_offset:06X}"

        # Atualizar entradas com cluster ID
        for entry in text_entries:
            entry.cluster_id = cluster_id

        # Adicionar ao cache
        self.text_clusters[cluster_id] = cluster

        return cluster

    def _extract_text_from_pointer(self, file_data: bytes, pointer: PointerInfo) -> Optional[str]:
        """Extrai texto apontado por um ponteiro"""
        if pointer.target_offset >= len(file_data):
            return None

        # Tentar diferentes formatos de texto
        formats_to_try = [
            TextFormat.NULL_TERMINATED,
            TextFormat.LENGTH_PREFIXED,
            TextFormat.PASCAL_STRING
        ]

        for format_type in formats_to_try:
            try:
                text = self._extract_text_with_format(file_data, pointer.target_offset, format_type)
                if text and self._is_valid_extracted_text(text):
                    return text
            except Exception:
                continue

        return None

    def _extract_text_with_format(self, data: bytes, offset: int, format_type: TextFormat) -> str:
        """Extrai texto usando formato específico"""
        if offset >= len(data):
            return ""

        if format_type == TextFormat.NULL_TERMINATED:
            end_pos = offset
            while end_pos < len(data) and data[end_pos] != 0:
                end_pos += 1
            text_bytes = data[offset:end_pos]

        elif format_type == TextFormat.LENGTH_PREFIXED:
            if offset + 1 >= len(data):
                return ""
            length = data[offset]
            text_bytes = data[offset + 1:offset + 1 + length]

        elif format_type == TextFormat.PASCAL_STRING:
            if offset + 1 >= len(data):
                return ""
            length = data[offset]
            text_bytes = data[offset + 1:offset + 1 + length]

        else:
            return ""

        # Tentar decodificar com diferentes encodings
        encodings = ['ascii', 'shift-jis', 'iso-8859-1']

        for encoding in encodings:
            try:
                text = text_bytes.decode(encoding, errors='ignore')
                if text.strip():
                    return text
            except Exception:
                continue

        return ""

    def _is_valid_extracted_text(self, text: str) -> bool:
        """Valida se texto extraído é válido"""
        if not text or len(text.strip()) < 2:
            return False

        # Verificar ratio de caracteres válidos
        valid_chars = sum(1 for c in text if c.isprintable() or c.isspace())
        ratio = valid_chars / len(text)

        return ratio > 0.7

    def _detect_compression(self, data: bytes, offset: int, size: int) -> CompressionType:
        """Detecta tipo de compressão nos dados"""
        if offset + size > len(data):
            return CompressionType.NONE

        sample = data[offset:offset + min(size, 256)]

        # Verificar assinaturas conhecidas
        if sample.startswith(b'LZ'):
            return CompressionType.LZSS
        if sample.startswith(b'RLE'):
            return CompressionType.RLE

        # Análise estatística
        entropy = self._calculate_entropy(sample)

        if entropy < 0.5:
            return CompressionType.RLE  # Baixa entropia sugere RLE
        elif entropy > 0.95:
            return CompressionType.HUFFMAN  # Alta entropia sugere Huffman

        return CompressionType.NONE

    def _calculate_entropy(self, data: bytes) -> float:
        """Calcula entropia de Shannon dos dados"""
        if not data:
            return 0.0

        # Contar frequência de bytes
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1

        # Calcular entropia
        entropy = 0.0
        length = len(data)

        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * (p.bit_length() - 1)

        return entropy / 8.0  # Normalizar para 0-1

    # ================================
    # MELHORIA 3: COMPRESSÃO LZ77/LZSS
    # ================================

    def compress_lzss(self, data: bytes) -> bytes:
        """
        Comprime dados usando LZSS (Lempel-Ziv-Storer-Szymanski)
        Implementação otimizada para textos de jogos PS1
        """
        if not data:
            return b''

        # Parâmetros LZSS otimizados para PS1
        window_size = 4096
        lookahead_size = 18
        min_match_length = 3

        compressed = bytearray()
        pos = 0

        while pos < len(data):
            # Procurar melhor match na janela
            best_match = self._find_best_match(
                data, pos, window_size, lookahead_size, min_match_length
            )

            if best_match:
                offset, length = best_match
                # Codificar match: bit 1 + offset + length
                compressed.append(0x80 | (offset >> 8))
                compressed.append(offset & 0xFF)
                compressed.append(length)
                pos += length
            else:
                # Codificar literal: bit 0 + byte
                compressed.append(data[pos])
                pos += 1

        return bytes(compressed)

    def _find_best_match(self, data: bytes, pos: int, window_size: int,
                        lookahead_size: int, min_match_length: int) -> Optional[Tuple[int, int]]:
        """Encontra melhor match para compressão LZSS"""
        if pos >= len(data):
            return None

        # Definir janela de busca
        start = max(0, pos - window_size)
        end = min(len(data), pos + lookahead_size)

        best_offset = 0
        best_length = 0

        # Procurar matches na janela
        for i in range(start, pos):
            # Verificar match começando em i
            length = 0
            while (i + length < pos and
                   pos + length < end and
                   data[i + length] == data[pos + length]):
                length += 1

            # Atualizar melhor match
            if length >= min_match_length and length > best_length:
                best_offset = pos - i
                best_length = length

        return (best_offset, best_length) if best_length >= min_match_length else None

    def decompress_lzss(self, compressed_data: bytes) -> bytes:
        """
        Descomprime dados LZSS
        """
        if not compressed_data:
            return b''

        decompressed = bytearray()
        pos = 0

        while pos < len(compressed_data):
            if pos >= len(compressed_data):
                break

            # Ler primeiro byte
            control = compressed_data[pos]
            pos += 1

            if control & 0x80:  # Match
             if pos + 1 >= len(compressed_data):
                         # Bit 1: dados literais
               literal_byte = data[pos]
               pos += 1
               output.append(literal_byte)
            else:
               # Bit 0: referência (offset + length)
               if pos + 1 >= len(data):
                   break

               # Lê o par offset/length (formato little-endian)
               ref_data = struct.unpack('<H', data[pos:pos+2])[0]
               pos += 2

               # Extrai offset e length do formato LZSS
               offset = (ref_data >> 4) & 0xFFF  # 12 bits para offset
               length = (ref_data & 0xF) + 3     # 4 bits para length, +3 para range 3-18

               # Valida a referência
               if offset == 0 or offset > len(output):
                break

               # Copia dados da janela deslizante
               start_pos = len(output) - offset
               for i in range(length):
                   if start_pos + i < len(output):
                       output.append(output[start_pos + i])
                   else:
                       # Padding com zeros se necessário
                       output.append(0)

           # Avança para o próximo bit de controle
        control <<= 1
        bit_count += 1

        return bytes(output)

def analyze_compression_ratio(original_size, compressed_size):
   """Analisa a eficiência da compressão"""
   if original_size == 0:
       return 0.0

   ratio = (compressed_size / original_size) * 100
   saved_bytes = original_size - compressed_size

   return {
       'ratio': ratio,
       'saved_bytes': saved_bytes,
       'efficiency': f"{100 - ratio:.1f}%"
   }

def create_compression_report(file_path, original_data, compressed_data):
   """Gera relatório detalhado da compressão"""
   stats = analyze_compression_ratio(len(original_data), len(compressed_data))

   report = f"""
=== RELATÓRIO DE COMPRESSÃO LZSS ===
Arquivo: {file_path}
Tamanho original: {len(original_data):,} bytes
Tamanho comprimido: {len(compressed_data):,} bytes
Taxa de compressão: {stats['ratio']:.1f}%
Bytes economizados: {stats['saved_bytes']:,}
Eficiência: {stats['efficiency']}
"""

   return report

# Melhorias na função principal de processamento
def process_lzss_compression(self, file_path):
   """Processa compressão LZSS com relatório detalhado"""
   try:
       # Carrega o arquivo
       with open(file_path, 'rb') as f:
           original_data = f.read()

       self.logger.info(f"Iniciando compressão LZSS: {file_path}")

       # Comprime os dados
       compressed_data = compress_data(original_data)

       # Gera relatório
       report = create_compression_report(file_path, original_data, compressed_data)
       self.logger.info(report)

       # Salva arquivo comprimido
       output_path = file_path.replace('.bin', '_compressed.lzss')
       with open(output_path, 'wb') as f:
           f.write(compressed_data)

       # Teste de integridade: descomprime e compara
       decompressed_data = decompress_data(compressed_data)
       if decompressed_data == original_data:
           self.logger.info("✓ Teste de integridade passou - dados íntegros")
       else:
           self.logger.error("✗ Erro na integridade - dados corrompidos")

       return {
           'success': True,
           'original_size': len(original_data),
           'compressed_size': len(compressed_data),
           'output_path': output_path,
           'integrity_check': decompressed_data == original_data
       }

   except Exception as e:
       self.logger.error(f"Erro na compressão LZSS: {str(e)}")
       return {'success': False, 'error': str(e)}

# Adicionando suporte para batch processing
def batch_compress_lzss(self, directory_path, file_pattern="*.bin"):
   """Comprime múltiplos arquivos em lote"""
   import glob

   files = glob.glob(os.path.join(directory_path, file_pattern))
   results = []

   self.logger.info(f"Iniciando compressão em lote: {len(files)} arquivos")

   for file_path in files:
       result = self.process_lzss_compression(file_path)
       results.append({
           'file': os.path.basename(file_path),
           'result': result
       })

   # Relatório consolidado
   successful = sum(1 for r in results if r['result']['success'])
   total_original = sum(r['result'].get('original_size', 0) for r in results if r['result']['success'])
   total_compressed = sum(r['result'].get('compressed_size', 0) for r in results if r['result']['success'])

   self.logger.info(f"""
=== RELATÓRIO CONSOLIDADO ===
Arquivos processados: {successful}/{len(files)}
Tamanho total original: {total_original:,} bytes
Tamanho total comprimido: {total_compressed:,} bytes
Economia total: {total_original - total_compressed:,} bytes
""")

   return results

# Integração com o motor principal
def integrate_lzss_engine(self):
   """Integra o motor LZSS ao sistema principal"""
   # Adiciona os métodos à classe principal
   self.compress_lzss = compress_data
   self.decompress_lzss = decompress_data
   self.process_lzss_compression = lambda path: process_lzss_compression(self, path)
   self.batch_compress_lzss = lambda dir_path, pattern="*.bin": batch_compress_lzss(self, dir_path, pattern)

   self.logger.info("✓ Motor LZSS integrado com sucesso")

# Comando para o CLI
def add_lzss_commands(self):
   """Adiciona comandos LZSS ao CLI"""
   lzss_commands = {
       'compress': {
           'description': 'Comprime arquivo usando LZSS',
           'usage': 'compress <arquivo>',
           'function': lambda args: self.process_lzss_compression(args[0]) if args else print("Uso: compress <arquivo>")
       },
       'decompress': {
           'description': 'Descomprime arquivo LZSS',
           'usage': 'decompress <arquivo.lzss>',
           'function': lambda args: self.decompress_lzss_file(args[0]) if args else print("Uso: decompress <arquivo.lzss>")
       },
       'batch_compress': {
           'description': 'Comprime múltiplos arquivos',
           'usage': 'batch_compress <diretório> [padrão]',
           'function': lambda args: self.batch_compress_lzss(args[0], args[1] if len(args) > 1 else "*.bin")
       }
   }

   # Adiciona ao dicionário de comandos existente
   self.commands.update(lzss_commands)

def decompress_lzss_file(self, file_path):
   """Descomprime arquivo LZSS para arquivo original"""
   try:
       with open(file_path, 'rb') as f:
           compressed_data = f.read()

       decompressed_data = decompress_data(compressed_data)

       output_path = file_path.replace('.lzss', '_decompressed.bin')
       with open(output_path, 'wb') as f:
           f.write(decompressed_data)

       self.logger.info(f"Arquivo descomprimido: {output_path}")
       return {'success': True, 'output_path': output_path}

   except Exception as e:
       self.logger.error(f"Erro na descompressão: {str(e)}")
       return {'success': False, 'error': str(e)}
       def detect_pointers_automatically(self, rom_data, start_address=0, end_address=None, min_confidence=0.7):

        """Detecta ponteiros automaticamente usando análise heurística
   Como um bom disassembler, procura padrões que fazem sentido
   """
   if end_address is None:
       end_address = len(rom_data)

   detected_pointers = []
   confidence_scores = []

   self.logger.info(f"Iniciando detecção automática de ponteiros: {start_address:06X} - {end_address:06X}")

   # Analisa cada posição possível
   for addr in range(start_address, end_address - 1, 2):  # Step 2 para little-endian
       try:
           # Lê valor como ponteiro de 16-bit
           pointer_value = struct.unpack('<H', rom_data[addr:addr+2])[0]

           # Calcula métricas de confiança
           confidence = self.calculate_pointer_confidence(rom_data, addr, pointer_value)

           if confidence >= min_confidence:
               detected_pointers.append({
                   'address': addr,
                   'value': pointer_value,
                   'confidence': confidence,
                   'target_address': self.resolve_pointer_address(pointer_value),
                   'context': self.analyze_pointer_context(rom_data, addr)
               })

       except (struct.error, IndexError):
           continue

   # Ordena por confiança e remove duplicatas
   detected_pointers = self.filter_and_rank_pointers(detected_pointers)

   self.logger.info(f"Detectados {len(detected_pointers)} ponteiros com confiança >= {min_confidence}")
   return detected_pointers

def calculate_pointer_confidence(self, rom_data, pointer_addr, pointer_value):
   """
   Calcula confiança de que um valor é realmente um ponteiro
   Usa várias heurísticas como um detective de assembly
   """
   confidence = 0.0

   # Heurística 1: Valor está em range válido de ROM
   if 0x8000 <= pointer_value <= 0xFFFF:
       confidence += 0.3

   # Heurística 2: Aponta para área de dados válida
   target_addr = self.resolve_pointer_address(pointer_value)
   if 0 <= target_addr < len(rom_data):
       confidence += 0.2

       # Heurística 3: Destino parece ser início de string ou dados
       if self.looks_like_text_start(rom_data, target_addr):
           confidence += 0.3
       elif self.looks_like_data_structure(rom_data, target_addr):
           confidence += 0.2

   # Heurística 4: Está em área típica de tabela de ponteiros
   if self.is_in_pointer_table_area(pointer_addr):
       confidence += 0.2

   # Heurística 5: Sequência de ponteiros adjacentes
   if self.has_adjacent_pointers(rom_data, pointer_addr):
       confidence += 0.2

   # Heurística 6: Padrão de incremento lógico
   if self.follows_logical_sequence(rom_data, pointer_addr):
       confidence += 0.1

   return min(confidence, 1.0)

def looks_like_text_start(self, rom_data, address):
   """Verifica se endereço parece início de texto"""
   try:
       # Procura por caracteres imprimíveis
       sample = rom_data[address:address+16]
       printable_count = sum(1 for b in sample if 32 <= b <= 126 or b in [0x00, 0xFF])

       # Verifica terminadores comuns
       has_terminator = any(b in [0x00, 0xFF, 0xFE] for b in sample)

       return (printable_count / len(sample)) > 0.6 and has_terminator
   except IndexError:
       return False

def looks_like_data_structure(self, rom_data, address):
   """Verifica se parece estrutura de dados"""
   try:
       # Procura por padrões repetitivos
       sample = rom_data[address:address+32]

       # Verifica se há padrões de bytes
       pattern_score = 0
       for i in range(0, len(sample)-4, 4):
           chunk = sample[i:i+4]
           if len(set(chunk)) <= 2:  # Máximo 2 valores diferentes
               pattern_score += 1

       return pattern_score >= 2
   except IndexError:
       return False

def is_in_pointer_table_area(self, address):
   """Verifica se está em área típica de tabelas"""
   # Áreas comuns de tabelas em ROMs
   common_table_areas = [
       (0x0000, 0x4000),  # Área de header e tabelas iniciais
       (0x8000, 0x9000),  # Área de dados do jogo
       (0xF000, 0xFFFF),  # Área de vetores e tabelas finais
   ]

   return any(start <= address <= end for start, end in common_table_areas)

def has_adjacent_pointers(self, rom_data, address):
   """Verifica se há ponteiros adjacentes (indicativo de tabela)"""
   try:
       # Verifica 4 posições antes e depois
       adjacent_valid = 0

       for offset in [-4, -2, 2, 4]:
           check_addr = address + offset
           if 0 <= check_addr < len(rom_data) - 1:
               pointer_val = struct.unpack('<H', rom_data[check_addr:check_addr+2])[0]
               if 0x8000 <= pointer_val <= 0xFFFF:
                   adjacent_valid += 1

       return adjacent_valid >= 2
   except (struct.error, IndexError):
       return False

def follows_logical_sequence(self, rom_data, address):
   """Verifica se ponteiros seguem sequência lógica"""
   try:
       current_ptr = struct.unpack('<H', rom_data[address:address+2])[0]

       # Verifica próximo ponteiro
       if address + 2 < len(rom_data) - 1:
           next_ptr = struct.unpack('<H', rom_data[address+2:address+4])[0]

           # Deve haver incremento lógico (não muito grande)
           diff = next_ptr - current_ptr
           return 0 < diff < 0x1000  # Diferença razoável

       return False
   except (struct.error, IndexError):
       return False

def analyze_pointer_context(self, rom_data, address):
   """Analisa contexto ao redor do ponteiro"""
   try:
       # Pega contexto de 8 bytes antes e depois
       start = max(0, address - 8)
       end = min(len(rom_data), address + 10)
       context = rom_data[start:end]

       # Identifica padrões
       patterns = []

       # Verifica se está em tabela de ponteiros
       if self.has_adjacent_pointers(rom_data, address):
           patterns.append("pointer_table")

       # Verifica se há padding característico
       if any(b == 0x00 for b in context[:4]) or any(b == 0xFF for b in context[:4]):
           patterns.append("padded_data")

       # Verifica se há opcodes próximos
       if any(b in [0x20, 0x4C, 0x60, 0xA9] for b in context):  # JSR, JMP, RTS, LDA
           patterns.append("near_code")

       return patterns
   except IndexError:
       return []

def filter_and_rank_pointers(self, detected_pointers):
   """Filtra e ordena ponteiros por relevância"""
   # Remove duplicatas baseadas em endereço
   unique_pointers = {}
   for ptr in detected_pointers:
       addr = ptr['address']
       if addr not in unique_pointers or ptr['confidence'] > unique_pointers[addr]['confidence']:
           unique_pointers[addr] = ptr

   # Ordena por confiança
   sorted_pointers = sorted(unique_pointers.values(), key=lambda x: x['confidence'], reverse=True)

   # Agrupa por contexto similar
   grouped = self.group_pointers_by_context(sorted_pointers)

   return grouped

def group_pointers_by_context(self, pointers):
   """Agrupa ponteiros por contexto similar"""
   groups = {}

   for ptr in pointers:
       context_key = tuple(sorted(ptr['context']))
       if context_key not in groups:
           groups[context_key] = []
       groups[context_key].append(ptr)

   # Retorna lista ordenada por grupo
   result = []
   for context, group in groups.items():
       result.extend(group)

   return result

def create_pointer_detection_report(self, detected_pointers):
   """Gera relatório da detecção automática"""
   report = ["=== RELATÓRIO DE DETECÇÃO AUTOMÁTICA DE PONTEIROS ===\n"]

   # Estatísticas gerais
   total_pointers = len(detected_pointers)
   avg_confidence = sum(p['confidence'] for p in detected_pointers) / total_pointers if total_pointers > 0 else 0

   report.append(f"Total de ponteiros detectados: {total_pointers}")
   report.append(f"Confiança média: {avg_confidence:.2f}")
   report.append("")

   # Análise por contexto
   context_stats = {}
   for ptr in detected_pointers:
       for context in ptr['context']:
           context_stats[context] = context_stats.get(context, 0) + 1

   report.append("=== ANÁLISE POR CONTEXTO ===")
   for context, count in sorted(context_stats.items(), key=lambda x: x[1], reverse=True):
       report.append(f"{context}: {count} ponteiros")

   report.append("")

   # Top 10 ponteiros mais confiáveis
   report.append("=== TOP 10 PONTEIROS MAIS CONFIÁVEIS ===")
   for i, ptr in enumerate(detected_pointers[:10]):
       report.append(f"{i+1:2d}. {ptr['address']:06X} -> {ptr['target_address']:06X} "
                    f"(confiança: {ptr['confidence']:.2f}) {ptr['context']}")

   return "\n".join(report)

def integrate_pointer_detection(self):
   """Integra detecção automática ao sistema principal"""
   # Adiciona métodos à classe
   self.detect_pointers_automatically = lambda rom_data, start=0, end=None, conf=0.7: detect_pointers_automatically(self, rom_data, start, end, conf)
   self.calculate_pointer_confidence = lambda rom_data, addr, val: calculate_pointer_confidence(self, rom_data, addr, val)
   self.create_pointer_detection_report = lambda pointers: create_pointer_detection_report(self, pointers)

   # Adiciona comando CLI
   self.commands['detect_pointers'] = {
       'description': 'Detecta ponteiros automaticamente',
       'usage': 'detect_pointers [start] [end] [confidence]',
       'function': self.cmd_detect_pointers
   }

   self.logger.info("✓ Sistema de detecção automática de ponteiros integrado")

def cmd_detect_pointers(self, args):
   """Comando CLI para detecção automática"""
   if not self.current_rom_data:
       print("Erro: Nenhuma ROM carregada")
       return

   # Processa argumentos
   start_addr = int(args[0], 16) if len(args) > 0 else 0
   end_addr = int(args[1], 16) if len(args) > 1 else None
   confidence = float(args[2]) if len(args) > 2 else 0.7

   print(f"Detectando ponteiros com confiança >= {confidence}...")

   # Executa detecção
   pointers = self.detect_pointers_automatically(
       self.current_rom_data,
       start_addr,
       end_addr,
       confidence
   )

   # Mostra resultados
   if pointers:
       print(f"\n{len(pointers)} ponteiros detectados:")
       for i, ptr in enumerate(pointers[:20]):  # Mostra só os primeiros 20
           print(f"{i+1:2d}. {ptr['address']:06X} -> {ptr['target_address']:06X} "
                 f"(conf: {ptr['confidence']:.2f})")

       if len(pointers) > 20:
           print(f"... e mais {len(pointers) - 20} ponteiros")

       # Gera relatório completo
       report = self.create_pointer_detection_report(pointers)
       print(f"\n{report}")

       # Salva em arquivo
       with open('pointer_detection_report.txt', 'w') as f:
           f.write(report)
       print("\nRelatório salvo em: pointer_detection_report.txt")
   else:
       print("Nenhum ponteiro detectado com a confiança especificada")

def auto_detect_and_extract_text(self, confidence_threshold=0.8):
   """
   Combina detecção automática com extração de texto
   Como um combo move bem executado!
   """
   if not self.current_rom_data:
       self.logger.error("Nenhuma ROM carregada")
       return

   self.logger.info("Iniciando detecção automática + extração de texto...")

   # Detecta ponteiros automaticamente
   pointers = self.detect_pointers_automatically(
       self.current_rom_data,
       min_confidence=confidence_threshold
   )

   # Filtra apenas ponteiros que apontam para texto
   text_pointers = []
   for ptr in pointers:
       if 'pointer_table' in ptr['context'] or self.looks_like_text_start(self.current_rom_data, ptr['target_address']):
           text_pointers.append(ptr)

   self.logger.info(f"Encontrados {len(text_pointers)} ponteiros de texto")

   # Extrai texto usando os ponteiros detectados
   extracted_texts = []
   for ptr in text_pointers:
       try:
           text = self.extract_text_by_pointer(ptr['address'])
           if text and len(text.strip()) > 0:
               extracted_texts.append({
                   'pointer_address': ptr['address'],
                   'text_address': ptr['target_address'],
                   'text': text,
                   'confidence': ptr['confidence']
               })
       except Exception as e:
           self.logger.warning(f"Erro ao extrair texto do ponteiro {ptr['address']:06X}: {e}")

   self.logger.info(f"Extraídos {len(extracted_texts)} textos automaticamente")

   # Salva resultados
   self.save_auto_extracted_texts(extracted_texts)

   return extracted_texts

def save_auto_extracted_texts(self, extracted_texts):
   """Salva textos extraídos automaticamente"""
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"auto_extracted_texts_{timestamp}.txt"

   with open(filename, 'w', encoding='utf-8') as f:
       f.write("=== TEXTOS EXTRAÍDOS AUTOMATICAMENTE ===\n\n")

       for i, item in enumerate(extracted_texts):
           f.write(f"#{i+1:03d} | Ponteiro: {item['pointer_address']:06X} | "
                  f"Texto: {item['text_address']:06X} | Conf: {item['confidence']:.2f}\n")
           f.write(f"Texto: {item['text']}\n")
           f.write("-" * 80 + "\n\n")

   self.logger.info(f"Textos salvos em: {filename}")
   def decompress_data(data: bytes, method: str) -> bytes:

     """Sistema unificado de descompressão - como um emulador multi-core
   Suporta múltiplos algoritmos mantendo interface consistente
   """
   method = method.lower()

   compression_methods = {
       'lzss': decompress_lzss,
       'rnc': decompress_rnc,
       'rnc1': lambda data: decompress_rnc(data, version=1),
       'rnc2': lambda data: decompress_rnc(data, version=2),
       'packfire': lambda data: decompress_rnc(data, version=1),  # Alias para RNC1
       'lzs': decompress_lzs,
       'lz77': decompress_lz77,
       'lz77_custom': lambda data: decompress_lz77(data, custom_header=True)
   }

   if method not in compression_methods:
       raise ValueError(f"Método de compressão '{method}' não suportado. "
                       f"Métodos disponíveis: {list(compression_methods.keys())}")

   return compression_methods[method](data)

def compress_data(data: bytes, method: str, **kwargs) -> bytes:
   """
   Sistema unificado de compressão - o inverso do processo
   """
   method = method.lower()

   compression_methods = {
       'lzss': compress_lzss,
       'rnc1': lambda data: compress_rnc(data, version=1),
       'rnc2': lambda data: compress_rnc(data, version=2),
       'lzs': compress_lzs,
       'lz77': compress_lz77,
       'lz77_custom': lambda data: compress_lz77(data, custom_header=True)
   }

   if method not in compression_methods:
       raise ValueError(f"Método de compressão '{method}' não suportado.")

   return compression_methods[method](data, **kwargs)

# ==================== IMPLEMENTAÇÃO RNC ====================

def decompress_rnc(data: bytes, version: int = None) -> bytes:
   """
   Descomprime dados RNC (Rob Northen Compression)
   Como decifrar o código de um boss final - precisa conhecer os padrões!
   """
   if len(data) < 18:
       raise ValueError("Dados RNC muito pequenos para ter header válido")

   # Detecta versão automaticamente se não especificada
   if version is None:
       version = detect_rnc_version(data)

   # Valida header RNC
   if data[:4] != b'RNC\x01' and data[:4] != b'RNC\x02':
       raise ValueError("Header RNC inválido")

   # Lê informações do header
   header = parse_rnc_header(data)

   if version == 1:
       return decompress_rnc1(data, header)
   elif version == 2:
       return decompress_rnc2(data, header)
   else:
       raise ValueError(f"Versão RNC {version} não suportada")

def detect_rnc_version(data: bytes) -> int:
   """Detecta versão RNC pelo header"""
   if data[:4] == b'RNC\x01':
       return 1
   elif data[:4] == b'RNC\x02':
       return 2
   else:
       raise ValueError("Header RNC não reconhecido")

def parse_rnc_header(data: bytes) -> dict:
   """
   Parseia header RNC - como ler as specs de um chip custom
   """
   return {
       'signature': data[:4],
       'uncompressed_size': struct.unpack('>I', data[4:8])[0],
       'compressed_size': struct.unpack('>I', data[8:12])[0],
       'uncompressed_crc': struct.unpack('>H', data[12:14])[0],
       'compressed_crc': struct.unpack('>H', data[14:16])[0],
       'leeway': data[16],
       'pack_chunks': data[17],
       'data_start': 18
   }

def decompress_rnc1(data: bytes, header: dict) -> bytes:
   """
   Descomprime RNC versão 1 - o algoritmo original
   """
   compressed_data = data[header['data_start']:]
   output = bytearray()

   # Inicializa bit reader
   bit_reader = RNCBitReader(compressed_data)

   # Tabelas de Huffman para RNC1
   raw_table = build_rnc1_raw_table(bit_reader)
   pos_table = build_rnc1_pos_table(bit_reader)
   len_table = build_rnc1_len_table(bit_reader)

   # Loop principal de descompressão
   while len(output) < header['uncompressed_size']:
       if bit_reader.get_bit():
           # Bit 1: dado literal
           if raw_table:
               output.append(decode_huffman(bit_reader, raw_table))
           else:
               output.append(bit_reader.get_bits(8))
       else:
           # Bit 0: referência LZ77
           pos = decode_huffman(bit_reader, pos_table)
           length = decode_huffman(bit_reader, len_table)

           # Decodifica posição e comprimento
           if pos >= 2:
               pos = (1 << (pos - 1)) + bit_reader.get_bits(pos - 1)

           if length >= 2:
               length = (1 << (length - 1)) + bit_reader.get_bits(length - 1)

           length += 2  # Comprimento mínimo é 2

           # Copia dados da janela
           for i in range(length):
               if len(output) >= pos:
                   output.append(output[-pos])
               else:
                   output.append(0)  # Padding se necessário

   return bytes(output[:header['uncompressed_size']])

def decompress_rnc2(data: bytes, header: dict) -> bytes:
   """
   Descomprime RNC versão 2 - versão otimizada
   """
   compressed_data = data[header['data_start']:]
   output = bytearray()

   # RNC2 usa formato diferente - mais eficiente
   bit_reader = RNCBitReader(compressed_data)

   # Constrói tabelas específicas do RNC2
   huffman_tables = build_rnc2_tables(bit_reader)

   # Loop de descompressão RNC2
   while len(output) < header['uncompressed_size']:
       token = decode_huffman(bit_reader, huffman_tables['main'])

       if token < 256:
           # Literal byte
           output.append(token)
       else:
           # Match (referência)
           length = token - 254
           pos = decode_huffman(bit_reader, huffman_tables['distance'])

           # Decodifica distância
           if pos >= 4:
               extra_bits = (pos - 2) // 2
               pos = (2 + (pos % 2)) * (1 << extra_bits) + bit_reader.get_bits(extra_bits)

           # Copia match
           for i in range(length):
               if len(output) >= pos:
                   output.append(output[-pos])
               else:
                   output.append(0)

   return bytes(output[:header['uncompressed_size']])

class RNCBitReader:
   """
   Leitor de bits para RNC - como um shift register hardware
   """
   def __init__(self, data: bytes):
       self.data = data
       self.pos = 0
       self.bit_buffer = 0
       self.bits_available = 0

   def get_bit(self) -> int:
       """Lê um bit"""
       if self.bits_available == 0:
           self.refill_buffer()

       bit = self.bit_buffer & 1
       self.bit_buffer >>= 1
       self.bits_available -= 1
       return bit

   def get_bits(self, count: int) -> int:
       """Lê múltiplos bits"""
       result = 0
       for i in range(count):
           result |= (self.get_bit() << i)
       return result

   def refill_buffer(self):
       """Recarrega buffer de bits"""
       if self.pos < len(self.data):
           self.bit_buffer = self.data[self.pos]
           self.pos += 1
           self.bits_available = 8

# ==================== IMPLEMENTAÇÃO LZS ====================

def decompress_lzs(data: bytes) -> bytes:
   """
   Descomprime LZS - usado em jogos PS1 e Dreamcast
   Como um algoritmo que aprendeu com os erros do LZ77
   """
   if len(data) < 8:
       raise ValueError("Dados LZS muito pequenos")

   # Lê header LZS
   header = parse_lzs_header(data)

   compressed_data = data[header['data_start']:]
   output = bytearray()
   pos = 0

   while pos < len(compressed_data) and len(output) < header['uncompressed_size']:
       control_byte = compressed_data[pos]
       pos += 1

       # Processa 8 bits de controle
       for bit in range(8):
           if pos >= len(compressed_data):
               break

           if control_byte & (1 << bit):
               # Bit 1: literal
               output.append(compressed_data[pos])
               pos += 1
           else:
               # Bit 0: referência
               if pos + 1 >= len(compressed_data):
                   break

               # Formato LZS: 12 bits offset + 4 bits length
               ref_word = struct.unpack('<H', compressed_data[pos:pos+2])[0]
               pos += 2

               offset = ref_word & 0xFFF
               length = ((ref_word >> 12) & 0xF) + 3

               # Valida referência
               if offset == 0 or offset > len(output):
                   break

               # Copia dados
               start_pos = len(output) - offset
               for i in range(length):
                   if start_pos + i < len(output):
                       output.append(output[start_pos + i])
                   else:
                       output.append(0)

   return bytes(output[:header['uncompressed_size']])

def parse_lzs_header(data: bytes) -> dict:
   """Parseia header LZS"""
   return {
       'signature': data[:4],
       'uncompressed_size': struct.unpack('<I', data[4:8])[0],
       'data_start': 8
   }

def compress_lzs(data: bytes) -> bytes:
   """
   Comprime dados usando LZS
   """
   header = struct.pack('<4sI', b'LZS\x00', len(data))
   compressed = bytearray(header)

   # Implementa compressão LZS básica
   pos = 0
   while pos < len(data):
       control_byte = 0
       control_pos = len(compressed)
       compressed.append(0)  # Placeholder para byte de controle

       bit_count = 0
       while bit_count < 8 and pos < len(data):
           # Procura match na janela
           match = find_lzs_match(data, pos)

           if match and match['length'] >= 3:
               # Adiciona referência
               ref_word = match['offset'] | ((match['length'] - 3) << 12)
               compressed.extend(struct.pack('<H', ref_word))
               pos += match['length']
               # Bit 0 = referência (já é 0)
           else:
               # Adiciona literal
               compressed.append(data[pos])
               pos += 1
               control_byte |= (1 << bit_count)  # Bit 1 = literal

           bit_count += 1

       # Atualiza byte de controle
       compressed[control_pos] = control_byte

   return bytes(compressed)

def find_lzs_match(data: bytes, pos: int, window_size: int = 4096) -> dict:
   """Encontra melhor match LZS"""
   if pos < 3:
       return None

   start_window = max(0, pos - window_size)
   best_match = None

   for i in range(start_window, pos):
       length = 0
       while (pos + length < len(data) and
              i + length < pos and
              data[i + length] == data[pos + length] and
              length < 18):  # Max length = 15 + 3
           length += 1

       if length >= 3:
           offset = pos - i
           if not best_match or length > best_match['length']:
               best_match = {'offset': offset, 'length': length}

   return best_match

# ==================== IMPLEMENTAÇÃO LZ77 CUSTOMIZADO ====================

def decompress_lz77(data: bytes, custom_header: bool = False) -> bytes:
   """
   Descomprime LZ77 com suporte a headers customizados
   Como um LZ77 que cresceu e virou adulto
   """
   if custom_header:
       header = parse_lz77_custom_header(data)
       compressed_data = data[header['data_start']:]
       target_size = header['uncompressed_size']
   else:
       # Header padrão LZ77
       if len(data) < 4:
           raise ValueError("Dados LZ77 muito pequenos")
       target_size = struct.unpack('<I', data[:4])[0]
       compressed_data = data[4:]

   output = bytearray()
   pos = 0

   while pos < len(compressed_data) and len(output) < target_size:
       flags = compressed_data[pos]
       pos += 1

       for bit in range(8):
           if pos >= len(compressed_data) or len(output) >= target_size:
               break

           if flags & (1 << bit):
               # Literal
               output.append(compressed_data[pos])
               pos += 1
           else:
               # Referência
               if pos + 1 >= len(compressed_data):
                   break

               # Formato: 4 bits length + 12 bits offset
               ref_data = struct.unpack('<H', compressed_data[pos:pos+2])[0]
               pos += 2

               length = (ref_data >> 12) + 3
               offset = ref_data & 0xFFF

               if offset == 0:
                   break

               # Copia dados
               for i in range(length):
                   if len(output) >= offset:
                       output.append(output[-offset])
                   else:
                       output.append(0)

   return bytes(output[:target_size])

def parse_lz77_custom_header(data: bytes) -> dict:
   """
   Parseia headers customizados LZ77
   Diferentes jogos, diferentes formatos - como dialetos de assembly
   """
   # Detecta tipo de header baseado em padrões
   if data[:4] == b'LZ77':
       # Header tipo A: "LZ77" + size + compressed_size
       return {
           'signature': data[:4],
           'uncompressed_size': struct.unpack('<I', data[4:8])[0],
           'compressed_size': struct.unpack('<I', data[8:12])[0],
           'data_start': 12
       }
   elif data[0] == 0x10:  # Nintendo DS/GBA style
       # Header tipo B: flag + size (3 bytes little-endian)
       size = struct.unpack('<I', data[1:4] + b'\x00')[0]
       return {
           'signature': data[:1],
           'uncompressed_size': size,
           'data_start': 4
       }
   else:
       # Header tipo C: size simples
       return {
           'uncompressed_size': struct.unpack('<I', data[:4])[0],
           'data_start': 4
       }

def compress_lz77(data: bytes, custom_header: bool = False) -> bytes:
   """Comprime dados usando LZ77"""
   if custom_header:
       header = struct.pack('<4sII', b'LZ77', len(data), 0)  # Placeholder para compressed_size
   else:
       header = struct.pack('<I', len(data))

   compressed = bytearray()
   pos = 0

   while pos < len(data):
       flag_byte = 0
       flag_pos = len(compressed)
       compressed.append(0)  # Placeholder

       bit_count = 0
       while bit_count < 8 and pos < len(data):
           match = find_lz77_match(data, pos)

           if match and match['length'] >= 3:
               # Referência
               ref_data = ((match['length'] - 3) << 12) | match['offset']
               compressed.extend(struct.pack('<H', ref_data))
               pos += match['length']
           else:
               # Literal
               compressed.append(data[pos])
               pos += 1
               flag_byte |= (1 << bit_count)

           bit_count += 1

       compressed[flag_pos] = flag_byte

   # Atualiza header se necessário
   if custom_header:
       final_data = bytearray(header)
       struct.pack_into('<I', final_data, 8, len(compressed))  # compressed_size
       final_data.extend(compressed)
       return bytes(final_data)
   else:
       return header + bytes(compressed)

def find_lz77_match(data: bytes, pos: int) -> dict:
   """Encontra melhor match LZ77"""
   if pos < 3:
       return None

   window_size = min(4096, pos)
   best_match = None

   for i in range(max(0, pos - window_size), pos):
       length = 0
       while (pos + length < len(data) and
              data[i + length] == data[pos + length] and
              length < 18):
           length += 1

       if length >= 3:
           offset = pos - i
           if not best_match or length > best_match['length']:
               best_match = {'offset': offset, 'length': length}

   return best_match

# ==================== INTEGRAÇÃO E COMANDOS ====================

def integrate_compression_suite(self):
   """Integra suite completa de compressão"""
   # Adiciona métodos unificados
   self.decompress_data = lambda data, method: decompress_data(data, method)
   self.compress_data = lambda data, method, **kwargs: compress_data(data, method, **kwargs)

   # Adiciona comandos CLI específicos
   compression_commands = {
       'decompress': {
           'description': 'Descomprime dados usando método especificado',
           'usage': 'decompress <arquivo> <método>',
           'function': self.cmd_decompress
       },
       'compress': {
           'description': 'Comprime dados usando método especificado',
           'usage': 'compress <arquivo> <método>',
           'function': self.cmd_compress
       },
       'detect_compression': {
           'description': 'Detecta tipo de compressão automaticamente',
           'usage': 'detect_compression <arquivo>',
           'function': self.cmd_detect_compression
       }
   }

   self.commands.update(compression_commands)
   self.logger.info("✓ Suite completa de compressão integrada")

def cmd_decompress(self, args):
   """Comando para descompressão"""
   if len(args) < 2:
       print("Uso: decompress <arquivo> <método>")
       print("Métodos: lzss, rnc, rnc1, rnc2, lzs, lz77, lz77_custom")
       return

   filename, method = args[0], args[1]

   try:
       with open(filename, 'rb') as f:
           data = f.read()

       print(f"Descomprimindo {filename} usando {method}...")
       decompressed = self.decompress_data(data, method)

       output_file = filename.replace('.', f'_decompressed_{method}.')
       with open(output_file, 'wb') as f:
           f.write(decompressed)

       print(f"✓ Arquivo descomprimido salvo em: {output_file}")
       print(f"Tamanho original: {len(data):,} bytes")
       print(f"Tamanho descomprimido: {len(decompressed):,} bytes")

   except Exception as e:
       print(f"Erro na descompressão: {e}")

def detect_compression_type(data: bytes) -> str:
   """
   Detecta tipo de compressão automaticamente
   Como um scanner que reconhece diferentes tipos de arquivo
   """
   # Verifica assinaturas conhecidas
   if data[:4] == b'RNC\x01':
       return 'rnc1'
   elif data[:4] == b'RNC\x02':
       return 'rnc2'
   elif data[:4] == b'LZS\x00':
       return 'lzs'
   elif data[:4] == b'LZ77':
       return 'lz77_custom'
   elif data[0] == 0x10:  # Nintendo style
       return 'lz77_custom'
   elif len(data) >= 4:
       # Heurística para LZSS
       if detect_lzss_pattern(data):
           return 'lzss'
       # Heurística para LZ77 padrão
       elif detect_lz77_pattern(data):
           return 'lz77'

   return 'unknown'

def cmd_detect_compression(self, args):
   """Comando para detectar compressão"""
   if not args:
       print("Uso: detect_compression <arquivo>")
       return

   filename = args[0]

   try:
       with open(filename, 'rb') as f:
           data = f.read()

       compression_type = detect_compression_type(data)

       print(f"Arquivo: {filename}")
       print(f"Tipo detectado: {compression_type}")
       print(f"Tamanho: {len(data):,} bytes")

       if compression_type != 'unknown':
           print(f"Para descomprimir: decompress {filename} {compression_type}")

   except Exception as e:
       print(f"Erro na detecção: {e}")
       if __name__ == "__main__":
        print("Testando engine de tradução PS1...")

       engine = PS1Engine("exemplo.iso")
       print(f"Jogo detectado: {engine.game_profile.get('title', 'Desconhecido')}")
