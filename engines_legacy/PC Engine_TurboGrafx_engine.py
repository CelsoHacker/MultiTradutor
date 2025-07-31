#!/usr/bin/env python3
"""
Utilitários específicos para ROM hacking do PC Engine/TurboGrafx-16
Ferramentas avançadas para análise e manipulação de ROMs
"""

import struct
import os
import json
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PCEngineHeader:
    """Header da ROM PC Engine"""
    signature: bytes
    rom_size: int
    backup_ram_size: int
    region_code: int
    version: int
    checksum: int

class PCEngineROMUtils:
    """Utilitários específicos para ROMs do PC Engine"""

    def __init__(self):
        self.HUCARD_BANKS = {
            0x8000: "Bank 0 (Fixed)",
            0xA000: "Bank 1 (Switchable)",
            0xC000: "Bank 2 (Switchable)",
            0xE000: "Bank 3 (Switchable)"
        }

        self.MEMORY_MAP = {
            0x0000: "RAM - Zero Page",
            0x0100: "RAM - Stack",
            0x0200: "RAM - User",
            0x2000: "I/O Area",
            0x4000: "CD-ROM² Area",
            0x6000: "Backup RAM",
            0x8000: "ROM Area",
            0xF000: "Hardware I/O",
            0xFF00: "Interrupt Vectors"
        }

        # Instruções HuC6280 (baseado no 6502 com extensões)
        self.HUC6280_OPCODES = {
            0x00: "BRK", 0x01: "ORA (zp,X)", 0x02: "SXY", 0x03: "ST0",
            0x04: "TSB zp", 0x05: "ORA zp", 0x06: "ASL zp", 0x07: "RMB0",
            0x08: "PHP", 0x09: "ORA #", 0x0A: "ASL A", 0x0B: "---",
            0x0C: "TSB abs", 0x0D: "ORA abs", 0x0E: "ASL abs", 0x0F: "BBR0",
            # ... (continua com todos os opcodes)
        }

    def analyze_rom_structure(self, rom_path: str) -> Dict:
        """Analisa estrutura detalhada da ROM"""
        with open(rom_path, 'rb') as f:
            rom_data = f.read()

        analysis = {
            'file_info': self._analyze_file_info(rom_data),
            'header': self._parse_header(rom_data),
            'memory_layout': self._analyze_memory_layout(rom_data),
            'code_segments': self._identify_code_segments(rom_data),
            'text_segments': self._identify_text_segments(rom_data),
            'graphics_data': self._identify_graphics_data(rom_data),
            'sound_data': self._identify_sound_data(rom_data),
            'compression': self._detect_compression(rom_data),
            'bank_structure': self._analyze_bank_structure(rom_data)
        }

        return analysis

    def _analyze_file_info(self, rom_data: bytes) -> Dict:
        """Analisa informações básicas do arquivo"""
        size = len(rom_data)

        # Detectar formato
        format_type = "Unknown"
        if size % 0x2000 == 0:  # Múltiplo de 8KB
            format_type = "HuCard"
        elif size % 0x4000 == 0:  # Múltiplo de 16KB
            format_type = "HuCard (16KB banks)"

        # Detectar se tem header
        has_header = False
        if size % 0x2000 == 0x200:  # 512 bytes de header
            has_header = True
            size -= 0x200

        return {
            'total_size': len(rom_data),
            'rom_size': size,
            'format': format_type,
            'has_header': has_header,
            'bank_count': size // 0x2000,
            'expected_banks': self._calculate_expected_banks(size)
        }

    def _parse_header(self, rom_data: bytes) -> Optional[PCEngineHeader]:
        """Analisa header da ROM se presente"""
        if len(rom_data) < 0x20:
            return None

        # Procurar por assinatura conhecida
        signatures = [b'PC Engine', b'TurboGrafx', b'HUDSON']

        for i in range(0, min(len(rom_data), 0x1000), 16):
            chunk = rom_data[i:i+16]
            for sig in signatures:
                if sig in chunk:
                    return PCEngineHeader(
                        signature=sig,
                        rom_size=len(rom_data),
                        backup_ram_size=0,
                        region_code=0,
                        version=1,
                        checksum=sum(rom_data) & 0xFFFF
                    )

        return None

    def _analyze_memory_layout(self, rom_data: bytes) -> Dict:
        """Analisa layout de memória"""
        layout = {}

        # Analisar vetores de interrupção (final da ROM)
        if len(rom_data) >= 6:
            vectors = {
                'reset': struct.unpack('<H', rom_data[-6:-4])[0],
                'irq': struct.unpack('<H', rom_data[-4:-2])[0],
                'nmi': struct.unpack('<H', rom_data[-2:])[0]
            }
            layout['interrupt_vectors'] = vectors

        # Identificar regiões de código
        code_regions = []
        for i in range(0, len(rom_data), 0x2000):  # Por banco
            bank_data = rom_data[i:i+0x2000]
            if self._is_code_region(bank_data):
                code_regions.append({
                    'start': i,
                    'end': min(i + 0x2000, len(rom_data)),
                    'bank': i // 0x2000,
                    'confidence': self._calculate_code_confidence(bank_data)
                })

        layout['code_regions'] = code_regions
        return layout

    def _identify_code_segments(self, rom_data: bytes) -> List[Dict]:
        """Identifica segmentos de código"""
        segments = []

        # Procurar por padrões de código HuC6280
        for i in range(0, len(rom_data) - 1):
            opcode = rom_data[i]

            # Verificar se é um opcode válido
            if opcode in self.HUC6280_OPCODES:
                # Analisar sequência de instruções
                segment = self._analyze_instruction_sequence(rom_data, i)
               # Análise do opcode para determinar tipo e tamanho da instrução
        if opcode in [0x4C, 0x6C, 0x7C]:  # JMP absoluto, indireto, indireto indexado
            instruction_size = 3 if opcode == 0x4C else 3
            if opcode == 0x4C:  # JMP absoluto
                target = rom_data[i + 1] | (rom_data[i + 2] << 8)
                jumps.append(target)
            current_segment['code_density'] += 3

        elif opcode in [0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0]:  # Branches condicionais
            instruction_size = 2
            relative_addr = rom_data[i + 1]
            # Converte para signed byte
            if relative_addr > 127:
                relative_addr = relative_addr - 256
            target = (i + 2 + relative_addr) & 0xFFFF
            branches.append(target)
            current_segment['code_density'] += 2

        elif opcode == 0x20:  # JSR
            instruction_size = 3
            target = rom_data[i + 1] | (rom_data[i + 2] << 8)
            subroutines.append(target)
            current_segment['code_density'] += 3

        elif opcode in [0x60, 0x40]:  # RTS, RTI
            instruction_size = 1
            current_segment['code_density'] += 1
            # Possível fim de função - marca para análise
            if len(current_segment['instructions']) > 5:  # Só considera se tem instruções suficientes
                current_segment['has_return'] = True

        elif opcode == 0x00:  # BRK
            instruction_size = 1
            current_segment['code_density'] += 1
            # BRK pode indicar dados ou fim de código
            consecutive_brk += 1
            if consecutive_brk > 3:
                current_segment['likely_data'] = True

        else:
            # Usar tabela de tamanhos de instruções do 6502/HuC6280
            instruction_size = get_instruction_size(opcode)
            if instruction_size == 0:  # Opcode inválido
                current_segment['invalid_opcodes'] += 1
                instruction_size = 1
            else:
                current_segment['code_density'] += instruction_size
                consecutive_brk = 0  # Reset contador de BRK

        # Verifica se temos bytes suficientes para a instrução completa
        if i + instruction_size > len(rom_data):
            break

        # Análise de padrões para identificar dados vs código
        if instruction_size > 1:
            # Coleta operandos para análise de padrões
            operands = rom_data[i + 1:i + instruction_size]

            # Verifica padrões suspeitos que podem indicar dados
            if len(operands) >= 2:
                # Sequências repetitivas podem ser dados
                if operands[0] == operands[1] and current_segment['repetitive_bytes'] < 10:
                    current_segment['repetitive_bytes'] += 1

                # Valores muito altos podem indicar dados gráficos
                if all(b > 0x7F for b in operands):
                    current_segment['high_byte_count'] += len(operands)

        # Adiciona instrução ao segmento atual
        current_segment['instructions'].append({
            'offset': i,
            'opcode': opcode,
            'size': instruction_size,
            'operands': rom_data[i + 1:i + instruction_size] if instruction_size > 1 else []
        })

        i += instruction_size
         # Verifica se devemos fechar o segmento atual
        segment_length = len(current_segment['instructions'])

        # Critérios para fechar segmento:
        # 1. Muitos opcodes inválidos (provavelmente dados)
        # 2. Padrões repetitivos demais
        # 3. Encontrou RTS/RTI (fim de função)
        # 4. Segmento ficou muito longo
        should_close_segment = (
            current_segment['invalid_opcodes'] > segment_length * 0.3 or  # 30% inválidos
            current_segment['repetitive_bytes'] > 8 or
            current_segment.get('has_return', False) or
            segment_length > 500 or  # Segmento muito longo
            consecutive_brk > 5  # Muitos BRKs seguidos
        )

        if should_close_segment:
            # Calcula métricas finais do segmento
            total_bytes = sum(instr['size'] for instr in current_segment['instructions'])
            if total_bytes > 0:
                current_segment['code_ratio'] = current_segment['code_density'] / total_bytes
                current_segment['invalid_ratio'] = current_segment['invalid_opcodes'] / segment_length
                current_segment['high_byte_ratio'] = current_segment['high_byte_count'] / total_bytes

                # Determina o tipo mais provável do segmento
                if current_segment['code_ratio'] > 0.8 and current_segment['invalid_ratio'] < 0.1:
                    current_segment['type'] = 'CODE'
                elif current_segment['high_byte_ratio'] > 0.6 or current_segment['repetitive_bytes'] > 5:
                    current_segment['type'] = 'DATA'
                else:
                    current_segment['type'] = 'MIXED'

                # Adiciona segmento à lista se tem conteúdo válido
                if segment_length > 3:  # Pelo menos 3 instruções
                    current_segment['end_offset'] = i - 1
                    segments.append(current_segment)

            # Inicia novo segmento
            current_segment = {
                'start_offset': i,
                'end_offset': i,
                'instructions': [],
                'code_density': 0,
                'invalid_opcodes': 0,
                'repetitive_bytes': 0,
                'high_byte_count': 0,
                'type': 'UNKNOWN'
            }
            consecutive_brk = 0

    # Finaliza o último segmento se houver
    if current_segment['instructions']:
        segment_length = len(current_segment['instructions'])
        total_bytes = sum(instr['size'] for instr in current_segment['instructions'])
        if total_bytes > 0:
            current_segment['code_ratio'] = current_segment['code_density'] / total_bytes
            current_segment['invalid_ratio'] = current_segment['invalid_opcodes'] / segment_length
            current_segment['high_byte_ratio'] = current_segment['high_byte_count'] / total_bytes

            if current_segment['code_ratio'] > 0.8 and current_segment['invalid_ratio'] < 0.1:
                current_segment['type'] = 'CODE'
            elif current_segment['high_byte_ratio'] > 0.6 or current_segment['repetitive_bytes'] > 5:
                current_segment['type'] = 'DATA'
            else:
                current_segment['type'] = 'MIXED'

            current_segment['end_offset'] = len(rom_data) - 1
            segments.append(current_segment)

    # Análise final e geração de relatório
    analysis_result = {
        'segments': segments,
        'total_segments': len(segments),
        'jump_targets': sorted(set(jumps)),
        'branch_targets': sorted(set(branches)),
        'subroutines': sorted(set(subroutines)),
        'code_segments': [s for s in segments if s['type'] == 'CODE'],
        'data_segments': [s for s in segments if s['type'] == 'DATA'],
        'mixed_segments': [s for s in segments if s['type'] == 'MIXED'],
        'total_code_bytes': sum(s['code_density'] for s in segments if s['type'] == 'CODE'),
        'total_data_bytes': sum(s.get('end_offset', 0) - s.get('start_offset', 0) + 1
                               for s in segments if s['type'] == 'DATA'),
        'analysis_confidence': calculate_analysis_confidence(segments)
    }

    return analysis_result
def calculate_analysis_confidence(segments):
    """
    Calcula a confiança da análise baseada na consistência dos segmentos encontrados.

    Retorna um score de 0.0 a 1.0 onde:
    - 0.9-1.0: Análise muito confiável
    - 0.7-0.9: Análise confiável
    - 0.5-0.7: Análise moderada
    - 0.3-0.5: Análise duvidosa
    - 0.0-0.3: Análise não confiável
    """
    if not segments:
        return 0.0

    confidence_factors = []

    # Fator 1: Consistência da classificação (peso: 30%)
    # Segmentos bem definidos tendem a ter ratios extremos (muito código ou muito dados)
    classification_scores = []
    for segment in segments:
        code_ratio = segment.get('code_ratio', 0)
        invalid_ratio = segment.get('invalid_ratio', 1)

        if segment['type'] == 'CODE':
            # Código bom deve ter alto code_ratio e baixo invalid_ratio
            score = code_ratio * (1 - invalid_ratio)
        elif segment['type'] == 'DATA':
            # Dados devem ter padrões claros (alto invalid_ratio ou padrões repetitivos)
            pattern_score = min(1.0, (invalid_ratio + segment.get('repetitive_bytes', 0) / 10))
            score = pattern_score
        else:  # MIXED
            # Segmentos mistos são naturalmente menos confiáveis
            score = 0.5

        classification_scores.append(score)

    avg_classification = sum(classification_scores) / len(classification_scores)
    confidence_factors.append(('classification_consistency', avg_classification, 0.30))

    # Fator 2: Distribuição de tamanhos (peso: 20%)
    # Segmentos muito pequenos ou muito grandes reduzem a confiança
    sizes = [len(s['instructions']) for s in segments]
    avg_size = sum(sizes) / len(sizes)

    # Penaliza segmentos muito pequenos (< 5 instruções) ou muito grandes (> 200)
    good_sized_segments = sum(1 for size in sizes if 5 <= size <= 200)
    size_factor = good_sized_segments / len(segments)
    confidence_factors.append(('size_distribution', size_factor, 0.20))

    # Fator 3: Coerência do fluxo de controle (peso: 25%)
    # Analisa se jumps/branches fazem sentido com os segmentos encontrados
    code_segments = [s for s in segments if s['type'] == 'CODE']
    if code_segments:
        # Verifica se há uma distribuição razoável de instruções de controle
        total_instructions = sum(len(s['instructions']) for s in code_segments)
        control_instructions = 0

        for segment in code_segments:
            for instr in segment['instructions']:
                opcode = instr['opcode']
                if opcode in [0x4C, 0x6C, 0x7C, 0x20, 0x60, 0x40,  # JMP, JSR, RTS, RTI
                             0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0]:  # Branches
                    control_instructions += 1

        if total_instructions > 0:
            control_ratio = control_instructions / total_instructions
            # Código normal tem ~5-15% de instruções de controle
            if 0.05 <= control_ratio <= 0.15:
                control_factor = 1.0
            elif 0.02 <= control_ratio <= 0.25:
                control_factor = 0.8
            else:
                control_factor = 0.5
        else:
            control_factor = 0.3
    else:
        control_factor = 0.5  # Sem código identificado é suspeito

    confidence_factors.append(('control_flow_coherence', control_factor, 0.25))

    # Fator 4: Detecção de padrões conhecidos (peso: 15%)
    # Procura por padrões típicos de ROMs de PC Engine
    pattern_score = 0.0

    # Verifica se há segmentos que parecem headers ou vetores de interrupção
    if len(segments) > 0:
        first_segment = segments[0]
        # Primeiros bytes geralmente são vetores ou header
        if first_segment.get('start_offset', 0) == 0:
            if first_segment['type'] in ['DATA', 'MIXED']:
                pattern_score += 0.3  # Esperado para área de vetores

    # Procura por padrões de padding (bytes 0x00 ou 0xFF repetidos)
    padding_found = False
    for segment in segments:
        if segment['type'] == 'DATA':
            # Verifica se o segmento tem características de padding
            if segment.get('repetitive_bytes', 0) > 10:
                padding_found = True
                break

    if padding_found:
        pattern_score += 0.3

    # Penaliza se não encontrou código algum (muito suspeito)
    if not any(s['type'] == 'CODE' for s in segments):
        pattern_score = 0.0
    else:
        pattern_score += 0.4

    confidence_factors.append(('pattern_recognition', pattern_score, 0.15))

    # Fator 5: Completude da análise (peso: 10%)
    # Verifica se a análise cobriu uma porcentagem razoável da ROM
    total_analyzed = sum(len(s['instructions']) for s in segments)
    if total_analyzed > 100:  # Pelo menos 100 instruções analisadas
        completeness_factor = min(1.0, total_analyzed / 1000)  # Normaliza até 1000 instruções
    else:
        completeness_factor = total_analyzed / 100

    confidence_factors.append(('analysis_completeness', completeness_factor, 0.10))

    # Calcula score final ponderado
    weighted_score = sum(score * weight for name, score, weight in confidence_factors)

    # Aplica ajustes finais baseados em heurísticas
    final_confidence = min(1.0, max(0.0, weighted_score))

    # Bonus para análises que encontraram boa distribuição de código/dados
    code_count = len([s for s in segments if s['type'] == 'CODE'])
    data_count = len([s for s in segments if s['type'] == 'DATA'])

    if code_count > 0 and data_count > 0:
        final_confidence = min(1.0, final_confidence * 1.1)  # 10% bonus

    return round(final_confidence, 3)


def get_instruction_size(opcode):
    """
    Retorna o tamanho da instrução baseado no opcode do HuC6280.
    Implementação simplificada - na versão final, usar tabela completa.
    """
    # Tabela simplificada dos tamanhos mais comuns
    # Formato: opcode -> tamanho_em_bytes
    size_table = {
        # Instruções de 1 byte
        0x00: 1, 0x08: 1, 0x18: 1, 0x28: 1, 0x38: 1, 0x40: 1, 0x48: 1, 0x58: 1,
        0x60: 1, 0x68: 1, 0x78: 1, 0x88: 1, 0x8A: 1, 0x98: 1, 0x9A: 1, 0xA8: 1,
        0xAA: 1, 0xB8: 1, 0xBA: 1, 0xC8: 1, 0xCA: 1, 0xD8: 1, 0xE8: 1, 0xEA: 1,
        0xF8: 1,

        # Instruções de 2 bytes (imediato, zero page, relativo)
        0x09: 2, 0x0A: 2, 0x10: 2, 0x29: 2, 0x30: 2, 0x49: 2, 0x50: 2, 0x69: 2,
        0x70: 2, 0x89: 2, 0x90: 2, 0xA0: 2, 0xA2: 2, 0xA9: 2, 0xB0: 2, 0xC0: 2,
        0xC9: 2, 0xD0: 2, 0xE0: 2, 0xE9: 2, 0xF0: 2,

        # Instruções de 3 bytes (absoluto)
        0x0D: 3, 0x1D: 3, 0x2D: 3, 0x3D: 3, 0x4C: 3, 0x4D: 3, 0x5D: 3, 0x6C: 3,
        0x6D: 3, 0x7D: 3, 0x8D: 3, 0x9D: 3, 0xAD: 3, 0xBD: 3, 0xCD: 3, 0xDD: 3,
        0xED: 3, 0xFD: 3, 0x20: 3,
    }

      return size_table.get(opcode, 0)  # Retorna 0 para opcodes desconhecidos


def disassemble_rom(rom_data, start_offset=0, length=None, base_address=0x8000):
    """
    Disassembla uma ROM do PC Engine gerando assembly legível.

    Args:
        rom_data: Dados binários da ROM
        start_offset: Offset inicial para disassembly (padrão: 0)
        length: Quantidade de bytes para disassemblar (padrão: toda a ROM)
        base_address: Endereço base para cálculo de labels (padrão: 0x8000)

    Returns:
        Dict com assembly formatado, estatísticas e informações de debug
    """
    if length is None:
        length = len(rom_data) - start_offset

    end_offset = min(start_offset + length, len(rom_data))

    # Primeiro, faz análise automática para identificar segmentos
    print("🔍 Analisando estrutura da ROM...")
    analysis = identify_code_segments(rom_data, start_offset, length)

    confidence = analysis['analysis_confidence']
    print(f"📊 Confiança da análise: {confidence:.1%}")

    if confidence < 0.5:
        print("⚠️  Análise com baixa confiança - ROM pode ter estrutura não-padrão")

    # Prepara tabelas de lookup para disassembly
    mnemonic_table = build_mnemonic_table()

    # Coleta todos os alvos de jumps/branches para criar labels
    all_targets = set()
    all_targets.update(analysis['jump_targets'])
    all_targets.update(analysis['branch_targets'])
    all_targets.update(analysis['subroutines'])

    # Gera nomes de labels
    label_names = generate_label_names(all_targets, base_address)

    # Estrutura do resultado
    result = {
        'assembly_lines': [],
        'statistics': {
            'total_bytes': end_offset - start_offset,
            'code_bytes': 0,
            'data_bytes': 0,
            'instructions_count': 0,
            'labels_count': len(label_names),
            'analysis_confidence': confidence
        },
        'segments': analysis['segments'],
        'labels': label_names,
        'warnings': []
    }

    current_address = base_address + start_offset
    i = start_offset

    # Determina qual segmento estamos processando
    current_segment = None
    for segment in analysis['segments']:
        if segment['start_offset'] <= i <= segment.get('end_offset', len(rom_data)):
            current_segment = segment
            break

    print(f"🛠️  Iniciando disassembly de {end_offset - start_offset} bytes...")

    while i < end_offset:
        # Verifica se mudou de segmento
        if current_segment is None or i > current_segment.get('end_offset', len(rom_data)):
            # Procura próximo segmento
            current_segment = None
            for segment in analysis['segments']:
                if segment['start_offset'] <= i <= segment.get('end_offset', len(rom_data)):
                    current_segment = segment
                    break

        # Adiciona label se este endereço é um target
        if current_address in all_targets:
            label_name = label_names.get(current_address, f"L{current_address:04X}")
            result['assembly_lines'].append(f"{label_name}:")

        # Determina como processar baseado no tipo de segmento
        if current_segment and current_segment['type'] == 'DATA':
            # Processa como dados
            data_line, bytes_consumed = format_data_bytes(rom_data, i, current_address,
                                                         min(16, end_offset - i))
            result['assembly_lines'].append(data_line)
            result['statistics']['data_bytes'] += bytes_consumed
            i += bytes_consumed
            current_address += bytes_consumed

        else:
            # Processa como código
            if i >= len(rom_data):
                break

            opcode = rom_data[i]
            instruction_size = get_instruction_size(opcode)

            if instruction_size == 0:
                # Opcode desconhecido - trata como byte de dados
                result['assembly_lines'].append(f"    .byte ${rom_data[i]:02X}        ; Opcode desconhecido: ${opcode:02X}")
                result['warnings'].append(f"Opcode desconhecido ${opcode:02X} em ${current_address:04X}")
                i += 1
                current_address += 1
                continue

            # Verifica se temos bytes suficientes
            if i + instruction_size > len(rom_data):
                result['warnings'].append(f"Instrução truncada em ${current_address:04X}")
                break

            # Extrai operandos
            operands = rom_data[i + 1:i + instruction_size] if instruction_size > 1 else []

            # Formata a instrução
            asm_line = format_instruction(opcode, operands, current_address,
                                        label_names, mnemonic_table)

            result['assembly_lines'].append(asm_line)
            result['statistics']['code_bytes'] += instruction_size
            result['statistics']['instructions_count'] += 1

            i += instruction_size
            current_address += instruction_size

    # Estatísticas finais
    result['statistics']['code_ratio'] = (result['statistics']['code_bytes'] /
                                         result['statistics']['total_bytes']) if result['statistics']['total_bytes'] > 0 else 0

    print(f"✅ Disassembly concluído!")
    print(f"   📈 {result['statistics']['instructions_count']} instruções")
    print(f"   🎯 {result['statistics']['labels_count']} labels")
    print(f"   📊 {result['statistics']['code_ratio']:.1%} código")

    return result


def build_mnemonic_table():
    """
    Constrói tabela de mnemonics do HuC6280.
    Versão simplificada - expandir para instrução completa.
    """
    mnemonics = {
        # Loads/Stores
        0xA9: "LDA", 0xA5: "LDA", 0xB5: "LDA", 0xAD: "LDA", 0xBD: "LDA", 0xB9: "LDA",
        0xA1: "LDA", 0xB1: "LDA",
        0xA2: "LDX", 0xA6: "LDX", 0xB6: "LDX", 0xAE: "LDX", 0xBE: "LDX",
        0xA0: "LDY", 0xA4: "LDY", 0xB4: "LDY", 0xAC: "LDY", 0xBC: "LDY",

        0x85: "STA", 0x95: "STA", 0x8D: "STA", 0x9D: "STA", 0x99: "STA",
        0x81: "STA", 0x91: "STA",
        0x86: "STX", 0x96: "STX", 0x8E: "STX",
        0x84: "STY", 0x94: "STY", 0x8C: "STY",

        # Arithmetic
        0x69: "ADC", 0x65: "ADC", 0x75: "ADC", 0x6D: "ADC", 0x7D: "ADC", 0x79: "ADC",
        0x61: "ADC", 0x71: "ADC",
        0xE9: "SBC", 0xE5: "SBC", 0xF5: "SBC", 0xED: "SBC", 0xFD: "SBC", 0xF9: "SBC",
        0xE1: "SBC", 0xF1: "SBC",

        # Logic
        0x29: "AND", 0x25: "AND", 0x35: "AND", 0x2D: "AND", 0x3D: "AND", 0x39: "AND",
        0x21: "AND", 0x31: "AND",
        0x09: "ORA", 0x05: "ORA", 0x15: "ORA", 0x0D: "ORA", 0x1D: "ORA", 0x19: "ORA",
        0x01: "ORA", 0x11: "ORA",
        0x49: "EOR", 0x45: "EOR", 0x55: "EOR", 0x4D: "EOR", 0x5D: "EOR", 0x59: "EOR",
        0x41: "EOR", 0x51: "EOR",

        # Shifts
        0x0A: "ASL", 0x06: "ASL", 0x16: "ASL", 0x0E: "ASL", 0x1E: "ASL",
        0x4A: "LSR", 0x46: "LSR", 0x56: "LSR", 0x4E: "LSR", 0x5E: "LSR",
        0x2A: "ROL", 0x26: "ROL", 0x36: "ROL", 0x2E: "ROL", 0x3E: "ROL",
        0x6A: "ROR", 0x66: "ROR", 0x76: "ROR", 0x6E: "ROR", 0x7E: "ROR",

        # Jumps/Branches
        0x4C: "JMP", 0x6C: "JMP",
        0x20: "JSR",
        0x60: "RTS", 0x40: "RTI",

        0x10: "BPL", 0x30: "BMI", 0x50: "BVC", 0x70: "BVS",
        0x90: "BCC", 0xB0: "BCS", 0xD0: "BNE", 0xF0: "BEQ",

        # Stack
        0x48: "PHA", 0x68: "PLA", 0x08: "PHP", 0x28: "PLP",

        # Flags
        0x18: "CLC", 0x38: "SEC", 0x58: "CLI", 0x78: "SEI",
        0xB8: "CLV", 0xD8: "CLD", 0xF8: "SED",

        # Misc
        0xEA: "NOP", 0x00: "BRK",
        0x8A: "TXA", 0x98: "TYA", 0xAA: "TAX", 0xA8: "TAY",
        0x9A: "TXS", 0xBA: "TSX",
        0xE8: "INX", 0xC8: "INY", 0xCA: "DEX", 0x88: "DEY",

        # Compare
        0xC9: "CMP", 0xC5: "CMP", 0xD5: "CMP", 0xCD: "CMP", 0xDD: "CMP", 0xD9: "CMP",
        0xC1: "CMP", 0xD1: "CMP",
        0xE0: "CPX", 0xE4: "CPX", 0xEC: "CPX",
        0xC0: "CPY", 0xC4: "CPY", 0xCC: "CPY",

        # Inc/Dec
        0xE6: "INC", 0xF6: "INC", 0xEE: "INC", 0xFE: "INC",
        0xC6: "DEC", 0xD6: "DEC", 0xCE: "DEC", 0xDE: "DEC",

        # Bit test
        0x24: "BIT", 0x2C: "BIT",
    }

    return mnemonics
def format_instruction(opcode, operands, current_address, label_names, mnemonic_table):
    """
    Formata uma instrução assembly com operandos apropriados.

    Args:
        opcode: Byte do opcode
        operands: Lista de bytes dos operandos
        current_address: Endereço atual da instrução
        label_names: Dict de endereços -> nomes de labels
        mnemonic_table: Tabela de mnemonics

    Returns:
        String com a instrução formatada
    """
    mnemonic = mnemonic_table.get(opcode, f"???")

    # Determina o modo de endereçamento baseado no opcode
    addressing_mode = get_addressing_mode(opcode)

    # Formata os operandos baseado no modo de endereçamento
    if addressing_mode == "IMPLIED":
        # Sem operandos (RTS, NOP, etc.)
        operand_str = ""

    elif addressing_mode == "IMMEDIATE":
        # #$XX
        operand_str = f"#${operands[0]:02X}"

    elif addressing_mode == "ZERO_PAGE":
        # $XX
        operand_str = f"${operands[0]:02X}"

    elif addressing_mode == "ZERO_PAGE_X":
        # $XX,X
        operand_str = f"${operands[0]:02X},X"

    elif addressing_mode == "ZERO_PAGE_Y":
        # $XX,Y
        operand_str = f"${operands[0]:02X},Y"

    elif addressing_mode == "ABSOLUTE":
        # $XXXX ou label se disponível
        addr = operands[0] | (operands[1] << 8)
        if addr in label_names:
            operand_str = label_names[addr]
        else:
            operand_str = f"${addr:04X}"

    elif addressing_mode == "ABSOLUTE_X":
        # $XXXX,X
        addr = operands[0] | (operands[1] << 8)
        operand_str = f"${addr:04X},X"

    elif addressing_mode == "ABSOLUTE_Y":
        # $XXXX,Y
        addr = operands[0] | (operands[1] << 8)
        operand_str = f"${addr:04X},Y"

    elif addressing_mode == "INDIRECT":
        # ($XXXX) - usado por JMP
        addr = operands[0] | (operands[1] << 8)
        operand_str = f"(${addr:04X})"

    elif addressing_mode == "INDIRECT_X":
        # ($XX,X)
        operand_str = f"(${operands[0]:02X},X)"

    elif addressing_mode == "INDIRECT_Y":
        # ($XX),Y
        operand_str = f"(${operands[0]:02X}),Y"

    elif addressing_mode == "RELATIVE":
        # Branches - calcula endereço de destino
        relative_addr = operands[0]
        # Converte para signed byte
        if relative_addr > 127:
            relative_addr = relative_addr - 256

        target_addr = (current_address + 2 + relative_addr) & 0xFFFF

        if target_addr in label_names:
            operand_str = label_names[target_addr]
        else:
            operand_str = f"${target_addr:04X}"

    elif addressing_mode == "ACCUMULATOR":
        # Shifts no acumulador (ASL A, etc.)
        operand_str = "A"

    else:
        # Modo desconhecido - mostra bytes raw
        if operands:
            operand_str = ", ".join(f"${b:02X}" for b in operands)
        else:
            operand_str = ""

    # Monta a linha completa com padding para alinhamento
    if operand_str:
        instruction = f"{mnemonic} {operand_str}"
    else:
        instruction = mnemonic

    # Adiciona comentário com bytes hex para debug
    hex_bytes = f"{opcode:02X}"
    if operands:
        hex_bytes += " " + " ".join(f"{b:02X}" for b in operands)

    # Formata com padding para alinhamento visual
    padded_instruction = f"    {instruction:<20}"
    comment = f"; ${current_address:04X}: {hex_bytes}"

    return f"{padded_instruction}{comment}"


def get_addressing_mode(opcode):
    """
    Determina o modo de endereçamento baseado no opcode.
    Implementação baseada na arquitetura do 6502/HuC6280.
    """
    # Tabela de modos de endereçamento por opcode
    # Esta é uma implementação simplificada - na prática, seria uma lookup table completa
    addressing_modes = {
        # IMPLIED (sem operandos)
        0x00: "IMPLIED",  # BRK
        0x08: "IMPLIED",  # PHP
        0x18: "IMPLIED",  # CLC
        0x28: "IMPLIED",  # PLP
        0x38: "IMPLIED",  # SEC
        0x40: "IMPLIED",  # RTI
        0x48: "IMPLIED",  # PHA
        0x58: "IMPLIED",  # CLI
        0x60: "IMPLIED",  # RTS
        0x68: "IMPLIED",  # PLA
        0x78: "IMPLIED",  # SEI
        0x88: "IMPLIED",  # DEY
        0x8A: "IMPLIED",  # TXA
        0x98: "IMPLIED",  # TYA
        0x9A: "IMPLIED",  # TXS
        0xA8: "IMPLIED",  # TAY
        0xAA: "IMPLIED",  # TAX
        0xB8: "IMPLIED",  # CLV
        0xBA: "IMPLIED",  # TSX
        0xC8: "IMPLIED",  # INY
        0xCA: "IMPLIED",  # DEX
        0xD8: "IMPLIED",  # CLD
        0xE8: "IMPLIED",  # INX
        0xEA: "IMPLIED",  # NOP
        0xF8: "IMPLIED",  # SED

        # IMMEDIATE (#$XX)
        0x09: "IMMEDIATE",  # ORA #
        0x29: "IMMEDIATE",  # AND #
        0x49: "IMMEDIATE",  # EOR #
        0x69: "IMMEDIATE",  # ADC #
        0x89: "IMMEDIATE",  # BIT # (HuC6280 específico)
        0xA0: "IMMEDIATE",  # LDY #
        0xA2: "IMMEDIATE",  # LDX #
        0xA9: "IMMEDIATE",  # LDA #
        0xC0: "IMMEDIATE",  # CPY #
        0xC9: "IMMEDIATE",  # CMP #
        0xE0: "IMMEDIATE",  # CPX #
        0xE9: "IMMEDIATE",  # SBC #

        # ZERO_PAGE ($XX)
        0x05: "ZERO_PAGE",  # ORA zp
        0x06: "ZERO_PAGE",  # ASL zp
        0x24: "ZERO_PAGE",  # BIT zp
        0x25: "ZERO_PAGE",  # AND zp
        0x26: "ZERO_PAGE",  # ROL zp
        0x45: "ZERO_PAGE",  # EOR zp
        0x46: "ZERO_PAGE",  # LSR zp
        0x65: "ZERO_PAGE",  # ADC zp
        0x66: "ZERO_PAGE",  # ROR zp
        0x84: "ZERO_PAGE",  # STY zp
        0x85: "ZERO_PAGE",  # STA zp
        0x86: "ZERO_PAGE",  # STX zp
        0xA4: "ZERO_PAGE",  # LDY zp
        0xA5: "ZERO_PAGE",  # LDA zp
        0xA6: "ZERO_PAGE",  # LDX zp
        0xC4: "ZERO_PAGE",  # CPY zp
        0xC5: "ZERO_PAGE",  # CMP zp
        0xC6: "ZERO_PAGE",  # DEC zp
        0xE4: "ZERO_PAGE",  # CPX zp
        0xE5: "ZERO_PAGE",  # SBC zp
        0xE6: "ZERO_PAGE",  # INC zp

        # ZERO_PAGE_X ($XX,X)
        0x15: "ZERO_PAGE_X",  # ORA zp,X
        0x16: "ZERO_PAGE_X",  # ASL zp,X
        0x35: "ZERO_PAGE_X",  # AND zp,X
        0x36: "ZERO_PAGE_X",  # ROL zp,X
        0x55: "ZERO_PAGE_X",  # EOR zp,X
        0x56: "ZERO_PAGE_X",  # LSR zp,X
        0x75: "ZERO_PAGE_X",  # ADC zp,X
        0x76: "ZERO_PAGE_X",  # ROR zp,X
        0x94: "ZERO_PAGE_X",  # STY zp,X
        0x95: "ZERO_PAGE_X",  # STA zp,X
        0xB4: "ZERO_PAGE_X",  # LDY zp,X
        0xB5: "ZERO_PAGE_X",  # LDA zp,X
        0xD5: "ZERO_PAGE_X",  # CMP zp,X
        0xD6: "ZERO_PAGE_X",  # DEC zp,X
        0xF5: "ZERO_PAGE_X",  # SBC zp,X
        0xF6: "ZERO_PAGE_X",  # INC zp,X

        # ZERO_PAGE_Y ($XX,Y)
        0x96: "ZERO_PAGE_Y",  # STX zp,Y
        0xB6: "ZERO_PAGE_Y",  # LDX zp,Y

        # ABSOLUTE ($XXXX)
        0x0D: "ABSOLUTE",  # ORA abs
        0x0E: "ABSOLUTE",  # ASL abs
        0x20: "ABSOLUTE",  # JSR abs
        0x2C: "ABSOLUTE",  # BIT abs
        0x2D: "ABSOLUTE",  # AND abs
        0x2E: "ABSOLUTE",  # ROL abs
        0x4C: "ABSOLUTE",  # JMP abs
        0x4D: "ABSOLUTE",  # EOR abs
        0x4E: "ABSOLUTE",  # LSR abs
        0x6D: "ABSOLUTE",  # ADC abs
        0x6E: "ABSOLUTE",  # ROR abs
        0x8C: "ABSOLUTE",  # STY abs
        0x8D: "ABSOLUTE",  # STA abs
        0x8E: "ABSOLUTE",  # STX abs
        0xAC: "ABSOLUTE",  # LDY abs
        0xAD: "ABSOLUTE",  # LDA abs
        0xAE: "ABSOLUTE",  # LDX abs
        0xCC: "ABSOLUTE",  # CPY abs
        0xCD: "ABSOLUTE",  # CMP abs
        0xCE: "ABSOLUTE",  # DEC abs
        0xEC: "ABSOLUTE",  # CPX abs
        0xED: "ABSOLUTE",  # SBC abs
        0xEE: "ABSOLUTE",  # INC abs

        # ABSOLUTE_X ($XXXX,X)
        0x1D: "ABSOLUTE_X",  # ORA abs,X
        0x1E: "ABSOLUTE_X",  # ASL abs,X
        0x3D: "ABSOLUTE_X",  # AND abs,X
        0x3E: "ABSOLUTE_X",  # ROL abs,X
        0x5D: "ABSOLUTE_X",  # EOR abs,X
        0x5E: "ABSOLUTE_X",  # LSR abs,X
        0x7D: "ABSOLUTE_X",  # ADC abs,X
        0x7E: "ABSOLUTE_X",  # ROR abs,X
        0x9D: "ABSOLUTE_X",  # STA abs,X
        0xBD: "ABSOLUTE_X",  # LDA abs,X
        0xDD: "ABSOLUTE_X",  # CMP abs,X
        0xDE: "ABSOLUTE_X",  # DEC abs,X
        0xFD: "ABSOLUTE_X",  # SBC abs,X
        0xFE: "ABSOLUTE_X",  # INC abs,X

        # ABSOLUTE_Y ($XXXX,Y)
        0x19: "ABSOLUTE_Y",  # ORA abs,Y
        0x39: "ABSOLUTE_Y",  # AND abs,Y
        0x59: "ABSOLUTE_Y",  # EOR abs,Y
        0x79: "ABSOLUTE_Y",  # ADC abs,Y
        0x99: "ABSOLUTE_Y",  # STA abs,Y
        0xB9: "ABSOLUTE_Y",  # LDA abs,Y
        0xBC: "ABSOLUTE_Y",  # LDY abs,X
        0xBE: "ABSOLUTE_Y",  # LDX abs,Y
        0xD9: "ABSOLUTE_Y",  # CMP abs,Y
        0xF9: "ABSOLUTE_Y",  # SBC abs,Y

        # INDIRECT (($XXXX))
        0x6C: "INDIRECT",  # JMP (abs)

        # INDIRECT_X (($XX,X))
        0x01: "INDIRECT_X",  # ORA (zp,X)
        0x21: "INDIRECT_X",  # AND (zp,X)
        0x41: "INDIRECT_X",  # EOR (zp,X)
        0x61: "INDIRECT_X",  # ADC (zp,X)
        0x81: "INDIRECT_X",  # STA (zp,X)
        0xA1: "INDIRECT_X",  # LDA (zp,X)
        0xC1: "INDIRECT_X",  # CMP (zp,X)
        0xE1: "INDIRECT_X",  # SBC (zp,X)

        # INDIRECT_Y (($XX),Y)
        0x11: "INDIRECT_Y",  # ORA (zp),Y
        0x31: "INDIRECT_Y",  # AND (zp),Y
        0x51: "INDIRECT_Y",  # EOR (zp),Y
        0x71: "INDIRECT_Y",  # ADC (zp),Y
        0x91: "INDIRECT_Y",  # STA (zp),Y
        0xB1: "INDIRECT_Y",  # LDA (zp),Y
        0xD1: "INDIRECT_Y",  # CMP (zp),Y
        0xF1: "INDIRECT_Y",  # SBC (zp),Y

        # RELATIVE (branches)
        0x10: "RELATIVE",  # BPL
        0x30: "RELATIVE",  # BMI
        0x50: "RELATIVE",  # BVC
        0x70: "RELATIVE",  # BVS
        0x90: "RELATIVE",  # BCC
        0xB0: "RELATIVE",  # BCS
        0xD0: "RELATIVE",  # BNE
        0xF0: "RELATIVE",  # BEQ

        # ACCUMULATOR (A)
        0x0A: "ACCUMULATOR",  # ASL A
        0x2A: "ACCUMULATOR",  # ROL A
        0x4A: "ACCUMULATOR",  # LSR A
        0x6A: "ACCUMULATOR",  # ROR A
    }

    return addressing_modes.get(opcode, "UNKNOWN")
def format_data_bytes(data_bytes, start_address=0x8000):
    """
    Formata um array de bytes em assembly 6502 legível.

    Args:
        data_bytes: Array de bytes a serem formatados
        start_address: Endereço inicial para cálculos de branches relativos

    Returns:
        String formatada em assembly
    """

    # Tabela de mnemonics para cada opcode
    MNEMONICS = {
        # ADC - Add with Carry
        0x69: "ADC", 0x65: "ADC", 0x75: "ADC", 0x6D: "ADC", 0x7D: "ADC",
        0x79: "ADC", 0x61: "ADC", 0x71: "ADC",

        # AND - Logical AND
        0x29: "AND", 0x25: "AND", 0x35: "AND", 0x2D: "AND", 0x3D: "AND",
        0x39: "AND", 0x21: "AND", 0x31: "AND",

        # ASL - Arithmetic Shift Left
        0x0A: "ASL", 0x06: "ASL", 0x16: "ASL", 0x0E: "ASL", 0x1E: "ASL",

        # BCC - Branch if Carry Clear
        0x90: "BCC",

        # BCS - Branch if Carry Set
        0xB0: "BCS",

        # BEQ - Branch if Equal
        0xF0: "BEQ",

        # BIT - Bit Test
        0x24: "BIT", 0x2C: "BIT",

        # BMI - Branch if Minus
        0x30: "BMI",

        # BNE - Branch if Not Equal
        0xD0: "BNE",

        # BPL - Branch if Positive
        0x10: "BPL",

        # BRK - Force Break
        0x00: "BRK",

        # BVC - Branch if Overflow Clear
        0x50: "BVC",

        # BVS - Branch if Overflow Set
        0x70: "BVS",

        # CLC - Clear Carry Flag
        0x18: "CLC",

        # CLD - Clear Decimal Flag
        0xD8: "CLD",

        # CLI - Clear Interrupt Flag
        0x58: "CLI",

        # CLV - Clear Overflow Flag
        0xB8: "CLV",

        # CMP - Compare
        0xC9: "CMP", 0xC5: "CMP", 0xD5: "CMP", 0xCD: "CMP", 0xDD: "CMP",
        0xD9: "CMP", 0xC1: "CMP", 0xD1: "CMP",

        # CPX - Compare X Register
        0xE0: "CPX", 0xE4: "CPX", 0xEC: "CPX",

        # CPY - Compare Y Register
        0xC0: "CPY", 0xC4: "CPY", 0xCC: "CPY",

        # DEC - Decrement Memory
        0xC6: "DEC", 0xD6: "DEC", 0xCE: "DEC", 0xDE: "DEC",

        # DEX - Decrement X Register
        0xCA: "DEX",

        # DEY - Decrement Y Register
        0x88: "DEY",

        # EOR - Exclusive OR
        0x49: "EOR", 0x45: "EOR", 0x55: "EOR", 0x4D: "EOR", 0x5D: "EOR",
        0x59: "EOR", 0x41: "EOR", 0x51: "EOR",

        # INC - Increment Memory
        0xE6: "INC", 0xF6: "INC", 0xEE: "INC", 0xFE: "INC",

        # INX - Increment X Register
        0xE8: "INX",

        # INY - Increment Y Register
        0xC8: "INY",

        # JMP - Jump
        0x4C: "JMP", 0x6C: "JMP",

        # JSR - Jump to Subroutine
        0x20: "JSR",

        # LDA - Load Accumulator
        0xA9: "LDA", 0xA5: "LDA", 0xB5: "LDA", 0xAD: "LDA", 0xBD: "LDA",
        0xB9: "LDA", 0xA1: "LDA", 0xB1: "LDA",

        # LDX - Load X Register
        0xA2: "LDX", 0xA6: "LDX", 0xB6: "LDX", 0xAE: "LDX", 0xBE: "LDX",

        # LDY - Load Y Register
        0xA0: "LDY", 0xA4: "LDY", 0xB4: "LDY", 0xAC: "LDY", 0xBC: "LDY",

        # LSR - Logical Shift Right
        0x4A: "LSR", 0x46: "LSR", 0x56: "LSR", 0x4E: "LSR", 0x5E: "LSR",

        # NOP - No Operation
        0xEA: "NOP",

        # ORA - Logical Inclusive OR
        0x09: "ORA", 0x05: "ORA", 0x15: "ORA", 0x0D: "ORA", 0x1D: "ORA",
        0x19: "ORA", 0x01: "ORA", 0x11: "ORA",

        # PHA - Push Accumulator
        0x48: "PHA",

        # PHP - Push Processor Status
        0x08: "PHP",

        # PLA - Pull Accumulator
        0x68: "PLA",

        # PLP - Pull Processor Status
        0x28: "PLP",

        # ROL - Rotate Left
        0x2A: "ROL", 0x26: "ROL", 0x36: "ROL", 0x2E: "ROL", 0x3E: "ROL",

        # ROR - Rotate Right
        0x6A: "ROR", 0x66: "ROR", 0x76: "ROR", 0x6E: "ROR", 0x7E: "ROR",

        # RTI - Return from Interrupt
        0x40: "RTI",

        # RTS - Return from Subroutine
        0x60: "RTS",

        # SBC - Subtract with Carry
        0xE9: "SBC", 0xE5: "SBC", 0xF5: "SBC", 0xED: "SBC", 0xFD: "SBC",
        0xF9: "SBC", 0xE1: "SBC", 0xF1: "SBC",

        # SEC - Set Carry Flag
        0x38: "SEC",

        # SED - Set Decimal Flag
        0xF8: "SED",

        # SEI - Set Interrupt Flag
        0x78: "SEI",

        # STA - Store Accumulator
        0x85: "STA", 0x95: "STA", 0x8D: "STA", 0x9D: "STA", 0x99: "STA",
        0x81: "STA", 0x91: "STA",

        # STX - Store X Register
        0x86: "STX", 0x96: "STX", 0x8E: "STX",

        # STY - Store Y Register
        0x84: "STY", 0x94: "STY", 0x8C: "STY",

        # TAX - Transfer Accumulator to X
        0xAA: "TAX",

        # TAY - Transfer Accumulator to Y
        0xA8: "TAY",

        # TSX - Transfer Stack Pointer to X
        0xBA: "TSX",

        # TXA - Transfer X to Accumulator
        0x8A: "TXA",

        # TXS - Transfer X to Stack Pointer
        0x9A: "TXS",

        # TYA - Transfer Y to Accumulator
        0x98: "TYA",
    }

    # Tabela de modos de endereçamento (já definida anteriormente)
    ADDRESSING_MODES = {
        0x00: "IMPLIED",     # BRK
        0x01: "INDIRECT_X",  # ORA ($44,X)
        0x05: "ZERO_PAGE",   # ORA $44
        0x06: "ZERO_PAGE",   # ASL $44
        0x08: "IMPLIED",     # PHP
        0x09: "IMMEDIATE",   # ORA #$44
        0x0A: "ACCUMULATOR", # ASL A
        0x0D: "ABSOLUTE",    # ORA $4400
        0x0E: "ABSOLUTE",    # ASL $4400
        0x10: "RELATIVE",    # BPL
        0x11: "INDIRECT_Y",  # ORA ($44),Y
        0x15: "ZERO_PAGE_X", # ORA $44,X
        0x16: "ZERO_PAGE_X", # ASL $44,X
        0x18: "IMPLIED",     # CLC
        0x19: "ABSOLUTE_Y",  # ORA $4400,Y
        0x1D: "ABSOLUTE_X",  # ORA $4400,X
        0x1E: "ABSOLUTE_X",  # ASL $4400,X
        0x20: "ABSOLUTE",    # JSR $4400
        0x21: "INDIRECT_X",  # AND ($44,X)
        0x24: "ZERO_PAGE",   # BIT $44
        0x25: "ZERO_PAGE",   # AND $44
        0x26: "ZERO_PAGE",   # ROL $44
        0x28: "IMPLIED",     # PLP
        0x29: "IMMEDIATE",   # AND #$44
        0x2A: "ACCUMULATOR", # ROL A
        0x2C: "ABSOLUTE",    # BIT $4400
        0x2D: "ABSOLUTE",    # AND $4400
        0x2E: "ABSOLUTE",    # ROL $4400
        0x30: "RELATIVE",    # BMI
        0x31: "INDIRECT_Y",  # AND ($44),Y
        0x35: "ZERO_PAGE_X", # AND $44,X
        0x36: "ZERO_PAGE_X", # ROL $44,X
        0x38: "IMPLIED",     # SEC
        0x39: "ABSOLUTE_Y",  # AND $4400,Y
        0x3D: "ABSOLUTE_X",  # AND $4400,X
        0x3E: "ABSOLUTE_X",  # ROL $4400,X
        0x40: "IMPLIED",     # RTI
        0x41: "INDIRECT_X",  # EOR ($44,X)
        0x45: "ZERO_PAGE",   # EOR $44
        0x46: "ZERO_PAGE",   # LSR $44
        0x48: "IMPLIED",     # PHA
        0x49: "IMMEDIATE",   # EOR #$44
        0x4A: "ACCUMULATOR", # LSR A
        0x4C: "ABSOLUTE",    # JMP $4400
        0x4D: "ABSOLUTE",    # EOR $4400
        0x4E: "ABSOLUTE",    # LSR $4400
        0x50: "RELATIVE",    # BVC
        0x51: "INDIRECT_Y",  # EOR ($44),Y
        0x55: "ZERO_PAGE_X", # EOR $44,X
        0x56: "ZERO_PAGE_X", # LSR $44,X
        0x58: "IMPLIED",     # CLI
        0x59: "ABSOLUTE_Y",  # EOR $4400,Y
        0x5D: "ABSOLUTE_X",  # EOR $4400,X
        0x5E: "ABSOLUTE_X",  # LSR $4400,X
        0x60: "IMPLIED",     # RTS
        0x61: "INDIRECT_X",  # ADC ($44,X)
        0x65: "ZERO_PAGE",   # ADC $44
        0x66: "ZERO_PAGE",   # ROR $44
        0x68: "IMPLIED",     # PLA
        0x69: "IMMEDIATE",   # ADC #$44
        0x6A: "ACCUMULATOR", # ROR A
        0x6C: "INDIRECT",    # JMP ($4400)
        0x6D: "ABSOLUTE",    # ADC $4400
        0x6E: "ABSOLUTE",    # ROR $4400
        0x70: "RELATIVE",    # BVS
        0x71: "INDIRECT_Y",  # ADC ($44),Y
        0x75: "ZERO_PAGE_X", # ADC $44,X
        0x76: "ZERO_PAGE_X", # ROR $44,X
        0x78: "IMPLIED",     # SEI
        0x79: "ABSOLUTE_Y",  # ADC $4400,Y
        0x7D: "ABSOLUTE_X",  # ADC $4400,X
        0x7E: "ABSOLUTE_X",  # ROR $4400,X
        0x81: "INDIRECT_X",  # STA ($44,X)
        0x84: "ZERO_PAGE",   # STY $44
        0x85: "ZERO_PAGE",   # STA $44
        0x86: "ZERO_PAGE",   # STX $44
        0x88: "IMPLIED",     # DEY
        0x8A: "IMPLIED",     # TXA
        0x8C: "ABSOLUTE",    # STY $4400
        0x8D: "ABSOLUTE",    # STA $4400
        0x8E: "ABSOLUTE",    # STX $4400
        0x90: "RELATIVE",    # BCC
        0x91: "INDIRECT_Y",  # STA ($44),Y
        0x94: "ZERO_PAGE_X", # STY $44,X
        0x95: "ZERO_PAGE_X", # STA $44,X
        0x96: "ZERO_PAGE_Y", # STX $44,Y
        0x98: "IMPLIED",     # TYA
        0x99: "ABSOLUTE_Y",  # STA $4400,Y
        0x9A: "IMPLIED",     # TXS
        0x9D: "ABSOLUTE_X",  # STA $4400,X
        0xA0: "IMMEDIATE",   # LDY #$44
        0xA1: "INDIRECT_X",  # LDA ($44,X)
        0xA2: "IMMEDIATE",   # LDX #$44
        0xA4: "ZERO_PAGE",   # LDY $44
        0xA5: "ZERO_PAGE",   # LDA $44
        0xA6: "ZERO_PAGE",   # LDX $44
        0xA8: "IMPLIED",     # TAY
        0xA9: "IMMEDIATE",   # LDA #$44
        0xAA: "IMPLIED",     # TAX
        0xAC: "ABSOLUTE",    # LDY $4400
        0xAD: "ABSOLUTE",    # LDA $4400
        0xAE: "ABSOLUTE",    # LDX $4400
        0xB0: "RELATIVE",    # BCS
        0xB1: "INDIRECT_Y",  # LDA ($44),Y
        0xB4: "ZERO_PAGE_X", # LDY $44,X
        0xB5: "ZERO_PAGE_X", # LDA $44,X
        0xB6: "ZERO_PAGE_Y", # LDX $44,Y
        0xB8: "IMPLIED",     # CLV
        0xB9: "ABSOLUTE_Y",  # LDA $4400,Y
        0xBA: "IMPLIED",     # TSX
        0xBC: "ABSOLUTE_X",  # LDY $4400,X
        0xBD: "ABSOLUTE_X",  # LDA $4400,X
        0xBE: "ABSOLUTE_Y",  # LDX $4400,Y
        0xC0: "IMMEDIATE",   # CPY #$44
        0xC1: "INDIRECT_X",  # CMP ($44,X)
        0xC4: "ZERO_PAGE",   # CPY $44
        0xC5: "ZERO_PAGE",   # CMP $44
        0xC6: "ZERO_PAGE",   # DEC $44
        0xC8: "IMPLIED",     # INY
        0xC9: "IMMEDIATE",   # CMP #$44
        0xCA: "IMPLIED",     # DEX
        0xCC: "ABSOLUTE",    # CPY $4400
        0xCD: "ABSOLUTE",    # CMP $4400
        0xCE: "ABSOLUTE",    # DEC $4400
        0xD0: "RELATIVE",    # BNE
        0xD1: "INDIRECT_Y",  # CMP ($44),Y
        0xD5: "ZERO_PAGE_X", # CMP $44,X
        0xD6: "ZERO_PAGE_X", # DEC $44,X
        0xD8: "IMPLIED",     # CLD
        0xD9: "ABSOLUTE_Y",  # CMP $4400,Y
        0xDD: "ABSOLUTE_X",  # CMP $4400,X
        0xDE: "ABSOLUTE_X",  # DEC $4400,X
        0xE0: "IMMEDIATE",   # CPX #$44
        0xE1: "INDIRECT_X",  # SBC ($44,X)
        0xE4: "ZERO_PAGE",   # CPX $44
        0xE5: "ZERO_PAGE",   # SBC $44
        0xE6: "ZERO_PAGE",   # INC $44
        0xE8: "IMPLIED",     # INX
        0xE9: "IMMEDIATE",   # SBC #$44
        0xEA: "IMPLIED",     # NOP
        0xEC: "ABSOLUTE",    # CPX $4400
        0xED: "ABSOLUTE",    # SBC $4400
        0xEE: "ABSOLUTE",    # INC $4400
        0xF0: "RELATIVE",    # BEQ
        0xF1: "INDIRECT_Y",  # SBC ($44),Y
        0xF5: "ZERO_PAGE_X", # SBC $44,X
        0xF6: "ZERO_PAGE_X", # INC $44,X
        0xF8: "IMPLIED",     # SED
        0xF9: "ABSOLUTE_Y",  # SBC $4400,Y
        0xFD: "ABSOLUTE_X",  # SBC $4400,X
        0xFE: "ABSOLUTE_X",  # INC $4400,X
    }

    result = []
    i = 0
    current_address = start_address

    while i < len(data_bytes):
        opcode = data_bytes[i]

        # Se não reconhecemos o opcode, tratamos como dados
        if opcode not in MNEMONICS:
            result.append(f"${current_address:04X}: .db ${opcode:02X}        ; Data byte")
            i += 1
            current_address += 1
            continue

        mnemonic = MNEMONICS[opcode]
        addressing_mode = ADDRESSING_MODES.get(opcode, "IMPLIED")

        # Montar a instrução baseada no modo de endereçamento
        instruction = f"${current_address:04X}: "
        bytes_consumed = 1

        if addressing_mode == "IMPLIED":
            instruction += f"{mnemonic}"

        elif addressing_mode == "ACCUMULATOR":
            instruction += f"{mnemonic} A"

        elif addressing_mode == "IMMEDIATE":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} #${data_bytes[i+1]:02X}"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} #$??"

        elif addressing_mode == "ZERO_PAGE":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} ${data_bytes[i+1]:02X}"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $??"

        elif addressing_mode == "ZERO_PAGE_X":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} ${data_bytes[i+1]:02X},X"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $??,X"

        elif addressing_mode == "ZERO_PAGE_Y":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} ${data_bytes[i+1]:02X},Y"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $??,Y"

        elif addressing_mode == "ABSOLUTE":
            if i + 2 < len(data_bytes):
                # Little endian: low byte primeiro, high byte depois
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} ${address:04X}"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} $????"

        elif addressing_mode == "ABSOLUTE_X":
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} ${address:04X},X"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} $????,X"

        elif addressing_mode == "ABSOLUTE_Y":
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} ${address:04X},Y"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} $????,Y"

        elif addressing_mode == "INDIRECT":
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} (${address:04X})"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} ($????)"

        elif addressing_mode == "INDIRECT_X":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} (${data_bytes[i+1]:02X},X)"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} ($??,X)"

        elif addressing_mode == "INDIRECT_Y":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} (${data_bytes[i+1]:02X}),Y"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} ($??),Y"

        elif addressing_mode == "RELATIVE":
            if i + 1 < len(data_bytes):
                # Branch relativo - calcular endereço destino
                offset = data_bytes[i+1]
                # Converter para signed byte
                if offset > 127:
                    offset = offset - 256
                target_address = current_address + 2 + offset
                instruction += f"{mnemonic} ${target_address:04X}"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $????"

        # Adicionar comentário com os bytes raw
        raw_bytes = " ".join(f"{data_bytes[i+j]:02X}" for j in range(bytes_consumed) if i+j < len(data_bytes))
        instruction += f"        ; {raw_bytes}"

        result.append(instruction)
        i += bytes_consumed
        current_address += bytes_consumed

    return "\n".join(result)


def detect_common_patterns(data_bytes, start_address=0x8000):
    """
    Detecta padrões comuns em código 6502 para análise de ROM hacking.

    Returns:
        dict: Dicionário com padrões detectados e suas localizações
    """
    patterns = {
        'loops': [],
        'subroutines': [],
        'data_tables': [],
        'string_tables': [],
        'sound_patterns': [],
        'graphics_patterns': [],
        'interrupt_vectors': [],
        'memory_fills': [],
        'bit_manipulations': [],
        'rng_patterns': []
    }

    i = 0
    current_address = start_address

    while i < len(data_bytes) - 2:
        # Detectar loops (padrão clássico de games)
        if (data_bytes[i] == 0xD0 and  # BNE
            i + 1 < len(data_bytes) and
            data_bytes[i+1] > 0x80):  # Branch negativo (loop back)
            offset = data_bytes[i+1] - 256
            loop_start = current_address + 2 + offset
            patterns['loops'].append({
                'address': current_address,
                'type': 'BNE_LOOP',
                'target': loop_start,
                'description': f'Loop infinito ou contador detectado'
            })

        # Detectar clear de memória (padrão super comum em ROMs)
        if (i + 5 < len(data_bytes) and
            data_bytes[i] == 0xA9 and data_bytes[i+1] == 0x00 and  # LDA #$00
            data_bytes[i+2] == 0x85 and  # STA $xx
            data_bytes[i+4] == 0xE6 and  # INC $xx
            data_bytes[i+5] == data_bytes[i+3]):  # Mesmo endereço
            patterns['memory_fills'].append({
                'address': current_address,
                'type': 'MEMORY_CLEAR',
                'description': f'Rotina de limpeza de memória detectada'
            })

        # Detectar tabelas de dados (sequência de bytes sem opcodes válidos)
        consecutive_data = 0
        temp_i = i
        while (temp_i < len(data_bytes) and
               data_bytes[temp_i] not in [0xA9, 0x85, 0xA5, 0x4C, 0x20, 0x60]):
            consecutive_data += 1
            temp_i += 1

        if consecutive_data >= 8:  # 8+ bytes consecutivos sem instruções comuns
            patterns['data_tables'].append({
                'address': current_address,
                'type': 'DATA_TABLE',
                'size': consecutive_data,
                'description': f'Tabela de dados ({consecutive_data} bytes)'
            })

        # Detectar strings ASCII (padrão para textos de jogos)
        if (i + 3 < len(data_bytes) and
            all(0x20 <= data_bytes[i+j] <= 0x7E for j in range(4))):  # 4 chars ASCII
            string_length = 0
            temp_i = i
            while (temp_i < len(data_bytes) and
                   0x20 <= data_bytes[temp_i] <= 0x7E):
                string_length += 1
                temp_i += 1

            if string_length >= 4:
                text = ''.join(chr(data_bytes[i+j]) for j in range(min(string_length, 20)))
                patterns['string_tables'].append({
                    'address': current_address,
                    'type': 'ASCII_STRING',
                    'length': string_length,
                    'preview': text,
                    'description': f'String ASCII: "{text}{"..." if string_length > 20 else ""}"'
                })

        # Detectar padrões de áudio (APU do NES)
        if (data_bytes[i] == 0x8D and i + 2 < len(data_bytes) and
            data_bytes[i+1] in [0x00, 0x04, 0x08, 0x0C] and  # APU registers
            data_bytes[i+2] == 0x40):  # $4000-$400F APU range
            patterns['sound_patterns'].append({
                'address': current_address,
                'type': 'APU_WRITE',
                'register': f'${0x4000 + data_bytes[i+1]:04X}',
                'description': f'Escrita no registrador de áudio APU'
            })

        # Detectar padrões de PPU (gráficos do NES)
        if (data_bytes[i] == 0x8D and i + 2 < len(data_bytes) and
            data_bytes[i+1] in [0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07] and
            data_bytes[i+2] == 0x20):  # $2000-$2007 PPU range
            patterns['graphics_patterns'].append({
                'address': current_address,
                'type': 'PPU_WRITE',
                'register': f'${0x2000 + data_bytes[i+1]:04X}',
                'description': f'Escrita no registrador de vídeo PPU'
            })

        # Detectar bit manipulation (muito comum em games)
        if (data_bytes[i] in [0x09, 0x29, 0x49] and  # ORA, AND, EOR
            i + 1 < len(data_bytes)):
            bit_mask = data_bytes[i+1]
            operation = {0x09: 'ORA', 0x29: 'AND', 0x49: 'EOR'}[data_bytes[i]]
            patterns['bit_manipulations'].append({
                'address': current_address,
                'type': 'BIT_MANIPULATION',
                'operation': operation,
                'mask': f'${bit_mask:02X}',
                'description': f'Manipulação de bits: {operation} #{bit_mask:02X}'
            })

        # Detectar possível RNG (Linear Feedback Shift Register)
        if (i + 7 < len(data_bytes) and
            data_bytes[i] == 0xA5 and  # LDA $xx
            data_bytes[i+2] == 0x0A and  # ASL
            data_bytes[i+3] == 0x90 and  # BCC
            data_bytes[i+5] == 0x49):  # EOR
            patterns['rng_patterns'].append({
                'address': current_address,
                'type': 'LFSR_RNG',
                'description': 'Possível gerador de números aleatórios (LFSR)'
            })

        # Detectar JSR/RTS (subroutines)
        if data_bytes[i] == 0x20:  # JSR
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                target = (high_byte << 8) | low_byte
                patterns['subroutines'].append({
                    'address': current_address,
                    'type': 'JSR_CALL',
                    'target': target,
                    'description': f'Chamada de subrotina para ${target:04X}'
                })

        # Detectar interrupt vectors (final da ROM)
        if (current_address >= 0xFFFA and current_address <= 0xFFFF):
            patterns['interrupt_vectors'].append({
                'address': current_address,
                'type': 'INTERRUPT_VECTOR',
                'description': 'Vetor de interrupção (NMI/RESET/IRQ)'
            })

        i += 1
        current_address += 1

    return patterns


def format_data_bytes_with_analysis(data_bytes, start_address=0x8000):
    """
    Versão melhorada que combina disassembly com análise de padrões.
    """
    # Primeiro, faça o disassembly normal
    disassembly = format_data_bytes(data_bytes, start_address)

    # Depois, detecte padrões
    patterns = detect_common_patterns(data_bytes, start_address)

    # Adicione comentários de análise
    lines = disassembly.split('\n')
    enhanced_lines = []

    for line in lines:
        enhanced_lines.append(line)

        # Extrair endereço da linha
        if line.startswith('):
            try:
                addr_str = line.split(':')[0][1:]  # Remove $ e pega até :
                addr = int(addr_str, 16)

                # Verificar se há padrões neste endereço
                for pattern_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if pattern['address'] == addr:
                            enhanced_lines.append(f"                     ; ** {pattern['description']} **")

                            # Adicionar informações extras para alguns padrões
                            if pattern_type == 'loops':
                                enhanced_lines.append(f"                     ;    Target: ${pattern['target']:04X}")
                            elif pattern_type == 'data_tables':
                                enhanced_lines.append(f"                     ;    Size: {pattern['size']} bytes")
                            elif pattern_type == 'string_tables':
                                enhanced_lines.append(f"                     ;    Length: {pattern['length']} chars")

            except (ValueError, IndexError):
                continue

    # Adicionar sumário de análise
    analysis_summary = [
        "",
        "=== ANÁLISE DE PADRÕES DETECTADOS ===",
        ""
    ]

    total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
    if total_patterns == 0:
        analysis_summary.append("Nenhum padrão específico detectado.")
    else:
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                pattern_names = {
                    'loops': 'Loops e Contadores',
                    'subroutines': 'Chamadas de Subrotinas',
                    'data_tables': 'Tabelas de Dados',
                    'string_tables': 'Strings ASCII',
                    'sound_patterns': 'Padrões de Áudio (APU)',
                    'graphics_patterns': 'Padrões de Vídeo (PPU)',
                    'interrupt_vectors': 'Vetores de Interrupção',
                    'memory_fills': 'Preenchimento de Memória',
                    'bit_manipulations': 'Manipulação de Bits',
                    'rng_patterns': 'Geradores de Números Aleatórios'
                }

                analysis_summary.append(f"• {pattern_names.get(pattern_type, pattern_type)}: {len(pattern_list)} ocorrência(s)")

                # Mostrar detalhes dos primeiros padrões
                for i, pattern in enumerate(pattern_list[:3]):
                    analysis_summary.append(f"  - ${pattern['address']:04X}: {pattern['description']}")

                if len(pattern_list) > 3:
                    analysis_summary.append(f"  ... e mais {len(pattern_list) - 3} padrões")

                analysis_summary.append("")

    return '\n'.join(enhanced_lines) + '\n' + '\n'.join(analysis_summary)


# Exemplo de uso com análise avançada:
if __name__ == "__main__":
    # Teste com padrões mais complexos
    test_bytes = [
        0xA9, 0x00,        # LDA #$00
        0x85, 0x10,        # STA $10
        0xE6, 0x10,        # INC $10         ; MEMORY_FILL pattern
        0xA5, 0x10,        # LDA $10
        0xC9, 0x0A,        # CMP #$0A
        0xD0, 0xFA,        # BNE $8006       ; LOOP pattern
        0x20, 0x50, 0x80,  # JSR $8050       ; SUBROUTINE pattern
        0x29, 0x0F,        # AND #$0F        ; BIT_MANIPULATION pattern
        0x8D, 0x00, 0x20,  # STA $2000       ; PPU_WRITE pattern
        0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00,  # ASCII "Hello"
        0x4C, 0x00, 0x80   # JMP $8000
    ]

   print("=== DISASSEMBLY COM ANÁLISE DE PADRÕES ===")
    print(format_data_bytes_with_analysis(test_bytes, 0x8000))):
            try:
                addr_str = line.split(':')[0][1:]  # Remove $ e pega até :
                addr = int(addr_str, 16)

                # Verificar se há padrões neste endereço
                for pattern_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if pattern['address'] == addr:
                            enhanced_lines.append(f"                     ; ** {pattern['description']} **")

                            # Adicionar informações extras para padrões específicos
                            if pattern_type == 'loops':
                                enhanced_lines.append(f"                     ;    Target: ${pattern['target']:04X}")
                            elif pattern_type == 'data_tables':
                                enhanced_lines.append(f"                     ;    Size: {pattern['size']} bytes")
                            elif pattern_type == 'string_tables' or pattern_type == 'dialog_systems':
                                enhanced_lines.append(f"                     ;    Length: {pattern['length']} chars")
                            elif pattern_type == 'enemy_patterns':
                                enhanced_lines.append(f"                     ;    HP: {pattern['hp']}, ATK: {pattern['attack']}, DEF: {pattern['defense']}")
                            elif pattern_type == 'sprite_animations':
                                enhanced_lines.append(f"                     ;    Frames: {pattern['frames']}, Speed: {pattern['speed']}")
                            elif pattern_type == 'dungeon_logic':
                                enhanced_lines.append(f"                     ;    Position: ({pattern['x']}, {pattern['y']}) Level: {pattern['level']}")

            except (ValueError, IndexError):
                continue

    # Adicionar sumário de análise especializado
    analysis_summary = [
        "",
        f"=== ANÁLISE DE PADRÕES NEUTOPIA/TG-16 ===",
        ""
    ]

    # Contadores especiais para Neutopia
    neutopia_specific_patterns = ['map_data', 'item_tables', 'enemy_patterns', 'dialog_systems',
                                 'save_systems', 'collision_detection', 'sprite_animations', 'dungeon_logic']
    neutopia_count = sum(len(patterns[p]) for p in neutopia_specific_patterns if p in patterns)

    total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())

    if total_patterns == 0:
        analysis_summary.append("Nenhum padrão específico detectado.")
    else:
        analysis_summary.append(f"Total de padrões detectados: {total_patterns}")
        analysis_summary.append(f"Padrões específicos de Action-RPG: {neutopia_count}")
        analysis_summary.append("")

        # Padrões importantes para ROM hacking de RPGs
        pattern_names = {
            'loops': 'Loops e Contadores',
            'subroutines': 'Chamadas de Subrotinas',def format_data_bytes(data_bytes, start_address=0x8000):
    """
    Formata um array de bytes em assembly 6502 legível.

    Args:
        data_bytes: Array de bytes a serem formatados
        start_address: Endereço inicial para cálculos de branches relativos

    Returns:
        String formatada em assembly
    """

    # Tabela de mnemonics para cada opcode
    MNEMONICS = {
        # ADC - Add with Carry
        0x69: "ADC", 0x65: "ADC", 0x75: "ADC", 0x6D: "ADC", 0x7D: "ADC",
        0x79: "ADC", 0x61: "ADC", 0x71: "ADC",

        # AND - Logical AND
        0x29: "AND", 0x25: "AND", 0x35: "AND", 0x2D: "AND", 0x3D: "AND",
        0x39: "AND", 0x21: "AND", 0x31: "AND",

        # ASL - Arithmetic Shift Left
        0x0A: "ASL", 0x06: "ASL", 0x16: "ASL", 0x0E: "ASL", 0x1E: "ASL",

        # BCC - Branch if Carry Clear
        0x90: "BCC",

        # BCS - Branch if Carry Set
        0xB0: "BCS",

        # BEQ - Branch if Equal
        0xF0: "BEQ",

        # BIT - Bit Test
        0x24: "BIT", 0x2C: "BIT",

        # BMI - Branch if Minus
        0x30: "BMI",

        # BNE - Branch if Not Equal
        0xD0: "BNE",

        # BPL - Branch if Positive
        0x10: "BPL",

        # BRK - Force Break
        0x00: "BRK",

        # BVC - Branch if Overflow Clear
        0x50: "BVC",

        # BVS - Branch if Overflow Set
        0x70: "BVS",

        # CLC - Clear Carry Flag
        0x18: "CLC",

        # CLD - Clear Decimal Flag
        0xD8: "CLD",

        # CLI - Clear Interrupt Flag
        0x58: "CLI",

        # CLV - Clear Overflow Flag
        0xB8: "CLV",

        # CMP - Compare
        0xC9: "CMP", 0xC5: "CMP", 0xD5: "CMP", 0xCD: "CMP", 0xDD: "CMP",
        0xD9: "CMP", 0xC1: "CMP", 0xD1: "CMP",

        # CPX - Compare X Register
        0xE0: "CPX", 0xE4: "CPX", 0xEC: "CPX",

        # CPY - Compare Y Register
        0xC0: "CPY", 0xC4: "CPY", 0xCC: "CPY",

        # DEC - Decrement Memory
        0xC6: "DEC", 0xD6: "DEC", 0xCE: "DEC", 0xDE: "DEC",

        # DEX - Decrement X Register
        0xCA: "DEX",

        # DEY - Decrement Y Register
        0x88: "DEY",

        # EOR - Exclusive OR
        0x49: "EOR", 0x45: "EOR", 0x55: "EOR", 0x4D: "EOR", 0x5D: "EOR",
        0x59: "EOR", 0x41: "EOR", 0x51: "EOR",

        # INC - Increment Memory
        0xE6: "INC", 0xF6: "INC", 0xEE: "INC", 0xFE: "INC",

        # INX - Increment X Register
        0xE8: "INX",

        # INY - Increment Y Register
        0xC8: "INY",

        # JMP - Jump
        0x4C: "JMP", 0x6C: "JMP",

        # JSR - Jump to Subroutine
        0x20: "JSR",

        # LDA - Load Accumulator
        0xA9: "LDA", 0xA5: "LDA", 0xB5: "LDA", 0xAD: "LDA", 0xBD: "LDA",
        0xB9: "LDA", 0xA1: "LDA", 0xB1: "LDA",

        # LDX - Load X Register
        0xA2: "LDX", 0xA6: "LDX", 0xB6: "LDX", 0xAE: "LDX", 0xBE: "LDX",

        # LDY - Load Y Register
        0xA0: "LDY", 0xA4: "LDY", 0xB4: "LDY", 0xAC: "LDY", 0xBC: "LDY",

        # LSR - Logical Shift Right
        0x4A: "LSR", 0x46: "LSR", 0x56: "LSR", 0x4E: "LSR", 0x5E: "LSR",

        # NOP - No Operation
        0xEA: "NOP",

        # ORA - Logical Inclusive OR
        0x09: "ORA", 0x05: "ORA", 0x15: "ORA", 0x0D: "ORA", 0x1D: "ORA",
        0x19: "ORA", 0x01: "ORA", 0x11: "ORA",

        # PHA - Push Accumulator
        0x48: "PHA",

        # PHP - Push Processor Status
        0x08: "PHP",

        # PLA - Pull Accumulator
        0x68: "PLA",

        # PLP - Pull Processor Status
        0x28: "PLP",

        # ROL - Rotate Left
        0x2A: "ROL", 0x26: "ROL", 0x36: "ROL", 0x2E: "ROL", 0x3E: "ROL",

        # ROR - Rotate Right
        0x6A: "ROR", 0x66: "ROR", 0x76: "ROR", 0x6E: "ROR", 0x7E: "ROR",

        # RTI - Return from Interrupt
        0x40: "RTI",

        # RTS - Return from Subroutine
        0x60: "RTS",

        # SBC - Subtract with Carry
        0xE9: "SBC", 0xE5: "SBC", 0xF5: "SBC", 0xED: "SBC", 0xFD: "SBC",
        0xF9: "SBC", 0xE1: "SBC", 0xF1: "SBC",

        # SEC - Set Carry Flag
        0x38: "SEC",

        # SED - Set Decimal Flag
        0xF8: "SED",

        # SEI - Set Interrupt Flag
        0x78: "SEI",

        # STA - Store Accumulator
        0x85: "STA", 0x95: "STA", 0x8D: "STA", 0x9D: "STA", 0x99: "STA",
        0x81: "STA", 0x91: "STA",

        # STX - Store X Register
        0x86: "STX", 0x96: "STX", 0x8E: "STX",

        # STY - Store Y Register
        0x84: "STY", 0x94: "STY", 0x8C: "STY",

        # TAX - Transfer Accumulator to X
        0xAA: "TAX",

        # TAY - Transfer Accumulator to Y
        0xA8: "TAY",

        # TSX - Transfer Stack Pointer to X
        0xBA: "TSX",

        # TXA - Transfer X to Accumulator
        0x8A: "TXA",

        # TXS - Transfer X to Stack Pointer
        0x9A: "TXS",

        # TYA - Transfer Y to Accumulator
        0x98: "TYA",
    }

    # Tabela de modos de endereçamento (já definida anteriormente)
    ADDRESSING_MODES = {
        0x00: "IMPLIED",     # BRK
        0x01: "INDIRECT_X",  # ORA ($44,X)
        0x05: "ZERO_PAGE",   # ORA $44
        0x06: "ZERO_PAGE",   # ASL $44
        0x08: "IMPLIED",     # PHP
        0x09: "IMMEDIATE",   # ORA #$44
        0x0A: "ACCUMULATOR", # ASL A
        0x0D: "ABSOLUTE",    # ORA $4400
        0x0E: "ABSOLUTE",    # ASL $4400
        0x10: "RELATIVE",    # BPL
        0x11: "INDIRECT_Y",  # ORA ($44),Y
        0x15: "ZERO_PAGE_X", # ORA $44,X
        0x16: "ZERO_PAGE_X", # ASL $44,X
        0x18: "IMPLIED",     # CLC
        0x19: "ABSOLUTE_Y",  # ORA $4400,Y
        0x1D: "ABSOLUTE_X",  # ORA $4400,X
        0x1E: "ABSOLUTE_X",  # ASL $4400,X
        0x20: "ABSOLUTE",    # JSR $4400
        0x21: "INDIRECT_X",  # AND ($44,X)
        0x24: "ZERO_PAGE",   # BIT $44
        0x25: "ZERO_PAGE",   # AND $44
        0x26: "ZERO_PAGE",   # ROL $44
        0x28: "IMPLIED",     # PLP
        0x29: "IMMEDIATE",   # AND #$44
        0x2A: "ACCUMULATOR", # ROL A
        0x2C: "ABSOLUTE",    # BIT $4400
        0x2D: "ABSOLUTE",    # AND $4400
        0x2E: "ABSOLUTE",    # ROL $4400
        0x30: "RELATIVE",    # BMI
        0x31: "INDIRECT_Y",  # AND ($44),Y
        0x35: "ZERO_PAGE_X", # AND $44,X
        0x36: "ZERO_PAGE_X", # ROL $44,X
        0x38: "IMPLIED",     # SEC
        0x39: "ABSOLUTE_Y",  # AND $4400,Y
        0x3D: "ABSOLUTE_X",  # AND $4400,X
        0x3E: "ABSOLUTE_X",  # ROL $4400,X
        0x40: "IMPLIED",     # RTI
        0x41: "INDIRECT_X",  # EOR ($44,X)
        0x45: "ZERO_PAGE",   # EOR $44
        0x46: "ZERO_PAGE",   # LSR $44
        0x48: "IMPLIED",     # PHA
        0x49: "IMMEDIATE",   # EOR #$44
        0x4A: "ACCUMULATOR", # LSR A
        0x4C: "ABSOLUTE",    # JMP $4400
        0x4D: "ABSOLUTE",    # EOR $4400
        0x4E: "ABSOLUTE",    # LSR $4400
        0x50: "RELATIVE",    # BVC
        0x51: "INDIRECT_Y",  # EOR ($44),Y
        0x55: "ZERO_PAGE_X", # EOR $44,X
        0x56: "ZERO_PAGE_X", # LSR $44,X
        0x58: "IMPLIED",     # CLI
        0x59: "ABSOLUTE_Y",  # EOR $4400,Y
        0x5D: "ABSOLUTE_X",  # EOR $4400,X
        0x5E: "ABSOLUTE_X",  # LSR $4400,X
        0x60: "IMPLIED",     # RTS
        0x61: "INDIRECT_X",  # ADC ($44,X)
        0x65: "ZERO_PAGE",   # ADC $44
        0x66: "ZERO_PAGE",   # ROR $44
        0x68: "IMPLIED",     # PLA
        0x69: "IMMEDIATE",   # ADC #$44
        0x6A: "ACCUMULATOR", # ROR A
        0x6C: "INDIRECT",    # JMP ($4400)
        0x6D: "ABSOLUTE",    # ADC $4400
        0x6E: "ABSOLUTE",    # ROR $4400
        0x70: "RELATIVE",    # BVS
        0x71: "INDIRECT_Y",  # ADC ($44),Y
        0x75: "ZERO_PAGE_X", # ADC $44,X
        0x76: "ZERO_PAGE_X", # ROR $44,X
        0x78: "IMPLIED",     # SEI
        0x79: "ABSOLUTE_Y",  # ADC $4400,Y
        0x7D: "ABSOLUTE_X",  # ADC $4400,X
        0x7E: "ABSOLUTE_X",  # ROR $4400,X
        0x81: "INDIRECT_X",  # STA ($44,X)
        0x84: "ZERO_PAGE",   # STY $44
        0x85: "ZERO_PAGE",   # STA $44
        0x86: "ZERO_PAGE",   # STX $44
        0x88: "IMPLIED",     # DEY
        0x8A: "IMPLIED",     # TXA
        0x8C: "ABSOLUTE",    # STY $4400
        0x8D: "ABSOLUTE",    # STA $4400
        0x8E: "ABSOLUTE",    # STX $4400
        0x90: "RELATIVE",    # BCC
        0x91: "INDIRECT_Y",  # STA ($44),Y
        0x94: "ZERO_PAGE_X", # STY $44,X
        0x95: "ZERO_PAGE_X", # STA $44,X
        0x96: "ZERO_PAGE_Y", # STX $44,Y
        0x98: "IMPLIED",     # TYA
        0x99: "ABSOLUTE_Y",  # STA $4400,Y
        0x9A: "IMPLIED",     # TXS
        0x9D: "ABSOLUTE_X",  # STA $4400,X
        0xA0: "IMMEDIATE",   # LDY #$44
        0xA1: "INDIRECT_X",  # LDA ($44,X)
        0xA2: "IMMEDIATE",   # LDX #$44
        0xA4: "ZERO_PAGE",   # LDY $44
        0xA5: "ZERO_PAGE",   # LDA $44
        0xA6: "ZERO_PAGE",   # LDX $44
        0xA8: "IMPLIED",     # TAY
        0xA9: "IMMEDIATE",   # LDA #$44
        0xAA: "IMPLIED",     # TAX
        0xAC: "ABSOLUTE",    # LDY $4400
        0xAD: "ABSOLUTE",    # LDA $4400
        0xAE: "ABSOLUTE",    # LDX $4400
        0xB0: "RELATIVE",    # BCS
        0xB1: "INDIRECT_Y",  # LDA ($44),Y
        0xB4: "ZERO_PAGE_X", # LDY $44,X
        0xB5: "ZERO_PAGE_X", # LDA $44,X
        0xB6: "ZERO_PAGE_Y", # LDX $44,Y
        0xB8: "IMPLIED",     # CLV
        0xB9: "ABSOLUTE_Y",  # LDA $4400,Y
        0xBA: "IMPLIED",     # TSX
        0xBC: "ABSOLUTE_X",  # LDY $4400,X
        0xBD: "ABSOLUTE_X",  # LDA $4400,X
        0xBE: "ABSOLUTE_Y",  # LDX $4400,Y
        0xC0: "IMMEDIATE",   # CPY #$44
        0xC1: "INDIRECT_X",  # CMP ($44,X)
        0xC4: "ZERO_PAGE",   # CPY $44
        0xC5: "ZERO_PAGE",   # CMP $44
        0xC6: "ZERO_PAGE",   # DEC $44
        0xC8: "IMPLIED",     # INY
        0xC9: "IMMEDIATE",   # CMP #$44
        0xCA: "IMPLIED",     # DEX
        0xCC: "ABSOLUTE",    # CPY $4400
        0xCD: "ABSOLUTE",    # CMP $4400
        0xCE: "ABSOLUTE",    # DEC $4400
        0xD0: "RELATIVE",    # BNE
        0xD1: "INDIRECT_Y",  # CMP ($44),Y
        0xD5: "ZERO_PAGE_X", # CMP $44,X
        0xD6: "ZERO_PAGE_X", # DEC $44,X
        0xD8: "IMPLIED",     # CLD
        0xD9: "ABSOLUTE_Y",  # CMP $4400,Y
        0xDD: "ABSOLUTE_X",  # CMP $4400,X
        0xDE: "ABSOLUTE_X",  # DEC $4400,X
        0xE0: "IMMEDIATE",   # CPX #$44
        0xE1: "INDIRECT_X",  # SBC ($44,X)
        0xE4: "ZERO_PAGE",   # CPX $44
        0xE5: "ZERO_PAGE",   # SBC $44
        0xE6: "ZERO_PAGE",   # INC $44
        0xE8: "IMPLIED",     # INX
        0xE9: "IMMEDIATE",   # SBC #$44
        0xEA: "IMPLIED",     # NOP
        0xEC: "ABSOLUTE",    # CPX $4400
        0xED: "ABSOLUTE",    # SBC $4400
        0xEE: "ABSOLUTE",    # INC $4400
        0xF0: "RELATIVE",    # BEQ
        0xF1: "INDIRECT_Y",  # SBC ($44),Y
        0xF5: "ZERO_PAGE_X", # SBC $44,X
        0xF6: "ZERO_PAGE_X", # INC $44,X
        0xF8: "IMPLIED",     # SED
        0xF9: "ABSOLUTE_Y",  # SBC $4400,Y
        0xFD: "ABSOLUTE_X",  # SBC $4400,X
        0xFE: "ABSOLUTE_X",  # INC $4400,X
    }

    result = []
    i = 0
    current_address = start_address

    while i < len(data_bytes):
        opcode = data_bytes[i]

        # Se não reconhecemos o opcode, tratamos como dados
        if opcode not in MNEMONICS:
            result.append(f"${current_address:04X}: .db ${opcode:02X}        ; Data byte")
            i += 1
            current_address += 1
            continue

        mnemonic = MNEMONICS[opcode]
        addressing_mode = ADDRESSING_MODES.get(opcode, "IMPLIED")

        # Montar a instrução baseada no modo de endereçamento
        instruction = f"${current_address:04X}: "
        bytes_consumed = 1

        if addressing_mode == "IMPLIED":
            instruction += f"{mnemonic}"

        elif addressing_mode == "ACCUMULATOR":
            instruction += f"{mnemonic} A"

        elif addressing_mode == "IMMEDIATE":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} #${data_bytes[i+1]:02X}"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} #$??"

        elif addressing_mode == "ZERO_PAGE":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} ${data_bytes[i+1]:02X}"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $??"

        elif addressing_mode == "ZERO_PAGE_X":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} ${data_bytes[i+1]:02X},X"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $??,X"

        elif addressing_mode == "ZERO_PAGE_Y":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} ${data_bytes[i+1]:02X},Y"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $??,Y"

        elif addressing_mode == "ABSOLUTE":
            if i + 2 < len(data_bytes):
                # Little endian: low byte primeiro, high byte depois
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} ${address:04X}"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} $????"

        elif addressing_mode == "ABSOLUTE_X":
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} ${address:04X},X"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} $????,X"

        elif addressing_mode == "ABSOLUTE_Y":
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} ${address:04X},Y"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} $????,Y"

        elif addressing_mode == "INDIRECT":
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                address = (high_byte << 8) | low_byte
                instruction += f"{mnemonic} (${address:04X})"
                bytes_consumed = 3
            else:
                instruction += f"{mnemonic} ($????)"

        elif addressing_mode == "INDIRECT_X":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} (${data_bytes[i+1]:02X},X)"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} ($??,X)"

        elif addressing_mode == "INDIRECT_Y":
            if i + 1 < len(data_bytes):
                instruction += f"{mnemonic} (${data_bytes[i+1]:02X}),Y"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} ($??),Y"

        elif addressing_mode == "RELATIVE":
            if i + 1 < len(data_bytes):
                # Branch relativo - calcular endereço destino
                offset = data_bytes[i+1]
                # Converter para signed byte
                if offset > 127:
                    offset = offset - 256
                target_address = current_address + 2 + offset
                instruction += f"{mnemonic} ${target_address:04X}"
                bytes_consumed = 2
            else:
                instruction += f"{mnemonic} $????"

        # Adicionar comentário com os bytes raw
        raw_bytes = " ".join(f"{data_bytes[i+j]:02X}" for j in range(bytes_consumed) if i+j < len(data_bytes))
        instruction += f"        ; {raw_bytes}"

        result.append(instruction)
        i += bytes_consumed
        current_address += bytes_consumed

    return "\n".join(result)


def detect_common_patterns(data_bytes, start_address=0x8000, console_type="generic"):
    """
    Detecta padrões comuns em código 6502 para análise de ROM hacking.

    Args:
        data_bytes: Array de bytes da ROM
        start_address: Endereço inicial
        console_type: "generic", "nes", "tg16", "neutopia" para padrões específicos

    Returns:
        dict: Dicionário com padrões detectados e suas localizações
    """
    patterns = {
        'loops': [],
        'subroutines': [],
        'data_tables': [],
        'string_tables': [],
        'sound_patterns': [],
        'graphics_patterns': [],
        'interrupt_vectors': [],
        'memory_fills': [],
        'bit_manipulations': [],
        'rng_patterns': [],
        'map_data': [],
        'item_tables': [],
        'enemy_patterns': [],
        'dialog_systems': [],
        'save_systems': [],
        'collision_detection': [],
        'sprite_animations': [],
        'dungeon_logic': []
    }

    i = 0
    current_address = start_address

    while i < len(data_bytes) - 2:
        # === PADRÕES BÁSICOS (UNIVERSAL) ===

        # Detectar loops (padrão clássico de games)
        if (data_bytes[i] == 0xD0 and  # BNE
            i + 1 < len(data_bytes) and
            data_bytes[i+1] > 0x80):  # Branch negativo (loop back)
            offset = data_bytes[i+1] - 256
            loop_start = current_address + 2 + offset
            patterns['loops'].append({
                'address': current_address,
                'type': 'BNE_LOOP',
                'target': loop_start,
                'description': f'Loop infinito ou contador detectado'
            })

        # Detectar clear de memória (padrão super comum em ROMs)
        if (i + 5 < len(data_bytes) and
            data_bytes[i] == 0xA9 and data_bytes[i+1] == 0x00 and  # LDA #$00
            data_bytes[i+2] == 0x85 and  # STA $xx
            data_bytes[i+4] == 0xE6 and  # INC $xx
            data_bytes[i+5] == data_bytes[i+3]):  # Mesmo endereço
            patterns['memory_fills'].append({
                'address': current_address,
                'type': 'MEMORY_CLEAR',
                'description': f'Rotina de limpeza de memória detectada'
            })

        # === PADRÕES ESPECÍFICOS DO TURBOGRAFX-16 ===

        if console_type in ["tg16", "neutopia"]:
            # TG-16 VDC (Video Display Controller) - $0000-$0003
            if (data_bytes[i] == 0x8D and i + 2 < len(data_bytes) and
                data_bytes[i+1] in [0x00, 0x02, 0x03] and
                data_bytes[i+2] == 0x00):  # VDC registers
                patterns['graphics_patterns'].append({
                    'address': current_address,
                    'type': 'TG16_VDC_WRITE',
                    'register': f'VDC_REG_{data_bytes[i+1]:02X}',
                    'description': f'Escrita no VDC (Video Display Controller)'
                })

            # TG-16 PSG (Programmable Sound Generator) - $0800-$0809
            if (data_bytes[i] == 0x8D and i + 2 < len(data_bytes) and
                data_bytes[i+1] in [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09] and
                data_bytes[i+2] == 0x08):  # PSG registers
                patterns['sound_patterns'].append({
                    'address': current_address,
                    'type': 'TG16_PSG_WRITE',
                    'register': f'PSG_CH{data_bytes[i+1]}',
                    'description': f'Escrita no PSG canal {data_bytes[i+1]}'
                })

        # === PADRÕES ESPECÍFICOS DO NEUTOPIA ===

        if console_type == "neutopia":
            # Detectar padrões de mapa tile-based (comum em RPGs)
            if (i + 7 < len(data_bytes) and
                all(data_bytes[i+j] < 0x80 for j in range(8)) and  # Tiles válidos
                len(set(data_bytes[i:i+8])) > 3):  # Variação suficiente
                patterns['map_data'].append({
                    'address': current_address,
                    'type': 'TILE_MAP_DATA',
                    'description': 'Possível dados de mapa/tiles detectados'
                })

            # Detectar tabelas de itens (padrão comum em RPGs)
            if (i + 5 < len(data_bytes) and
                data_bytes[i] in range(0x01, 0x64) and  # ID de item válido
                data_bytes[i+1] in range(0x00, 0xFF) and  # Tipo/propriedade
                data_bytes[i+2] in range(0x00, 0x64)):  # Quantidade ou valor
                patterns['item_tables'].append({
                    'address': current_address,
                    'type': 'ITEM_TABLE_ENTRY',
                    'item_id': data_bytes[i],
                    'description': f'Entrada de tabela de itens (ID: {data_bytes[i]})'
                })

            # Detectar padrões de inimigos
            if (i + 6 < len(data_bytes) and
                data_bytes[i] in range(0x01, 0x40) and  # ID de inimigo
                data_bytes[i+1] in range(0x01, 0xFF) and  # HP
                data_bytes[i+2] in range(0x01, 0x32) and  # Attack
                data_bytes[i+3] in range(0x00, 0x20)):  # Defense
                patterns['enemy_patterns'].append({
                    'address': current_address,
                    'type': 'ENEMY_STATS',
                    'enemy_id': data_bytes[i],
                    'hp': data_bytes[i+1],
                    'attack': data_bytes[i+2],
                    'defense': data_bytes[i+3],
                    'description': f'Stats de inimigo (ID: {data_bytes[i]}, HP: {data_bytes[i+1]})'
                })

            # Detectar padrões de diálogo/texto
            if (i + 3 < len(data_bytes) and
                data_bytes[i] == 0x00 and  # Possível terminador
                all(0x20 <= data_bytes[i+j] <= 0x7E for j in range(1, 4))):  # ASCII válido
                string_length = 0
                temp_i = i + 1
                while (temp_i < len(data_bytes) and
                       0x20 <= data_bytes[temp_i] <= 0x7E):
                    string_length += 1
                    temp_i += 1

                if string_length >= 8:  # Strings longas = diálogo
                    text = ''.join(chr(data_bytes[i+1+j]) for j in range(min(string_length, 30)))
                    patterns['dialog_systems'].append({
                        'address': current_address,
                        'type': 'DIALOG_TEXT',
                        'length': string_length,
                        'preview': text,
                        'description': f'Texto de diálogo: "{text}{"..." if string_length > 30 else ""}"'
                    })

            # Detectar padrões de save/load
            if (i + 4 < len(data_bytes) and
                data_bytes[i] == 0xA9 and  # LDA #$xx
                data_bytes[i+2] == 0x8D and  # STA $xxxx
                data_bytes[i+3] >= 0x60 and data_bytes[i+4] >= 0x60):  # Endereços altos (save RAM)
                patterns['save_systems'].append({
                    'address': current_address,
                    'type': 'SAVE_OPERATION',
                    'description': 'Possível operação de save/load detectada'
                })

            # Detectar padrões de animação de sprites
            if (i + 8 < len(data_bytes) and
                data_bytes[i] in range(0x00, 0x10) and  # Frame count
                data_bytes[i+1] in range(0x00, 0x10) and  # Speed
                all(data_bytes[i+j] < 0x80 for j in range(2, 6))):  # Sprite IDs
                patterns['sprite_animations'].append({
                    'address': current_address,
                    'type': 'SPRITE_ANIMATION',
                    'frames': data_bytes[i],
                    'speed': data_bytes[i+1],
                    'description': f'Animação de sprite ({data_bytes[i]} frames, speed {data_bytes[i+1]})'
                })

            # Detectar lógica de dungeons (doors, keys, etc.)
            if (i + 4 < len(data_bytes) and
                data_bytes[i] in [0x01, 0x02, 0x03, 0x04] and  # Tipos de porta/chave
                data_bytes[i+1] in range(0x00, 0x10) and  # Posição X
                data_bytes[i+2] in range(0x00, 0x10) and  # Posição Y
                data_bytes[i+3] in range(0x00, 0x08)):  # Nível/andar
                patterns['dungeon_logic'].append({
                    'address': current_address,
                    'type': 'DOOR_KEY_DATA',
                    'door_type': data_bytes[i],
                    'x': data_bytes[i+1],
                    'y': data_bytes[i+2],
                    'level': data_bytes[i+3],
                    'description': f'Porta/chave tipo {data_bytes[i]} em ({data_bytes[i+1]},{data_bytes[i+2]}) nível {data_bytes[i+3]}'
                })

        # === PADRÕES UNIVERSAIS CONTINUAÇÃO ===

        # Detectar tabelas de dados (sequência de bytes sem opcodes válidos)
        consecutive_data = 0
        temp_i = i
        while (temp_i < len(data_bytes) and
               data_bytes[temp_i] not in [0xA9, 0x85, 0xA5, 0x4C, 0x20, 0x60]):
            consecutive_data += 1
            temp_i += 1

        if consecutive_data >= 8:  # 8+ bytes consecutivos sem instruções comuns
            patterns['data_tables'].append({
                'address': current_address,
                'type': 'DATA_TABLE',
                'size': consecutive_data,
                'description': f'Tabela de dados ({consecutive_data} bytes)'
            })

        # Detectar strings ASCII genéricas
        if (i + 3 < len(data_bytes) and
            all(0x20 <= data_bytes[i+j] <= 0x7E for j in range(4))):  # 4 chars ASCII
            string_length = 0
            temp_i = i
            while (temp_i < len(data_bytes) and
                   0x20 <= data_bytes[temp_i] <= 0x7E):
                string_length += 1
                temp_i += 1

            if string_length >= 4:
                text = ''.join(chr(data_bytes[i+j]) for j in range(min(string_length, 20)))
                patterns['string_tables'].append({
                    'address': current_address,
                    'type': 'ASCII_STRING',
                    'length': string_length,
                    'preview': text,
                    'description': f'String ASCII: "{text}{"..." if string_length > 20 else ""}"'
                })

        # === PADRÕES DE HARDWARE NES (para compatibilidade) ===

        if console_type == "nes":
            # Detectar padrões de áudio (APU do NES)
            if (data_bytes[i] == 0x8D and i + 2 < len(data_bytes) and
                data_bytes[i+1] in [0x00, 0x04, 0x08, 0x0C] and  # APU registers
                data_bytes[i+2] == 0x40):  # $4000-$400F APU range
                patterns['sound_patterns'].append({
                    'address': current_address,
                    'type': 'APU_WRITE',
                    'register': f'${0x4000 + data_bytes[i+1]:04X}',
                    'description': f'Escrita no registrador de áudio APU'
                })

            # Detectar padrões de PPU (gráficos do NES)
            if (data_bytes[i] == 0x8D and i + 2 < len(data_bytes) and
                data_bytes[i+1] in [0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07] and
                data_bytes[i+2] == 0x20):  # $2000-$2007 PPU range
                patterns['graphics_patterns'].append({
                    'address': current_address,
                    'type': 'PPU_WRITE',
                    'register': f'${0x2000 + data_bytes[i+1]:04X}',
                    'description': f'Escrita no registrador de vídeo PPU'
                })

        # === PADRÕES AVANÇADOS UNIVERSAIS ===

        # Detectar bit manipulation (muito comum em games)
        if (data_bytes[i] in [0x09, 0x29, 0x49] and  # ORA, AND, EOR
            i + 1 < len(data_bytes)):
            bit_mask = data_bytes[i+1]
            operation = {0x09: 'ORA', 0x29: 'AND', 0x49: 'EOR'}[data_bytes[i]]
            patterns['bit_manipulations'].append({
                'address': current_address,
                'type': 'BIT_MANIPULATION',
                'operation': operation,
                'mask': f'${bit_mask:02X}',
                'description': f'Manipulação de bits: {operation} #{bit_mask:02X}'
            })

        # Detectar possível RNG (Linear Feedback Shift Register)
        if (i + 7 < len(data_bytes) and
            data_bytes[i] == 0xA5 and  # LDA $xx
            data_bytes[i+2] == 0x0A and  # ASL
            data_bytes[i+3] == 0x90 and  # BCC
            data_bytes[i+5] == 0x49):  # EOR
            patterns['rng_patterns'].append({
                'address': current_address,
                'type': 'LFSR_RNG',
                'description': 'Possível gerador de números aleatórios (LFSR)'
            })

        # Detectar JSR/RTS (subroutines)
        if data_bytes[i] == 0x20:  # JSR
            if i + 2 < len(data_bytes):
                low_byte = data_bytes[i+1]
                high_byte = data_bytes[i+2]
                target = (high_byte << 8) | low_byte
                patterns['subroutines'].append({
                    'address': current_address,
                    'type': 'JSR_CALL',
                    'target': target,
                    'description': f'Chamada de subrotina para ${target:04X}'
                })

        # Detectar collision detection patterns
        if (i + 6 < len(data_bytes) and
            data_bytes[i] == 0xA5 and  # LDA $xx (player X)
            data_bytes[i+2] == 0x18 and  # CLC
            data_bytes[i+3] == 0x69 and  # ADC #$xx (hitbox size)
            data_bytes[i+5] == 0xC5):  # CMP $xx (enemy X)
            patterns['collision_detection'].append({
                'address': current_address,
                'type': 'COLLISION_CHECK',
                'description': 'Possível rotina de detecção de colisão'
            })

        # Detectar interrupt vectors (final da ROM)
        if (current_address >= 0xFFFA and current_address <= 0xFFFF):
            patterns['interrupt_vectors'].append({
                'address': current_address,
                'type': 'INTERRUPT_VECTOR',
                'description': 'Vetor de interrupção (NMI/RESET/IRQ)'
            })

        i += 1
        current_address += 1

    return patterns


def format_data_bytes_with_analysis(data_bytes, start_address=0x8000):
    """
    Versão melhorada que combina disassembly com análise de padrões.
    """
    # Primeiro, faça o disassembly normal
    disassembly = format_data_bytes(data_bytes, start_address)

    # Depois, detecte padrões
    patterns = detect_common_patterns(data_bytes, start_address)

    # Adicione comentários de análise
    lines = disassembly.split('\n')
    enhanced_lines = []

    for line in lines:
        enhanced_lines.append(line)

        # Extrair endereço da linha
        if line.startswith('):
            try:
                addr_str = line.split(':')[0][1:]  # Remove $ e pega até :
                addr = int(addr_str, 16)

                # Verificar se há padrões neste endereço
                for pattern_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if pattern['address'] == addr:
                            enhanced_lines.append(f"                     ; ** {pattern['description']} **")

                            # Adicionar informações extras para alguns padrões
                            if pattern_type == 'loops':
                                enhanced_lines.append(f"                     ;    Target: ${pattern['target']:04X}")
                            elif pattern_type == 'data_tables':
                                enhanced_lines.append(f"                     ;    Size: {pattern['size']} bytes")
                            elif pattern_type == 'string_tables':
                                enhanced_lines.append(f"                     ;    Length: {pattern['length']} chars")

            except (ValueError, IndexError):
                continue

    # Adicionar sumário de análise
    analysis_summary = [
        "",
        "=== ANÁLISE DE PADRÕES DETECTADOS ===",
        ""
    ]

    total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
    if total_patterns == 0:
        analysis_summary.append("Nenhum padrão específico detectado.")
    else:
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                pattern_names = {
                    'loops': 'Loops e Contadores',
                    'subroutines': 'Chamadas de Subrotinas',
                    'data_tables': 'Tabelas de Dados',
                    'string_tables': 'Strings ASCII',
                    'sound_patterns': 'Padrões de Áudio (APU)',
                    'graphics_patterns': 'Padrões de Vídeo (PPU)',
                    'interrupt_vectors': 'Vetores de Interrupção',
                    'memory_fills': 'Preenchimento de Memória',
                    'bit_manipulations': 'Manipulação de Bits',
                    'rng_patterns': 'Geradores de Números Aleatórios'
                }

                analysis_summary.append(f"• {pattern_names.get(pattern_type, pattern_type)}: {len(pattern_list)} ocorrência(s)")

                # Mostrar detalhes dos primeiros padrões
                for i, pattern in enumerate(pattern_list[:3]):
                    analysis_summary.append(f"  - ${pattern['address']:04X}: {pattern['description']}")

                if len(pattern_list) > 3:
                    analysis_summary.append(f"  ... e mais {len(pattern_list) - 3} padrões")

                analysis_summary.append("")

    return '\n'.join(enhanced_lines) + '\n' + '\n'.join(analysis_summary)


# Exemplo de uso com análise avançada:
if __name__ == "__main__":
    # Teste com padrões mais complexos
    test_bytes = [
        0xA9, 0x00,        # LDA #$00
        0x85, 0x10,        # STA $10
        0xE6, 0x10,        # INC $10         ; MEMORY_FILL pattern
        0xA5, 0x10,        # LDA $10
        0xC9, 0x0A,        # CMP #$0A
        0xD0, 0xFA,        # BNE $8006       ; LOOP pattern
        0x20, 0x50, 0x80,  # JSR $8050       ; SUBROUTINE pattern
        0x29, 0x0F,        # AND #$0F        ; BIT_MANIPULATION pattern
        0x8D, 0x00, 0x20,  # STA $2000       ; PPU_WRITE pattern
        0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00,  # ASCII "Hello"
        0x4C, 0x00, 0x80   # JMP $8000
    ]

    print("=== DISASSEMBLY COM ANÁLISE DE PADRÕES ===")
    print(format_data_bytes_with_analysis(test_bytes, 0x8000))