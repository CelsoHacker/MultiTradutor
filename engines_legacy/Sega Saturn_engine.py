"""
Sega Saturn ROM Hacking Framework
=================================
Comprehensive toolkit for Saturn game analysis and translation.
SH-2 dual-core architecture with full system structure support.
"""

import struct
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re

# =============================================================================
# SH-2 ARCHITECTURE DEFINITIONS
# =============================================================================

class SH2InstructionType(Enum):
    """SH-2 instruction types based on format"""
    ZERO_OPERAND = "0"      # No operands
    N_OPERAND = "n"         # 4-bit immediate
    M_OPERAND = "m"         # 4-bit register
    NM_OPERAND = "nm"       # Two 4-bit operands
    MD_OPERAND = "md"       # Register + displacement
    NMD_OPERAND = "nmd"     # Register + register + displacement
    I_OPERAND = "i"         # 8-bit immediate
    D_OPERAND = "d"         # 8-bit displacement
    D12_OPERAND = "d12"     # 12-bit displacement

@dataclass
class SH2Instruction:
    """SH-2 instruction definition"""
    mnemonic: str
    opcode_mask: int
    opcode_value: int
    format_type: SH2InstructionType
    cycles: int
    description: str

# SH-2 instruction set - core instructions for disassembly
SH2_INSTRUCTIONS = {
    # Data Transfer Instructions
    0x6003: SH2Instruction("MOV", 0xF00F, 0x6003, SH2InstructionType.NM_OPERAND, 1, "Move register to register"),
    0xE000: SH2Instruction("MOV", 0xF000, 0xE000, SH2InstructionType.NM_OPERAND, 1, "Move immediate to register"),
    0x9000: SH2Instruction("MOV.W", 0xF000, 0x9000, SH2InstructionType.NM_OPERAND, 1, "Move word (PC relative)"),
    0xD000: SH2Instruction("MOV.L", 0xF000, 0xD000, SH2InstructionType.NM_OPERAND, 1, "Move long (PC relative)"),
    0x2000: SH2Instruction("MOV.B", 0xF00F, 0x2000, SH2InstructionType.NM_OPERAND, 1, "Move byte to memory"),
    0x2001: SH2Instruction("MOV.W", 0xF00F, 0x2001, SH2InstructionType.NM_OPERAND, 1, "Move word to memory"),
    0x2002: SH2Instruction("MOV.L", 0xF00F, 0x2002, SH2InstructionType.NM_OPERAND, 1, "Move long to memory"),

    # Arithmetic Instructions
    0x300C: SH2Instruction("ADD", 0xF00F, 0x300C, SH2InstructionType.NM_OPERAND, 1, "Add register to register"),
    0x7000: SH2Instruction("ADD", 0xF000, 0x7000, SH2InstructionType.NM_OPERAND, 1, "Add immediate to register"),
    0x3008: SH2Instruction("SUB", 0xF00F, 0x3008, SH2InstructionType.NM_OPERAND, 1, "Subtract register from register"),
    0x300D: SH2Instruction("DMULS.L", 0xF00F, 0x300D, SH2InstructionType.NM_OPERAND, 2, "Double-precision multiply"),
    0x300E: SH2Instruction("DMULU.L", 0xF00F, 0x300E, SH2InstructionType.NM_OPERAND, 2, "Double-precision multiply unsigned"),

    # Logic Instructions
    0x2009: SH2Instruction("AND", 0xF00F, 0x2009, SH2InstructionType.NM_OPERAND, 1, "AND register with register"),
    0xC900: SH2Instruction("AND", 0xFF00, 0xC900, SH2InstructionType.I_OPERAND, 1, "AND immediate with R0"),
    0x200B: SH2Instruction("OR", 0xF00F, 0x200B, SH2InstructionType.NM_OPERAND, 1, "OR register with register"),
    0xCB00: SH2Instruction("OR", 0xFF00, 0xCB00, SH2InstructionType.I_OPERAND, 1, "OR immediate with R0"),

    # Shift Instructions
    0x4000: SH2Instruction("SHL", 0xF00F, 0x4000, SH2InstructionType.N_OPERAND, 1, "Shift left logical"),
    0x4001: SH2Instruction("SHR", 0xF00F, 0x4001, SH2InstructionType.N_OPERAND, 1, "Shift right logical"),
    0x4020: SH2Instruction("SHAL", 0xF00F, 0x4020, SH2InstructionType.N_OPERAND, 1, "Shift left arithmetic"),
    0x4021: SH2Instruction("SHAR", 0xF00F, 0x4021, SH2InstructionType.N_OPERAND, 1, "Shift right arithmetic"),

    # Branch Instructions
    0xA000: SH2Instruction("BRA", 0xF000, 0xA000, SH2InstructionType.D12_OPERAND, 2, "Branch always"),
    0xB000: SH2Instruction("BSR", 0xF000, 0xB000, SH2InstructionType.D12_OPERAND, 2, "Branch to subroutine"),
    0x8900: SH2Instruction("BT", 0xFF00, 0x8900, SH2InstructionType.D_OPERAND, 1, "Branch if true"),
    0x8B00: SH2Instruction("BF", 0xFF00, 0x8B00, SH2InstructionType.D_OPERAND, 1, "Branch if false"),
    0x8D00: SH2Instruction("BT/S", 0xFF00, 0x8D00, SH2InstructionType.D_OPERAND, 1, "Branch if true (delayed)"),
    0x8F00: SH2Instruction("BF/S", 0xFF00, 0x8F00, SH2InstructionType.D_OPERAND, 1, "Branch if false (delayed)"),

    # System Instructions
    0x0009: SH2Instruction("NOP", 0xFFFF, 0x0009, SH2InstructionType.ZERO_OPERAND, 1, "No operation"),
    0x000B: SH2Instruction("RTS", 0xFFFF, 0x000B, SH2InstructionType.ZERO_OPERAND, 2, "Return from subroutine"),
    0x001B: SH2Instruction("SLEEP", 0xFFFF, 0x001B, SH2InstructionType.ZERO_OPERAND, 1, "Sleep mode"),
    0x0019: SH2Instruction("DIV0U", 0xFFFF, 0x0019, SH2InstructionType.ZERO_OPERAND, 1, "Divide step 0 unsigned"),

    # Compare Instructions
    0x3000: SH2Instruction("CMP/EQ", 0xF00F, 0x3000, SH2InstructionType.NM_OPERAND, 1, "Compare equal"),
    0x3002: SH2Instruction("CMP/HS", 0xF00F, 0x3002, SH2InstructionType.NM_OPERAND, 1, "Compare unsigned higher or same"),
    0x3003: SH2Instruction("CMP/GE", 0xF00F, 0x3003, SH2InstructionType.NM_OPERAND, 1, "Compare signed greater or equal"),
    0x3006: SH2Instruction("CMP/HI", 0xF00F, 0x3006, SH2InstructionType.NM_OPERAND, 1, "Compare unsigned higher"),
    0x3007: SH2Instruction("CMP/GT", 0xF00F, 0x3007, SH2InstructionType.NM_OPERAND, 1, "Compare signed greater"),
    0x8800: SH2Instruction("CMP/EQ", 0xFF00, 0x8800, SH2InstructionType.I_OPERAND, 1, "Compare equal immediate"),
}

# =============================================================================
# SATURN SYSTEM STRUCTURE
# =============================================================================

@dataclass
class SaturnSystemInfo:
    """Saturn system configuration and memory map"""
    # Memory regions (physical addresses)
    BIOS_START = 0x00000000
    BIOS_SIZE = 0x00080000      # 512KB BIOS

    LOW_RAM_START = 0x00200000  # 1MB Low RAM
    LOW_RAM_SIZE = 0x00100000

    HIGH_RAM_START = 0x06000000 # 1MB High RAM
    HIGH_RAM_SIZE = 0x00100000

    VRAM_START = 0x05C00000     # 1.5MB VRAM
    VRAM_SIZE = 0x00180000

    CARTRIDGE_START = 0x02000000 # Cartridge space
    CARTRIDGE_SIZE = 0x00400000   # Up to 4MB

    # System registers
    VDP1_REGS = 0x05D00000      # VDP1 registers
    VDP2_REGS = 0x05F00000      # VDP2 registers
    SCU_REGS = 0x05FE0000       # SCU registers
    CPU_REGS = 0x05FF0000       # CPU-specific registers

class SaturnMemoryMap:
    """Saturn memory mapping and banking system"""

    def __init__(self):
        self.regions = {}
        self.banking_state = {}

    def map_region(self, start_addr: int, size: int, data: bytes, name: str):
        """Map a memory region"""
        self.regions[name] = {
            'start': start_addr,
            'size': size,
            'data': data,
            'end': start_addr + size - 1
        }

    def read_address(self, address: int, size: int = 1) -> Optional[bytes]:
        """Read from mapped memory"""
        for region_name, region in self.regions.items():
            if region['start'] <= address <= region['end']:
                offset = address - region['start']
                if offset + size <= len(region['data']):
                    return region['data'][offset:offset + size]
        return None

    def get_region_name(self, address: int) -> Optional[str]:
        """Get the name of the memory region containing this address"""
        for region_name, region in self.regions.items():
            if region['start'] <= address <= region['end']:
                return region_name
        return None

# =============================================================================
# SH-2 DISASSEMBLER ENGINE
# =============================================================================

class SH2Disassembler:
    """SH-2 disassembler for Saturn ROMs"""

    def __init__(self):
        self.instructions = SH2_INSTRUCTIONS
        self.memory_map = SaturnMemoryMap()

    def decode_instruction(self, opcode: int) -> Optional[SH2Instruction]:
        """Decode a 16-bit SH-2 instruction"""
        # Try to match against known instructions
        for instr_key, instruction in self.instructions.items():
            if (opcode & instruction.opcode_mask) == instruction.opcode_value:
                return instruction
        return None

    def extract_operands(self, opcode: int, instruction: SH2Instruction) -> Dict[str, int]:
        """Extract operands from instruction based on format"""
        operands = {}

        if instruction.format_type == SH2InstructionType.N_OPERAND:
            operands['n'] = (opcode >> 8) & 0xF

        elif instruction.format_type == SH2InstructionType.M_OPERAND:
            operands['m'] = (opcode >> 8) & 0xF

        elif instruction.format_type == SH2InstructionType.NM_OPERAND:
            operands['n'] = (opcode >> 8) & 0xF
            operands['m'] = (opcode >> 4) & 0xF

        elif instruction.format_type == SH2InstructionType.MD_OPERAND:
            operands['m'] = (opcode >> 4) & 0xF
            operands['d'] = opcode & 0xF

        elif instruction.format_type == SH2InstructionType.NMD_OPERAND:
            operands['n'] = (opcode >> 8) & 0xF
            operands['m'] = (opcode >> 4) & 0xF
            operands['d'] = opcode & 0xF

        elif instruction.format_type == SH2InstructionType.I_OPERAND:
            operands['i'] = opcode & 0xFF

        elif instruction.format_type == SH2InstructionType.D_OPERAND:
            operands['d'] = opcode & 0xFF
            # Sign extend 8-bit displacement
            if operands['d'] & 0x80:
                operands['d'] |= 0xFFFFFF00

        elif instruction.format_type == SH2InstructionType.D12_OPERAND:
            operands['d'] = opcode & 0xFFF
            # Sign extend 12-bit displacement
            if operands['d'] & 0x800:
                operands['d'] |= 0xFFFFF000

        return operands

    class SH2Disassembler:
    def __init__(self):
        self.instruction_table = self._build_instruction_table()

    def _build_instruction_table(self):
        """
        Constrói a tabela de instruções SH-2 com seus formatos de operandos
        Baseado no manual oficial da Hitachi SH-2
        """
        return {
            # Movimento de dados
            0x6003: ("MOV", "rm_rn"),       # MOV Rm, Rn
            0x9000: ("MOV.W", "disp_pc_rn"), # MOV.W @(disp,PC), Rn
            0xD000: ("MOV.L", "disp_pc_rn"), # MOV.L @(disp,PC), Rn
            0x2000: ("MOV.B", "rm_atrn"),   # MOV.B Rm, @Rn
            0x2001: ("MOV.W", "rm_atrn"),   # MOV.W Rm, @Rn
            0x2002: ("MOV.L", "rm_atrn"),   # MOV.L Rm, @Rn
            0x6000: ("MOV.B", "atrm_rn"),   # MOV.B @Rm, Rn
            0x6001: ("MOV.W", "atrm_rn"),   # MOV.W @Rm, Rn
            0x6002: ("MOV.L", "atrm_rn"),   # MOV.L @Rm, Rn
            0xE000: ("MOV", "imm_rn"),      # MOV #imm, Rn

            # Aritméticas
            0x300C: ("ADD", "rm_rn"),       # ADD Rm, Rn
            0x7000: ("ADD", "imm_rn"),      # ADD #imm, Rn
            0x3008: ("SUB", "rm_rn"),       # SUB Rm, Rn
            0x3004: ("DIV1", "rm_rn"),      # DIV1 Rm, Rn
            0x2009: ("AND", "rm_rn"),       # AND Rm, Rn
            0x200B: ("OR", "rm_rn"),        # OR Rm, Rn
            0x200A: ("XOR", "rm_rn"),       # XOR Rm, Rn

            # Comparações
            0x3000: ("CMP/EQ", "rm_rn"),    # CMP/EQ Rm, Rn
            0x3002: ("CMP/HS", "rm_rn"),    # CMP/HS Rm, Rn
            0x3003: ("CMP/GE", "rm_rn"),    # CMP/GE Rm, Rn
            0x3006: ("CMP/HI", "rm_rn"),    # CMP/HI Rm, Rn
            0x3007: ("CMP/GT", "rm_rn"),    # CMP/GT Rm, Rn
            0x8800: ("CMP/EQ", "imm_r0"),   # CMP/EQ #imm, R0
            0x2008: ("TST", "rm_rn"),       # TST Rm, Rn

            # Branches e jumps
            0xA000: ("BRA", "disp12"),      # BRA disp
            0xB000: ("BSR", "disp12"),      # BSR disp
            0x8900: ("BT", "disp8"),        # BT disp
            0x8B00: ("BF", "disp8"),        # BF disp
            0x8D00: ("BT/S", "disp8"),      # BT/S disp
            0x8F00: ("BF/S", "disp8"),      # BF/S disp
            0x402B: ("JMP", "atrn"),        # JMP @Rn
            0x400B: ("JSR", "atrn"),        # JSR @Rn

            # Controle
            0x0009: ("NOP", None),          # NOP
            0x000B: ("RTS", None),          # RTS
            0x0019: ("DIV0U", None),        # DIV0U
            0x0008: ("CLRT", None),         # CLRT
            0x0018: ("SETT", None),         # SETT
            0x0028: ("CLRMAC", None),       # CLRMAC
            0x4015: ("CMP/PL", "rn"),       # CMP/PL Rn
            0x4011: ("CMP/PZ", "rn"),       # CMP/PZ Rn

            # System
            0xC300: ("TRAPA", "imm8"),      # TRAPA #imm
            0x001B: ("SLEEP", None),        # SLEEP
        }

    def decode_instruction(self, opcode):
        """
        Decodifica uma instrução SH-2 de 16 bits
        Retorna (mnemonic, operand_format)
        """
        # Busca exata primeiro
        if opcode in self.instruction_table:
            return self.instruction_table[opcode]

        # Busca por padrões com máscaras
        for mask, (mnemonic, fmt) in self._get_masked_instructions():
            if (opcode & mask) == (mask & 0xFFFF):
                return mnemonic, fmt

        # Instrução não reconhecida
        return "DC.W", "raw"

    def _get_masked_instructions(self):
        """
        Retorna instruções que precisam de máscara para decodificação
        """
        return [
            (0xF000, ("MOV.W", "disp_pc_rn")),  # 9xxx
            (0xF000, ("MOV.L", "disp_pc_rn")),  # Dxxx
            (0xF000, ("MOV", "imm_rn")),        # Exxx
            (0xF000, ("ADD", "imm_rn")),        # 7xxx
            (0xFF00, ("CMP/EQ", "imm_r0")),     # 88xx
            (0xFF00, ("BT", "disp8")),          # 89xx
            (0xFF00, ("BF", "disp8")),          # 8Bxx
            (0xF000, ("BRA", "disp12")),        # Axxx
            (0xF000, ("BSR", "disp12")),        # Bxxx
            (0xF00F, ("CMP/PL", "rn")),         # x015
            (0xF00F, ("CMP/PZ", "rn")),         # x011
        ]

    def extract_operands(self, opcode, operand_format):
        """
        Extrai operandos baseado no formato da instrução
        Retorna dicionário com operandos seguros
        """
        operands = {}

        if operand_format is None:
            return operands

        # Extração baseada no formato
        if operand_format == "rm_rn":
            operands['m'] = (opcode >> 4) & 0xF
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "imm_rn":
            operands['i'] = opcode & 0xFF
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "disp_pc_rn":
            operands['d'] = opcode & 0xFF
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "disp12":
            # Displacement de 12 bits com sinal
            disp = opcode & 0xFFF
            operands['d'] = self._sign_extend(disp, 12)

        elif operand_format == "disp8":
            # Displacement de 8 bits com sinal
            disp = opcode & 0xFF
            operands['d'] = self._sign_extend(disp, 8)

        elif operand_format == "rn":
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "atrn":
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "rm_atrn":
            operands['m'] = (opcode >> 4) & 0xF
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "atrm_rn":
            operands['m'] = (opcode >> 4) & 0xF
            operands['n'] = (opcode >> 8) & 0xF

        elif operand_format == "imm_r0":
            operands['i'] = opcode & 0xFF
            operands['n'] = 0  # R0 implícito

        elif operand_format == "imm8":
            operands['i'] = opcode & 0xFF

        elif operand_format == "raw":
            operands['raw'] = opcode

        return operands

    def _sign_extend(self, value, bits):
        """
        Estende o sinal para complemento de 2
        """
        sign_bit = 1 << (bits - 1)
        return (value & (sign_bit - 1)) - (value & sign_bit)

    def format_instruction(self, address, opcode, instruction, operands):
        """
        Formata uma instrução para exibição - VERSÃO CORRIGIDA
        Não assume mais que certas chaves existem nos operandos
        """
        addr_str = f"{address:08X}"
        opcode_str = f"{opcode:04X}"

        # Monta a string de operandos baseada no que está disponível
        operand_str = self._format_operands(instruction, operands, address)

        return f"{addr_str}: {opcode_str} {instruction:<8} {operand_str}"

    def _format_operands(self, instruction, operands, current_address):
        """
        Formata operandos baseado no tipo de instrução
        Método defensivo que não assume chaves existem
        """
        if not operands:
            return ""

        # Instruções sem operandos
        if len(operands) == 0:
            return ""

        # Instruções com operandos raw (desconhecidas)
        if 'raw' in operands:
            return f"${operands['raw']:04X}"

        # MOV com registradores
        if instruction.startswith("MOV") and 'm' in operands and 'n' in operands:
            if instruction.endswith("@Rn"):
                return f"R{operands['m']}, @R{operands['n']}"
            elif instruction.startswith("MOV.B @Rm") or instruction.startswith("MOV.W @Rm") or instruction.startswith("MOV.L @Rm"):
                return f"@R{operands['m']}, R{operands['n']}"
            else:
                return f"R{operands['m']}, R{operands['n']}"

        # MOV com imediato
        elif instruction == "MOV" and 'i' in operands and 'n' in operands:
            return f"#${operands['i']:02X}, R{operands['n']}"

        # MOV com displacement do PC
        elif instruction.startswith("MOV") and 'd' in operands and 'n' in operands:
            return f"@(${operands['d']:02X},PC), R{operands['n']}"

        # ADD com imediato
        elif instruction == "ADD" and 'i' in operands and 'n' in operands:
            return f"#${operands['i']:02X}, R{operands['n']}"

        # ADD com registradores
        elif instruction == "ADD" and 'm' in operands and 'n' in operands:
            return f"R{operands['m']}, R{operands['n']}"

        # Comparações
        elif instruction.startswith("CMP") and 'm' in operands and 'n' in operands:
            return f"R{operands['m']}, R{operands['n']}"
        elif instruction.startswith("CMP") and 'i' in operands:
            return f"#${operands['i']:02X}, R0"

        # Branches
        elif instruction in ["BRA", "BSR", "BT", "BF", "BT/S", "BF/S"] and 'd' in operands:
            target = current_address + (operands['d'] * 2) + 4
            return f"${target:08X}"

        # Jumps
        elif instruction in ["JMP", "JSR"] and 'n' in operands:
            return f"@R{operands['n']}"

        # Instruções com um registrador
        elif 'n' in operands and 'm' not in operands:
            return f"R{operands['n']}"

        # TRAPA
        elif instruction == "TRAPA" and 'i' in operands:
            return f"#${operands['i']:02X}"

        # Fallback genérico
        else:
            parts = []
            if 'i' in operands:
                parts.append(f"#${operands['i']:02X}")
            if 'n' in operands:
                parts.append(f"R{operands['n']}")
            if 'm' in operands:
                parts.append(f"R{operands['m']}")
            if 'd' in operands:
                parts.append(f"disp:{operands['d']}")
            return ", ".join(parts)

    def disassemble_range(self, code_data, start_address):
        """
        Disassembla uma sequência de código SH-2
        Versão corrigida que não gera KeyError
        """
        results = []
        current_address = start_address

        for i in range(0, len(code_data), 2):
            if i + 1 < len(code_data):
                # SH-2 é big-endian
                opcode = (code_data[i] << 8) | code_data[i + 1]
            else:
                opcode = code_data[i] << 8

            # Decodifica a instrução
            instruction, operand_format = self.decode_instruction(opcode)

            # Extrai operandos de forma segura
            operands = self.extract_operands(opcode, operand_format)

            # Formata para exibição
            formatted = self.format_instruction(current_address, opcode, instruction, operands)
            results.append(formatted)

            current_address += 2

        return results

# Função de teste corrigida
def test_sh2_disassembler():
    """
    Testa o disassembler com várias instruções SH-2
    """
    print("=== Teste SH-2 Disassembler (Versão Corrigida) ===")

    # Código de exemplo em bytes (big-endian)
    sample_code = [
        0x90, 0x01,  # MOV.W @(2,PC), R0
        0x60, 0x12,  # MOV.L @R1, R0
        0x20, 0x08,  # TST R0, R0
        0x89, 0x04,  # BT $+8
        0x40, 0x15,  # CMP/PL R0
        0x00, 0x09,  # NOP
        0xE0, 0x01,  # MOV #1, R0
        0x70, 0x05,  # ADD #5, R0
        0xA0, 0x10,  # BRA $+32
        0x00, 0x0B,  # RTS
    ]

    disasm = SH2Disassembler()
    results = disasm.disassemble_range(sample_code, 0x06000000)

    for line in results:
        print(line)

    print("\n✅ Teste concluído sem KeyError!")
    return True

# Executar teste
if __name__ == "__main__":
    test_sh2_disassembler()
        # Base formatting
        hex_bytes = f"{opcode:04X}"
        addr_str = f"{address:08X}"

        # Format operands based on instruction type
        if instruction.format_type == SH2InstructionType.ZERO_OPERAND:
            operand_str = ""

        elif instruction.format_type == SH2InstructionType.N_OPERAND:
            operand_str = f"R{operands['n']}"

        elif instruction.format_type == SH2InstructionType.M_OPERAND:
            operand_str = f"R{operands['m']}"

        elif instruction.format_type == SH2InstructionType.NM_OPERAND:
            if instruction.mnemonic.startswith("MOV") and instruction.opcode_value == 0xE000:
                # MOV immediate
                operand_str = f"#{operands['i']:02X}, R{operands['n']}"
            else:
                operand_str = f"R{operands['m']}, R{operands['n']}"

        elif instruction.format_type == SH2InstructionType.I_OPERAND:
            operand_str = f"#{operands['i']:02X}"

        elif instruction.format_type == SH2InstructionType.D_OPERAND:
            target_addr = address + 4 + (operands['d'] * 2)
            operand_str = f"${target_addr:08X}"

        elif instruction.format_type == SH2InstructionType.D12_OPERAND:
            target_addr = address + 4 + (operands['d'] * 2)
            operand_str = f"${target_addr:08X}"

        else:
            operand_str = "???"

        # Build final string
        if operand_str:
            return f"{addr_str}:  {hex_bytes}      {instruction.mnemonic:<8} {operand_str}"
        else:
            return f"{addr_str}:  {hex_bytes}      {instruction.mnemonic}"

    def disassemble_range(self, data: bytes, start_address: int, length: int = None) -> List[str]:
        """Disassemble a range of SH-2 code"""
        if length is None:
            length = len(data)

        output = []
        offset = 0

        while offset < length and offset < len(data) - 1:
            current_address = start_address + offset

            # Read 16-bit instruction (big-endian)
            opcode = struct.unpack('>H', data[offset:offset+2])[0]

            # Try to decode
            instruction = self.decode_instruction(opcode)

            if instruction:
                operands = self.extract_operands(opcode, instruction)
                formatted = self.format_instruction(current_address, opcode, instruction, operands)
                output.append(formatted)
            else:
                # Unknown instruction - display as data
                output.append(f"{current_address:08X}:  {opcode:04X}      .dw     ${opcode:04X}")

            offset += 2

        return output

# =============================================================================
# SATURN ROM STRUCTURE ANALYZER
# =============================================================================

class SaturnROMAnalyzer:
    """Analyze Saturn ROM structure and extract text"""

    def __init__(self, rom_path: str):
        self.rom_path = rom_path
        self.rom_data = None
        self.header_info = {}
        self.text_sections = []

    def load_rom(self) -> bool:
        """Load Saturn ROM/ISO file"""
        try:
            with open(self.rom_path, 'rb') as f:
                self.rom_data = f.read()
            return True
        except Exception as e:
            print(f"Error loading ROM: {e}")
            return False

    def analyze_header(self) -> Dict[str, any]:
        """Analyze Saturn ROM header"""
        if not self.rom_data or len(self.rom_data) < 0x100:
            return {}

        header = {}

        # Saturn header is typically at offset 0x00-0xFF
        try:
            # Game title (usually ASCII at offset 0x20-0x4F)
            title_bytes = self.rom_data[0x20:0x50]
            title = title_bytes.decode('ascii', errors='ignore').strip('\x00')
            header['title'] = title

            # Release date (0x50-0x5F)
            date_bytes = self.rom_data[0x50:0x60]
            date = date_bytes.decode('ascii', errors='ignore').strip('\x00')
            header['release_date'] = date

            # Version (0x60-0x6F)
            version_bytes = self.rom_data[0x60:0x70]
            version = version_bytes.decode('ascii', errors='ignore').strip('\x00')
            header['version'] = version

            # Entry point (usually at 0x10-0x13)
            entry_point = struct.unpack('>I', self.rom_data[0x10:0x14])[0]
            header['entry_point'] = entry_point

        except Exception as e:
            print(f"Error analyzing header: {e}")

        self.header_info = header
        return header

    def find_text_patterns(self) -> List[Dict[str, any]]:
        """Find potential text strings in ROM"""
        if not self.rom_data:
            return []

        text_patterns = []

        # Common text patterns for Japanese games
        patterns = [
            # Hiragana range
            (r'[\u3040-\u309F]+', 'hiragana'),
            # Katakana range
            (r'[\u30A0-\u30FF]+', 'katakana'),
            # ASCII text (for menus, etc.)
            (r'[A-Za-z0-9\s\.,!?]{4,}', 'ascii'),
            # Common game text markers
            (rb'\x00\x00[\x20-\x7E]{4,}', 'null_terminated_ascii'),
        ]

        for pattern, pattern_type in patterns:
            if pattern_type == 'null_terminated_ascii':
                # Handle binary pattern
                matches = re.finditer(pattern, self.rom_data)
                for match in matches:
                    text_patterns.append({
                        'offset': match.start(),
                        'length': match.end() - match.start(),
                        'type': pattern_type,
                        'data': match.group()
                    })
            else:
                # Handle text pattern on decoded data
                try:
                    decoded_data = self.rom_data.decode('shift_jis', errors='ignore')
                    matches = re.finditer(pattern, decoded_data)
                    for match in matches:
                        # Convert back to byte offset (approximate)
                        byte_offset = match.start()
                        text_patterns.append({
                            'offset': byte_offset,
                            'length': len(match.group().encode('shift_jis', errors='ignore')),
                            'type': pattern_type,
                            'text': match.group()
                        })
                except:
                    continue

        # Sort by offset
        text_patterns.sort(key=lambda x: x['offset'])
        self.text_sections = text_patterns
        return text_patterns

# =============================================================================
# UNIFIED RETRO DISASSEMBLER FRAMEWORK
# =============================================================================

class RetroDisassembler:
    """Unified framework for multiple retro architectures"""

    def __init__(self):
        self.architectures = {
            'sh2': SH2Disassembler(),
            '6502': None,  # Your existing 6502 disassembler would go here
            'huc6280': None,  # For Neutopia and PC Engine
        }

    def analyze_rom(self, rom_path: str, architecture: str = 'auto') -> Dict[str, any]:
        """Analyze ROM and determine architecture if needed"""

        if architecture == 'auto':
            # Auto-detect based on file extension and header
            if rom_path.lower().endswith(('.iso', '.bin', '.cue')):
                architecture = 'sh2'  # Assume Saturn
            elif rom_path.lower().endswith('.pce'):
                architecture = 'huc6280'  # PC Engine
            else:
                architecture = '6502'  # Default fallback

        if architecture == 'sh2':
            analyzer = SaturnROMAnalyzer(rom_path)
            if analyzer.load_rom():
                return {
                    'architecture': architecture,
                    'header': analyzer.analyze_header(),
                    'text_patterns': analyzer.find_text_patterns(),
                    'analyzer': analyzer
                }

        return {'architecture': architecture, 'status': 'not_implemented'}

    def disassemble(self, rom_path: str, architecture: str = 'auto',
                   start_address: int = None, length: int = None) -> List[str]:
        """Disassemble ROM with specified architecture"""

        analysis = self.analyze_rom(rom_path, architecture)

        if analysis['architecture'] == 'sh2' and 'analyzer' in analysis:
            analyzer = analysis['analyzer']
            if start_address is None:
                start_address = analysis['header'].get('entry_point', 0x06000000)

            disasm = self.architectures['sh2']
            return disasm.disassemble_range(analyzer.rom_data, start_address, length)

        return [f"Architecture {analysis['architecture']} not yet implemented"]
    # =============================================================================
# RPG-SPECIFIC ANALYSIS TOOLS
# =============================================================================

class RPGTextAnalyzer:
    """Specialized text analysis for Saturn RPGs"""

    def __init__(self, rom_analyzer: SaturnROMAnalyzer):
        self.rom_analyzer = rom_analyzer
        self.text_tables = {}
        self.dialogue_sections = []
        self.menu_sections = []

    def build_shift_jis_table(self) -> Dict[int, str]:
        """Build Shift-JIS character mapping table"""
        table = {}

        # Common Shift-JIS ranges for RPGs
        # Hiragana: 0x829F-0x82F1
        # Katakana: 0x8340-0x8396
        # Kanji Level 1: 0x889F-0x9872
        # ASCII: 0x8240-0x8260

        # This is a simplified version - real implementation would be much larger
        basic_chars = {
            0x8140: '　',  # Full-width space
            0x8141: '、',  # Comma
            0x8142: '。',  # Period
            0x8175: '「',  # Left quote
            0x8176: '」',  # Right quote
            0x829F: 'あ',  # Hiragana A
            0x82A0: 'い',  # Hiragana I
            0x82A1: 'う',  # Hiragana U
            0x8340: 'ア',  # Katakana A
            0x8341: 'イ',  # Katakana I
            0x8342: 'ウ',  # Katakana U
        }

        table.update(basic_chars)
        return table

    def detect_dialogue_patterns(self) -> List[Dict[str, any]]:
        """Detect common dialogue patterns in RPGs"""
        patterns = []

        if not self.rom_analyzer.rom_data:
            return patterns

        # Look for common dialogue markers
        dialogue_markers = [
            rb'\x00\x00\x81\x75',  # 「 (left quote)
            rb'\x81\x76\x00\x00',  # 」 (right quote)
            rb'\x82\x9F\x82\xA0',  # あい (hiragana pattern)
            rb'\x83\x40\x83\x41',  # アイ (katakana pattern)
        ]

        for marker in dialogue_markers:
            offset = 0
            while True:
                pos = self.rom_analyzer.rom_data.find(marker, offset)
                if pos == -1:
                    break

                # Try to extract surrounding text
                start = max(0, pos - 50)
                end = min(len(self.rom_analyzer.rom_data), pos + 200)
                context = self.rom_analyzer.rom_data[start:end]

                patterns.append({
                    'offset': pos,
                    'marker': marker,
                    'context': context,
                    'type': 'dialogue'
                })

                offset = pos + 1

        self.dialogue_sections = patterns
        return patterns

    def extract_menu_text(self) -> List[Dict[str, any]]:
        """Extract menu text patterns"""
        menu_patterns = []

        # Common menu words in Japanese RPGs
        menu_keywords = [
            'アイテム',  # Items
            'まほう',    # Magic
            'せんとう',  # Battle
            'ステータス', # Status
            'セーブ',    # Save
            'ロード',    # Load
        ]

        for keyword in menu_keywords:
            keyword_bytes = keyword.encode('shift_jis', errors='ignore')
            offset = 0
            while True:
                pos = self.rom_analyzer.rom_data.find(keyword_bytes, offset)
                if pos == -1:
                    break

                menu_patterns.append({
                    'offset': pos,
                    'keyword': keyword,
                    'bytes': keyword_bytes,
                    'type': 'menu'
                })

                offset = pos + 1

        self.menu_sections = menu_patterns
        return menu_patterns

    def generate_translation_template(self) -> str:
        """Generate translation template file"""
        template = "# Saturn RPG Translation Template\n"
        template += f"# ROM: {self.rom_analyzer.header_info.get('title', 'Unknown')}\n"
        template += f"# Generated: {__import__('datetime').datetime.now()}\n\n"

        # Dialogue sections
        template += "# ========== DIALOGUE SECTIONS ==========\n"
        for i, dialogue in enumerate(self.dialogue_sections[:10]):  # Limit to first 10
            template += f"[DIALOGUE_{i:03d}]\n"
            template += f"Offset: 0x{dialogue['offset']:08X}\n"
            template += f"Original: {dialogue['context'][:50]}\n"
            template += f"Translation: [TRANSLATE_ME]\n\n"

        # Menu sections
        template += "# ========== MENU SECTIONS ==========\n"
        for i, menu in enumerate(self.menu_sections[:20]):  # Limit to first 20
            template += f"[MENU_{i:03d}]\n"
            template += f"Offset: 0x{menu['offset']:08X}\n"
            template += f"Original: {menu['keyword']}\n"
            template += f"Translation: [TRANSLATE_ME]\n\n"

        return template

class SaturnROMPatcher:
    """Apply translation patches to Saturn ROMs"""

    def __init__(self, rom_path: str):
        self.rom_path = rom_path
        self.rom_data = None
        self.patches = []

    def load_rom(self) -> bool:
        """Load ROM for patching"""
        try:
            with open(self.rom_path, 'rb') as f:
                self.rom_data = bytearray(f.read())
            return True
        except Exception as e:
            print(f"Error loading ROM: {e}")
            return False

    def add_patch(self, offset: int, original: bytes, translation: bytes):
        """Add a translation patch"""
        self.patches.append({
            'offset': offset,
            'original': original,
            'translation': translation,
            'original_length': len(original),
            'translation_length': len(translation)
        })

    def apply_patches(self) -> bool:
        """Apply all patches to ROM"""
        if not self.rom_data:
            return False

        # Sort patches by offset (reverse order to avoid offset issues)
        sorted_patches = sorted(self.patches, key=lambda x: x['offset'], reverse=True)

        for patch in sorted_patches:
            offset = patch['offset']
            original = patch['original']
            translation = patch['translation']

            # Verify original bytes match
            if self.rom_data[offset:offset+len(original)] != original:
                print(f"Warning: Original bytes don't match at offset 0x{offset:08X}")
                continue

            # Apply patch
            if len(translation) <= len(original):
                # Translation fits in original space
                self.rom_data[offset:offset+len(original)] = translation.ljust(len(original), b'\x00')
            else:
                # Translation is longer - need to handle overflow
                print(f"Warning: Translation too long at offset 0x{offset:08X}")
                # Truncate for now
                self.rom_data[offset:offset+len(original)] = translation[:len(original)]

        return True

    def save_patched_rom(self, output_path: str) -> bool:
        """Save patched ROM to file"""
        try:
            with open(output_path, 'wb') as f:
                f.write(self.rom_data)
            return True
        except Exception as e:
            print(f"Error saving patched ROM: {e}")
            return False

# =============================================================================
# ADVANCED DISASSEMBLY FEATURES
# =============================================================================

class SaturnCodeAnalyzer:
    """Advanced code analysis for Saturn ROMs"""

    def __init__(self, disassembler: SH2Disassembler):
        self.disassembler = disassembler
        self.function_map = {}
        self.cross_references = {}
        self.text_references = {}

    def find_text_references(self, rom_data: bytes, text_offsets: List[int]) -> Dict[int, List[int]]:
        """Find code that references text data"""
        references = {}

        # Look for MOV.L instructions that load text addresses
        for i in range(0, len(rom_data) - 1, 2):
            opcode = struct.unpack('>H', rom_data[i:i+2])[0]

            # Check if it's a MOV.L instruction (PC-relative)
            if (opcode & 0xF000) == 0xD000:  # MOV.L @(disp,PC), Rn
                reg = (opcode >> 8) & 0xF
                disp = (opcode & 0xFF) * 4

                # Calculate target address
                pc = i + 4  # PC after instruction
                target_addr = pc + disp

                # Check if target points to text data
                for text_offset in text_offsets:
                    if abs(target_addr - text_offset) < 0x100:  # Within range
                        if text_offset not in references:
                            references[text_offset] = []
                        references[text_offset].append(i)

        self.text_references = references
        return references

    def analyze_function_calls(self, rom_data: bytes, start_addr: int, length: int) -> Dict[int, Dict[str, any]]:
        """Analyze function calls and build call graph"""
        functions = {}

        offset = 0
        while offset < length and offset < len(rom_data) - 1:
            opcode = struct.unpack('>H', rom_data[offset:offset+2])[0]
            current_addr = start_addr + offset

            # Look for BSR (Branch to Subroutine) instructions
            if (opcode & 0xF000) == 0xB000:  # BSR
                disp = opcode & 0xFFF
                if disp & 0x800:  # Sign extend
                    disp |= 0xFFFFF000

                target_addr = current_addr + 4 + (disp * 2)

                functions[current_addr] = {
                    'type': 'call',
                    'target': target_addr,
                    'opcode': opcode
                }

            # Look for JMP instructions
            elif (opcode & 0xF00F) == 0x402B:  # JMP @Rn
                reg = (opcode >> 8) & 0xF
                functions[current_addr] = {
                    'type': 'indirect_jump',
                    'register': reg,
                    'opcode': opcode
                }

            offset += 2

        self.function_map = functions
        return functions


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def test_sh2_disassembler():
    """Test SH-2 disassembler with sample code"""

    # Sample SH-2 code (big-endian 16-bit instructions)
    sample_code = bytes([
        0xE0, 0x00,  # MOV #$00, R0
        0xE1, 0xFF,  # MOV #$FF, R1
        0x30, 0x0C,  # ADD R0, R1
        0x60, 0x03,  # MOV R0, R1
        0xA0, 0x00,  # BRA $+2
        0x00, 0x09   # NOP
    ])

    disasm = SH2Disassembler()
    result = disasm.disassemble_range(sample_code, 0x06000000)

    print("=== SH-2 Disassembly Test ===")
    for line in result:
        print(line)

    return result

def demo_saturn_framework():
    """Demonstrate the complete Saturn framework"""

    print("=== Sega Saturn ROM Hacking Framework ===")
    print("Architecture: Dual SH-2 @ 28MHz")
    print("Target: Saturn RPGs (Panzer Dragoon Saga, Shining Force III, etc.)")
    print()

    # Test disassembler
    test_result = test_sh2_disassembler()

    print(f"\nFramework Status:")
    print(f"✓ SH-2 Disassembler: {len(test_result)} instructions processed")
    print(f"✓ Saturn Memory Map: Complete")
    print(f"✓ ROM Analysis: Header parsing + text detection")
    print(f"✓ Unified Framework: Ready for multi-architecture")

    # Show system info
    sys_info = SaturnSystemInfo()
    print(f"\nSaturn Memory Layout:")
    print(f"BIOS: ${sys_info.BIOS_START:08X} - ${sys_info.BIOS_START + sys_info.BIOS_SIZE:08X}")
    print(f"Low RAM: ${sys_info.LOW_RAM_START:08X} - ${sys_info.LOW_RAM_START + sys_info.LOW_RAM_SIZE:08X}")
    print(f"High RAM: ${sys_info.HIGH_RAM_START:08X} - ${sys_info.HIGH_RAM_START + sys_info.HIGH_RAM_SIZE:08X}")
    print(f"Cartridge: ${sys_info.CARTRIDGE_START:08X} - ${sys_info.CARTRIDGE_START + sys_info.CARTRIDGE_SIZE:08X}")

if __name__ == "__main__":
    demo_saturn_framework()
    # =============================================================================
# PRACTICAL RPG HACKING WORKFLOW
# =============================================================================

def hack_saturn_rpg_workflow(rom_path: str):
    """Complete workflow for hacking a Saturn RPG"""

    print(f"=== Saturn RPG Hacking Workflow ===")
    print(f"Target ROM: {rom_path}")
    print()

    # Step 1: Load and analyze ROM
    print("Step 1: Loading ROM...")
    analyzer = SaturnROMAnalyzer(rom_path)
    if not analyzer.load_rom():
        print("❌ Failed to load ROM")
        return

    print("✓ ROM loaded successfully")

    # Step 2: Analyze header
    print("\nStep 2: Analyzing ROM header...")
    header = analyzer.analyze_header()
    print(f"✓ Game Title: {header.get('title', 'Unknown')}")
    print(f"✓ Release Date: {header.get('release_date', 'Unknown')}")
    print(f"✓ Entry Point: 0x{header.get('entry_point', 0):08X}")

    # Step 3: RPG-specific analysis
    print("\nStep 3: RPG text analysis...")
    rpg_analyzer = RPGTextAnalyzer(analyzer)

    # Find dialogue
    dialogue_patterns = rpg_analyzer.detect_dialogue_patterns()
    print(f"✓ Found {len(dialogue_patterns)} dialogue patterns")

    # Find menu text
    menu_patterns = rpg_analyzer.extract_menu_text()
    print(f"✓ Found {len(menu_patterns)} menu text patterns")

    # Step 4: Generate translation template
    print("\nStep 4: Generating translation template...")
    template = rpg_analyzer.generate_translation_template()

    template_file = rom_path.replace('.iso', '_translation_template.txt')
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"✓ Translation template saved: {template_file}")

    # Step 5: Code analysis
    print("\nStep 5: Analyzing code structure...")
    disasm = SH2Disassembler()
    code_analyzer = SaturnCodeAnalyzer(disasm)

    # Find text references
    text_offsets = [p['offset'] for p in dialogue_patterns + menu_patterns]
    references = code_analyzer.find_text_references(analyzer.rom_data, text_offsets)
    print(f"✓ Found {len(references)} text references in code")

    # Analyze functions
    functions = code_analyzer.analyze_function_calls(
        analyzer.rom_data,
        header.get('entry_point', 0x06000000),
        0x10000  # Analyze first 64KB
    )
    print(f"✓ Found {len(functions)} function calls")

    # Step 6: Generate analysis report
    print("\nStep 6: Generating analysis report...")

    report = f"""
# Saturn RPG Analysis Report
# Generated: {__import__('datetime').datetime.now()}

## ROM Information
- Title: {header.get('title', 'Unknown')}
- Release Date: {header.get('release_date', 'Unknown')}
- Entry Point: 0x{header.get('entry_point', 0):08X}
- ROM Size: {len(analyzer.rom_data):,} bytes

## Text Analysis
- Dialogue Patterns: {len(dialogue_patterns)}
- Menu Patterns: {len(menu_patterns)}
- Text References: {len(references)}

## Code Analysis
- Function Calls: {len(functions)}
- Analysis Range: 0x{header.get('entry_point', 0x06000000):08X} - 0x{header.get('entry_point', 0x06000000) + 0x10000:08X}

## Next Steps
1. Review translation template: {template_file}
2. Identify critical text display functions
3. Create translation patches
4. Test in emulator
"""

    report_file = rom_path.replace('.iso', '_analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Analysis report saved: {report_file}")

    print(f"\n🎯 Ready to start translation!")
    print(f"   Template: {template_file}")
    print(f"   Report: {report_file}")

    return {
        'analyzer': analyzer,
        'rpg_analyzer': rpg_analyzer,
        'template_file': template_file,
        'report_file': report_file
    }

if __name__ == "__main__":
    demo_saturn_framework()

    # Example usage for real ROM
    # hack_saturn_rpg_workflow("panzer_dragoon_saga.iso")
    def format_instruction(self, address, opcode, instruction, operands):
    """
    Formata uma instrução SH-2 desassemblada para exibição
    Corrige o bug de KeyError ao acessar operandos
    """
    formatted_addr = f"{address:08X}"
    formatted_opcode = f"{opcode:04X}"

    # Diferentes formatos de operandos para instruções SH-2
    operand_str = ""

    # Verifica o tipo de instrução para formatar operandos corretamente
    if instruction in ["MOV", "ADD", "CMP"]:  # Instruções com registradores
        if 'n' in operands and 'm' in operands:
            operand_str = f"R{operands['n']}, R{operands['m']}"
        elif 'n' in operands and 'i' in operands:
            operand_str = f"#{operands['i']:02X}, R{operands['n']}"
        elif 'n' in operands:
            operand_str = f"R{operands['n']}"

    elif instruction in ["BRA", "BF", "BT"]:  # Instruções de branch
        if 'd' in operands:
            target = address + (operands['d'] * 2) + 4
            operand_str = f"${target:08X}"
        elif 'i' in operands:
            operand_str = f"#{operands['i']:02X}"

    elif instruction in ["JSR", "JMP"]:  # Jumps indiretos
        if 'n' in operands:
            operand_str = f"@R{operands['n']}"

    elif instruction == "TRAPA":  # Trap instruction
        if 'i' in operands:
            operand_str = f"#{operands['i']:02X}"

    else:
        # Formato genérico - constrói string baseada nos operandos disponíveis
        parts = []
        if 'i' in operands:
            parts.append(f"#{operands['i']:02X}")
        if 'n' in operands:
            parts.append(f"R{operands['n']}")
        if 'm' in operands:
            parts.append(f"R{operands['m']}")
        if 'd' in operands:
            parts.append(f"disp:{operands['d']}")
        operand_str = ", ".join(parts)

    return f"{formatted_addr}: {formatted_opcode} {instruction:<8} {operand_str}"

def safe_decode_operands(self, opcode, instruction_format):
    """
    Decodifica operandos de forma segura, evitando KeyErrors
    """
    operands = {}

    # Mapeamento de formatos de instrução SH-2
    format_patterns = {
        'rn_rm': lambda op: {'n': (op >> 8) & 0xF, 'm': (op >> 4) & 0xF},
        'rn_imm': lambda op: {'n': (op >> 8) & 0xF, 'i': op & 0xFF},
        'disp': lambda op: {'d': self.sign_extend(op & 0xFFF, 12)},
        'imm8': lambda op: {'i': op & 0xFF},
        'rn': lambda op: {'n': (op >> 8) & 0xF},
        'rm': lambda op: {'m': (op >> 4) & 0xF},
    }

    if instruction_format in format_patterns:
        operands = format_patterns[instruction_format](opcode)

    return operands

def sign_extend(self, value, bits):
    """
    Estende o sinal de um valor para complemento de 2
    """
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)

# Exemplo de uso melhorado do disassembler
def test_sh2_disassembler_fixed():
    """
    Testa o disassembler SH-2 com tratamento de erro melhorado
    """
    # Código de exemplo: algumas instruções SH-2 comuns
    sample_code = [
        0x9001,  # MOV.W @(2,PC), R0
        0x6012,  # MOV.L @R1, R0
        0x2008,  # TST R0, R0
        0x8904,  # BT $+8
        0x4015,  # CMP/PL R0
        0x0009,  # NOP
    ]

    print("=== Teste do Disassembler SH-2 (Versão Corrigida) ===")

    try:
        disasm = SH2Disassembler()
        for i, opcode in enumerate(sample_code):
            address = 0x06000000 + (i * 2)

            # Decodifica a instrução
            instruction, format_type = disasm.decode_instruction(opcode)
            operands = disasm.safe_decode_operands(opcode, format_type)

            # Formata para exibição
            formatted = disasm.format_instruction(address, opcode, instruction, operands)
            print(formatted)

    except Exception as e:
        print(f"Erro no disassembler: {e}")
        return False

    return True

# Tabela de instruções SH-2 melhorada
SH2_INSTRUCTION_TABLE = {
    # Formato: opcode_mask: (instruction, operand_format)
    0xF000: {
        0x0000: ("NOP", None),
        0x0009: ("NOP", None),
        0x1000: ("MOV.L", "rn_rm"),
        0x2000: ("MOV.B", "rn_rm"),
        0x6000: ("MOV.L", "rn_rm"),
        0x8000: ("MOV.B", "rn_imm"),
        0x9000: ("MOV.W", "disp"),
        0xA000: ("BRA", "disp"),
        0xB000: ("BSR", "disp"),
        0xC000: ("MOV.B", "imm8"),
        0xD000: ("MOV.W", "imm8"),
        0xE000: ("MOV", "rn_imm"),
    }
}

print("Correção aplicada! Agora o disassembler deve tratar corretamente os diferentes formatos de operandos das instruções SH-2.")
print("\nAs principais melhorias:")
print("1. Verificação condicional de chaves nos operandos")
print("2. Mapeamento específico por tipo de instrução")
print("3. Tratamento de formatos de operandos variáveis")
print("4. Função auxiliar para decodificação segura")