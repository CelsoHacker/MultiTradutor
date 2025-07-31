def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Platform ROM Translation Engine")
    parser.add_argument("#!/usr/bin/env python3
"""
PS2 ROM Translation Engine
Automated translation system for PlayStation 2 ROMs using AI and advanced hacking techniques.
"""

import os
import re
import struct
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import zipfile
import json

# Third-party imports (install with: pip install requests beautifulsoup4 openai)
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests/beautifulsoup4 not installed. Web features disabled.")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. AI translation features disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextEncoding(Enum):
    """PS2 text encoding types"""
    ASCII = "ascii"
    SHIFT_JIS = "shift_jis"
    UTF8 = "utf-8"
    UTF16 = "utf-16"
    CUSTOM = "custom"

class ROMType(Enum):
    """ROM file types for multiple platforms"""
    ISO = "iso"
    BIN = "bin"
    ELF = "elf"
    IRX = "irx"
    WAD = "wad"      # Doom/Hexen WAD files
    ROM = "rom"      # N64 ROM files
    V64 = "v64"      # N64 V64 format
    Z64 = "z64"      # N64 Z64 format
    UNKNOWN = "unknown"

@dataclass
class TextEntry:
    """Represents a translatable text entry"""
    offset: int
    original_text: str
    translated_text: str = ""
    encoding: TextEncoding = TextEncoding.ASCII
    context: str = ""
    length: int = 0

    def __post_init__(self):
        if self.length == 0:
            self.length = len(self.original_text)

@dataclass
class ROMInfo:
    """ROM metadata and analysis info"""
    filename: str
    file_size: int
    md5_hash: str
    rom_type: ROMType
    text_encoding: TextEncoding
    game_id: str = ""
    region: str = ""

class MultiPlatformROMTranslator:
    """Universal ROM Translation Engine - PS2, N64, PC Games"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the translator with optional OpenAI API key"""
        self.openai_api_key = openai_api_key
        self.text_patterns = self._load_text_patterns()
        self.supported_encodings = [
            TextEncoding.ASCII,
            TextEncoding.SHIFT_JIS,
            TextEncoding.UTF8,
            TextEncoding.UTF16
        ]

        # Initialize OpenAI if available
        if HAS_OPENAI and openai_api_key:
            openai.api_key = openai_api_key

    def _load_text_patterns(self) -> Dict[str, List[bytes]]:
        """Load common text patterns for different PS2 games"""
        return {
            "dialogue": [
                b'\x00\x00\x00\x00',  # NULL terminated strings
                b'\xFF\xFF\xFF\xFF',  # Common dialogue marker
                b'\x20\x20\x20\x20',  # Space padding
            ],
            "menu": [
                b'\x00\x01\x00\x01',  # Menu item separators
                b'\x80\x00\x80\x00',  # UTF-16 markers
            ],
            "system": [
                b'SYSTEM',
                b'ERROR',
                b'SAVE',
                b'LOAD'
            ]
        }

    def analyze_rom(self, rom_path: str) -> ROMInfo:
        """Analyze ROM file and extract metadata"""
        logger.info(f"Analyzing ROM: {rom_path}")

        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")

        # Calculate file hash
        with open(rom_path, 'rb') as f:
            file_data = f.read()
            md5_hash = hashlib.md5(file_data).hexdigest()

        # Determine ROM type
        rom_type = self._detect_rom_type(rom_path, file_data)

        # Detect text encoding
        text_encoding = self._detect_text_encoding(file_data)

        # Extract game information
        game_id = self._extract_game_id(file_data)
        region = self._detect_region(file_data)

        rom_info = ROMInfo(
            filename=os.path.basename(rom_path),
            file_size=len(file_data),
            md5_hash=md5_hash,
            rom_type=rom_type,
            text_encoding=text_encoding,
            game_id=game_id,
            region=region
        )

        logger.info(f"ROM Analysis Complete: {rom_info}")
        return rom_info

    def _detect_rom_type(self, rom_path: str, file_data: bytes) -> ROMType:
        """Detect the type of ROM file - Multi-platform support"""
        extension = Path(rom_path).suffix.lower()

        # Extension-based detection
        if extension == '.iso':
            return ROMType.ISO
        elif extension == '.bin':
            return ROMType.BIN
        elif extension == '.elf':
            return ROMType.ELF
        elif extension == '.irx':
            return ROMType.IRX
        elif extension == '.wad':
            return ROMType.WAD
        elif extension in ['.rom', '.n64']:
            return ROMType.ROM
        elif extension == '.v64':
            return ROMType.V64
        elif extension == '.z64':
            return ROMType.Z64

        # Signature-based detection
        if file_data[:4] == b'\x7fELF':
            return ROMType.ELF
        elif b'PlayStation' in file_data[:0x1000]:
            return ROMType.ISO
        elif file_data[:4] == b'IWAD' or file_data[:4] == b'PWAD':
            return ROMType.WAD  # Doom/Hexen WAD files
        elif len(file_data) > 64 and self._is_n64_rom(file_data):
            return ROMType.ROM
        elif file_data[:4] == b'\x40\x12\x37\x80':  # N64 Z64 format
            return ROMType.Z64
        elif file_data[:4] == b'\x37\x80\x40\x12':  # N64 V64 format
            return ROMType.V64

        return ROMType.UNKNOWN

    def _is_n64_rom(self, file_data: bytes) -> bool:
        """Check if file is a Nintendo 64 ROM"""
        if len(file_data) < 0x1000:
            return False

        # Check common N64 ROM signatures
        n64_signatures = [
            b'\x80\x37\x12\x40',  # Big-endian
            b'\x40\x12\x37\x80',  # Little-endian
            b'\x37\x80\x40\x12',  # Byte-swapped
        ]

        for sig in n64_signatures:
            if file_data[:4] == sig:
                return True

        # Check for N64 game titles in header
        header = file_data[0x20:0x34]
        try:
            title = header.decode('ascii', errors='ignore').strip()
            # Quest 64 specific check
            if 'QUEST64' in title.upper() or 'HOLY MAGIC' in title.upper():
                return True
        except:
            pass

        return False

    def _detect_text_encoding(self, file_data: bytes) -> TextEncoding:
        """Detect text encoding used in the ROM"""
        # Sample first 64KB for encoding detection
        sample = file_data[:65536]

        # Check for UTF-16 BOM
        if sample.startswith(b'\xFF\xFE') or sample.startswith(b'\xFE\xFF'):
            return TextEncoding.UTF16

        # Check for UTF-8 BOM
        if sample.startswith(b'\xEF\xBB\xBF'):
            return TextEncoding.UTF8

        # Heuristic detection for Shift-JIS (common in Japanese PS2 games)
        sjis_indicators = 0
        for i in range(0, len(sample) - 1, 2):
            byte1, byte2 = sample[i], sample[i+1]
            if (0x81 <= byte1 <= 0x9F) or (0xE0 <= byte1 <= 0xEF):
                if (0x40 <= byte2 <= 0x7E) or (0x80 <= byte2 <= 0xFC):
                    sjis_indicators += 1

        if sjis_indicators > 100:  # Threshold for Shift-JIS detection
            return TextEncoding.SHIFT_JIS

        return TextEncoding.ASCII

    def _extract_game_id(self, file_data: bytes) -> str:
        """Extract game ID from ROM data"""
        # Look for PS2 game ID pattern (e.g., SLUS-12345)
        pattern = rb'[A-Z]{4}-\d{5}'
        match = re.search(pattern, file_data[:0x10000])

        if match:
            return match.group(0).decode('ascii')

        return "UNKNOWN"

    def _detect_region(self, file_data: bytes) -> str:
        """Detect game region from ROM data"""
        region_markers = {
            b'NTSC-U': 'US',
            b'NTSC-J': 'Japan',
            b'PAL': 'Europe',
            b'SLUS': 'US',
            b'SLPS': 'Japan',
            b'SLES': 'Europe',
            b'SCUS': 'US',
            b'SCPS': 'Japan',
            b'SCES': 'Europe'
        }

        sample = file_data[:0x10000]
        for marker, region in region_markers.items():
            if marker in sample:
                return region

        return "Unknown"

    def extract_text(self, rom_path: str, rom_info: ROMInfo) -> List[TextEntry]:
        """Extract translatable text from ROM"""
        logger.info(f"Extracting text from {rom_path}")

        with open(rom_path, 'rb') as f:
            file_data = f.read()

        text_entries = []

        # Extract based on ROM type
        if rom_info.rom_type == ROMType.ISO:
            text_entries = self._extract_from_iso(file_data, rom_info)
        elif rom_info.rom_type == ROMType.ELF:
            text_entries = self._extract_from_elf(file_data, rom_info)
        elif rom_info.rom_type == ROMType.WAD:
            text_entries = self._extract_from_wad(file_data, rom_info)
        elif rom_info.rom_type in [ROMType.ROM, ROMType.V64, ROMType.Z64]:
            text_entries = self._extract_from_n64(file_data, rom_info)
    def _extract_from_n64(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Extract text from Nintendo 64 ROMs - Hexen & Quest 64 optimized"""
        text_entries = []

        # Normalize N64 ROM to big-endian format
        normalized_data = self._normalize_n64_rom(file_data)

        # Get game-specific extraction strategy
        game_id = self._get_n64_game_id(normalized_data)

        if "HEXEN" in game_id.upper():
            text_entries = self._extract_hexen_n64(normalized_data, rom_info)
        elif "QUEST64" in game_id.upper() or "HOLY MAGIC" in game_id.upper():
            text_entries = self._extract_quest64(normalized_data, rom_info)
        else:
            # Generic N64 extraction
            text_entries = self._extract_generic_n64(normalized_data, rom_info)

        logger.info(f"N64 extraction complete: {len(text_entries)} entries found")
        return text_entries

    def _normalize_n64_rom(self, file_data: bytes) -> bytes:
        """Normalize N64 ROM to big-endian format"""
        if len(file_data) < 4:
            return file_data

        # Check current endianness
        header = file_data[:4]

        if header == b'\x80\x37\x12\x40':
            # Already big-endian, no conversion needed
            return file_data
        elif header == b'\x40\x12\x37\x80':
            # Little-endian, swap bytes
            return bytes(struct.pack('>I', struct.unpack('<I', file_data[i:i+4])[0])
                        for i in range(0, len(file_data), 4))
        elif header == b'\x37\x80\x40\x12':
            # Byte-swapped, fix order
            return bytes(file_data[i+1:i+2] + file_data[i:i+1] + file_data[i+3:i+4] + file_data[i+2:i+3]
                        for i in range(0, len(file_data), 4))

        return file_data

    def _get_n64_game_id(self, normalized_data: bytes) -> str:
        """Extract game ID from N64 ROM header"""
        if len(normalized_data) < 0x40:
            return "UNKNOWN"

        # Game title is at offset 0x20-0x33
        title_bytes = normalized_data[0x20:0x34]
        try:
            title = title_bytes.decode('ascii', errors='ignore').strip('\x00 ')
            return title
        except:
            return "UNKNOWN"

    def _extract_hexen_n64(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Extract text from Hexen N64 - Doom engine derivative"""
        text_entries = []

        # Hexen uses WAD-like structure embedded in N64 ROM
        # Look for text lumps and string tables

        # Common Hexen text patterns
        hexen_patterns = [
            # Menu strings
            rb'NEW GAME\x00',
            rb'LOAD GAME\x00',
            rb'SAVE GAME\x00',
            rb'OPTIONS\x00',
            rb'QUIT\x00',

            # Character classes
            rb'FIGHTER\x00',
            rb'CLERIC\x00',
            rb'MAGE\x00',

            # Items and spells
            rb'BLUE MANA\x00',
            rb'GREEN MANA\x00',
            rb'COMBINED MANA\x00',

            # Hub messages
            rb'ENTERING\x00',
            rb'HUB\x00',
        ]

        # Search for Hexen-specific string tables
        for offset in range(0x1000, len(file_data) - 0x1000, 0x100):
            sector = file_data[offset:offset + 0x1000]

            # Look for string table markers
            if b'STRINGS' in sector or b'TEXT' in sector:
                strings = self._extract_hexen_strings(sector, offset)
                text_entries.extend(strings)

            # Pattern-based extraction
            for pattern in hexen_patterns:
                if pattern in sector:
                    context_start = max(0, sector.find(pattern) - 100)
                    context_end = min(len(sector), sector.find(pattern) + 200)
                    context_data = sector[context_start:context_end]

                    strings = self._extract_null_terminated_strings(
                        context_data, offset + context_start, TextEncoding.ASCII
                    )

                    for string_entry in strings:
                        string_entry.context = "hexen_menu"
                        text_entries.append(string_entry)

        # Remove duplicates
        unique_entries = []
        seen_texts = set()
        for entry in text_entries:
            if entry.original_text not in seen_texts:
                unique_entries.append(entry)
                seen_texts.add(entry.original_text)

        return unique_entries

    def _extract_hexen_strings(self, sector_data: bytes, base_offset: int) -> List[TextEntry]:
        """Extract strings from Hexen string table"""
        strings = []

        # Hexen string tables often have length prefixes
        i = 0
        while i < len(sector_data) - 4:
            # Try to read length prefix (16-bit big-endian)
            length = struct.unpack('>H', sector_data[i:i+2])[0]

            if 3 <= length <= 100 and i + 2 + length <= len(sector_data):
                string_data = sector_data[i+2:i+2+length]

                # Check if it's printable text
                try:
                    text = string_data.decode('ascii', errors='ignore').strip('\x00')
                    if self._is_valid_text(text) and len(text) > 2:
                        strings.append(TextEntry(
                            offset=base_offset + i,
                            original_text=text,
                            encoding=TextEncoding.ASCII,
                            context="hexen_string_table",
                            length=length + 2
                        ))
                        i += 2 + length
                        continue
                except:
                    pass

            i += 1

        return strings

    def _extract_quest64(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Extract text from Quest 64 (Holy Magic Century)"""
        text_entries = []

        # Quest 64 specific patterns
        quest64_patterns = [
            # Character names
            rb'BRIAN\x00',
            rb'ZELSE\x00',
            rb'MAMMON\x00',
            rb'BEIGIS\x00',

            # Spell names
            rb'AVALANCHE\x00',
            rb'LIGHTNING\x00',
            rb'ROCK BLAST\x00',
            rb'HEAL\x00',
            rb'BARRIER\x00',

            # Items
            rb'BREAD\x00',
            rb'WATER\x00',
            rb'MEDICINE\x00',
            rb'ANTIDOTE\x00',

            # Locations
            rb'CELTLAND\x00',
            rb'DONDORAN\x00',
            rb'BRANNOCH\x00',
            rb'CONNACHT\x00',
        ]

        # Quest 64 uses a unique text encoding system
        # Text is often stored in chunks with specific markers

        # Search for text data sections
        for offset in range(0x10000, len(file_data) - 0x2000, 0x200):
            sector = file_data[offset:offset + 0x2000]

            # Look for dialogue markers
            if b'\x00\x00\x01\x00' in sector or b'\xFF\xFF\x00\x00' in sector:
                strings = self._extract_quest64_dialogue(sector, offset)
                text_entries.extend(strings)

            # Pattern-based extraction
            for pattern in quest64_patterns:
                if pattern in sector:
                    # Extract surrounding context
                    pattern_offset = sector.find(pattern)
                    context_start = max(0, pattern_offset - 200)
                    context_end = min(len(sector), pattern_offset + 300)
                    context_data = sector[context_start:context_end]

                    strings = self._extract_null_terminated_strings(
                        context_data, offset + context_start, TextEncoding.ASCII
                    )

                    for string_entry in strings:
                        string_entry.context = "quest64_data"
                        text_entries.append(string_entry)

        # Extract menu text (usually in early ROM sections)
        menu_section = file_data[0x50000:0x80000]  # Common menu location
        menu_strings = self._extract_quest64_menu(menu_section, 0x50000)
        text_entries.extend(menu_strings)

        # Remove duplicates and filter
        unique_entries = []
        seen_combinations = set()

        for entry in text_entries:
            combo = (entry.offset, entry.original_text)
            if combo not in seen_combinations and len(entry.original_text) > 2:
                unique_entries.append(entry)
                seen_combinations.add(combo)

        return unique_entries

    def _extract_quest64_dialogue(self, sector_data: bytes, base_offset: int) -> List[TextEntry]:
        """Extract dialogue from Quest 64 dialogue sections"""
        strings = []

        # Quest 64 dialogue format: [marker][length][text][null]
        i = 0
        while i < len(sector_data) - 8:
            # Look for dialogue markers
            if (sector_data[i:i+2] == b'\x00\x01' or
                sector_data[i:i+2] == b'\xFF\x00'):

                # Try to read length (various formats)
                for length_format in ['>H', '<H', '>I', '<I']:
                    try:
                        if length_format.endswith('H'):
                            length = struct.unpack(length_format, sector_data[i+2:i+4])[0]
                            text_start = i + 4
                        else:
                            length = struct.unpack(length_format, sector_data[i+2:i+6])[0]
                            text_start = i + 6

                        if 5 <= length <= 200 and text_start + length <= len(sector_data):
                            text_data = sector_data[text_start:text_start + length]

                            # Remove null terminators and decode
                            text = text_data.rstrip(b'\x00').decode('ascii', errors='ignore')

                            if self._is_valid_text(text) and len(text) > 3:
                                strings.append(TextEntry(
                                    offset=base_offset + i,
                                    original_text=text,
                                    encoding=TextEncoding.ASCII,
                                    context="quest64_dialogue",
                                    length=text_start - i + length
                                ))
                                i = text_start + length
                                break
                    except:
                        continue
                else:
                    i += 1
            else:
                i += 1

        return strings

    def _extract_quest64_menu(self, sector_data: bytes, base_offset: int) -> List[TextEntry]:
        """Extract menu text from Quest 64"""
        strings = []

        # Quest 64 menu strings are often in tables
        # Look for repetitive patterns that indicate menu structures

        # Common menu string patterns
        menu_indicators = [
            b'NEW GAME',
            b'CONTINUE',
            b'OPTION',
            b'MAGIC',
            b'ITEM',
            b'STATUS',
            b'SAVE',
            b'LOAD'
        ]

        for indicator in menu_indicators:
            if indicator in sector_data:
                # Found menu section, extract surrounding strings
                start_pos = sector_data.find(indicator)

                # Extract strings in a window around the indicator
                window_start = max(0, start_pos - 500)
                window_end = min(len(sector_data), start_pos + 500)
                window_data = sector_data[window_start:window_end]

                menu_strings = self._extract_null_terminated_strings(
                    window_data, base_offset + window_start, TextEncoding.ASCII
                )

                for string_entry in menu_strings:
                    string_entry.context = "quest64_menu"
                    strings.append(string_entry)

        return strings

    def _extract_generic_n64(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Generic N64 text extraction for unknown games"""
        text_entries = []

        # N64 ROMs typically have text in specific sections
        # Skip the first 64KB (usually code/headers)
        text_start = 0x10000

        # Process in chunks to avoid memory issues
        chunk_size = 0x10000  # 64KB chunks

        for chunk_offset in range(text_start, len(file_data), chunk_size):
            chunk_end = min(chunk_offset + chunk_size, len(file_data))
            chunk_data = file_data[chunk_offset:chunk_end]

            # Multiple extraction strategies
            strategies = [
                lambda data, offset: self._extract_null_terminated_strings(data, offset, TextEncoding.ASCII),
                lambda data, offset: self._extract_length_prefixed_strings(data, offset, TextEncoding.ASCII),
                lambda data, offset: self._extract_pattern_based_strings(data, offset, TextEncoding.ASCII),
            ]

            for strategy in strategies:
                try:
                    chunk_strings = strategy(chunk_data, chunk_offset)
                    text_entries.extend(chunk_strings)
                except Exception as e:
                    logger.debug(f"N64 extraction strategy failed: {e}")

        # Filter and deduplicate
        filtered_entries = []
        seen_texts = set()

        for entry in text_entries:
            if (entry.original_text not in seen_texts and
                len(entry.original_text) > 2 and
                self._is_valid_text(entry.original_text)):
                filtered_entries.append(entry)
                seen_texts.add(entry.original_text)

    def _validate_n64_translation(self, entry: TextEntry, translated_text: str) -> bool:
        """Validate N64 translation considering hardware limitations"""
        # N64 has limited text rendering capabilities
        # Character limits and special considerations

        max_lengths = {
            "quest64_dialogue": 60,  # Quest 64 dialogue box limit
            "quest64_menu": 12,      # Menu item limit
            "hexen_menu": 15,        # Hexen menu limit
            "hexen_string_table": 40, # General string limit
            "default": 50
        }

        max_length = max_lengths.get(entry.context, max_lengths["default"])

        # Check length constraints
        if len(translated_text) > max_length:
            logger.warning(f"Translation too long for N64: '{translated_text}' ({len(translated_text)} chars)")
            return False

        # Check for unsupported characters
        # N64 games typically support limited character sets
        supported_chars = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " .,!?'-()[]"
            "áéíóúâêîôûàèìòùãõç"  # Common Portuguese chars
        )

        for char in translated_text:
            if char not in supported_chars:
                logger.warning(f"Unsupported character in N64 translation: '{char}'")
                return False

        return True

    def patch_n64_rom(self, rom_path: str, text_entries: List[TextEntry], output_path: str) -> bool:
        """Patch N64 ROM with translated text - handles endianness"""
        logger.info(f"Patching N64 ROM: {rom_path} -> {output_path}")

        try:
            # Read and normalize ROM
            with open(rom_path, 'rb') as f:
                original_data = f.read()

            # Detect original endianness
            original_endian = self._detect_n64_endianness(original_data)

            # Normalize to big-endian for patching
            normalized_data = bytearray(self._normalize_n64_rom(original_data))

            # Apply translations
            patches_applied = 0
            for entry in text_entries:
                if not entry.translated_text or entry.translated_text.startswith("[UNTRANSLATED]"):
                    continue

                # Validate translation for N64
                if not self._validate_n64_translation(entry, entry.translated_text):
                    logger.warning(f"Skipping invalid N64 translation: {entry.original_text}")
                    continue

                # Encode translated text
                try:
                    encoded_translation = entry.translated_text.encode('ascii', errors='replace')

                    # Add null terminator if original had one
                    if entry.original_text.endswith('\x00'):
                        encoded_translation += b'\x00'

                    # Check if translation fits
                    if len(encoded_translation) <= entry.length:
                        # Pad with null bytes or spaces depending on context
                        if "menu" in entry.context:
                            padded_translation = encoded_translation.ljust(entry.length, b' ')
                        else:
                            padded_translation = encoded_translation.ljust(entry.length, b'\x00')

                        # Apply patch
                        end_offset = entry.offset + entry.length
                        if end_offset <= len(normalized_data):
                            normalized_data[entry.offset:end_offset] = padded_translation
                            patches_applied += 1

                except Exception as e:
                    logger.error(f"Failed to encode translation: {e}")

            # Convert back to original endianness
            if original_endian != "big":
                final_data = self._convert_n64_endianness(normalized_data, original_endian)
            else:
                final_data = normalized_data

            # Write patched ROM
            with open(output_path, 'wb') as f:
                f.write(final_data)

            # Update ROM header checksum for N64
            self._update_n64_checksum(output_path)

            logger.info(f"N64 ROM patching complete: {patches_applied} patches applied")
            return True

        except Exception as e:
            logger.error(f"N64 ROM patching failed: {e}")
            return False

    def _detect_n64_endianness(self, file_data: bytes) -> str:
        """Detect N64 ROM endianness"""
        if len(file_data) < 4:
            return "unknown"

        header = file_data[:4]
        if header == b'\x80\x37\x12\x40':
            return "big"
        elif header == b'\x40\x12\x37\x80':
            return "little"
        elif header == b'\x37\x80\x40\x12':
            return "mixed"

        return "unknown"

    def _convert_n64_endianness(self, data: bytes, target_endian: str) -> bytes:
        """Convert N64 ROM endianness"""
        if target_endian == "big":
            return data
        elif target_endian == "little":
            # Convert to little-endian
            return b''.join(struct.pack('<I', struct.unpack('>I', data[i:i+4])[0])
                           for i in range(0, len(data), 4))
        elif target_endian == "mixed":
            # Convert to mixed/byte-swapped
            return b''.join(data[i+1:i+2] + data[i:i+1] + data[i+3:i+4] + data[i+2:i+3]
                           for i in range(0, len(data), 4))

        return data

    def _update_n64_checksum(self, rom_path: str) -> None:
        """Update N64 ROM checksum after patching"""
        try:
            with open(rom_path, 'r+b') as f:
                # Read ROM data
                f.seek(0)
                rom_data = f.read()

                # Calculate checksum for N64 ROM
                # This is a simplified checksum - real N64 checksums are more complex
                checksum = 0
                for i in range(0x40, min(0x101000, len(rom_data)), 4):
                    if i + 4 <= len(rom_data):
                        word = struct.unpack('>I', rom_data[i:i+4])[0]
                        checksum = (checksum + word) & 0xFFFFFFFF

                # Write checksum to header
                f.seek(0x10)
                f.write(struct.pack('>I', checksum))
                f.write(struct.pack('>I', checksum ^ 0xFFFFFFFF))

        except Exception as e:
            logger.warning(f"Failed to update N64 checksum: {e}")

    def create_n64_translation_project(self, rom_path: str, output_dir: str) -> str:
        """Create N64-specific translation project"""
        logger.info(f"Creating N64 translation project for {rom_path}")

        # Create project directory
        project_dir = Path(output_dir) / f"n64_translation_{Path(rom_path).stem}"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Analyze ROM
        rom_info = self.analyze_rom(rom_path)

        # Extract text with N64-specific methods
        text_entries = self.extract_text(rom_path, rom_info)

        # Create game-specific translation guide
        game_id = self._get_n64_game_id(open(rom_path, 'rb').read())
        guide_content = self._create_n64_translation_guide(game_id, text_entries)

        # Save translation guide
        guide_file = project_dir / "translation_guide.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        # Save project data with N64-specific metadata
        project_data = {
            "platform": "Nintendo 64",
            "game_id": game_id,
            "rom_info": {
                "filename": rom_info.filename,
                "file_size": rom_info.file_size,
                "md5_hash": rom_info.md5_hash,
                "rom_type": rom_info.rom_type.value,
                "text_encoding": rom_info.text_encoding.value,
                "endianness": self._detect_n64_endianness(open(rom_path, 'rb').read()),
                "region": rom_info.region
            },
            "text_entries": [
                {
                    "offset": entry.offset,
                    "original_text": entry.original_text,
                    "translated_text": entry.translated_text,
                    "encoding": entry.encoding.value,
                    "context": entry.context,
                    "length": entry.length,
                    "max_length": self._get_n64_max_length(entry.context)
                }
                for entry in text_entries
            ],
            "translation_notes": {
                "character_limits": True,
                "special_chars": "Limited to ASCII + basic Portuguese accents",
                "line_breaks": "Use \\n for line breaks",
                "formatting": "Preserve original formatting codes"
            }
        }

        # Save project file
        project_file = project_dir / "n64_project.json"
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        # Create specialized CSV for N64 translation
        csv_file = project_dir / "n64_translations.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Offset,Context,Original,Translation,Max_Length,Notes\n")
            for entry in text_entries:
                max_len = self._get_n64_max_length(entry.context)
                notes = self._get_n64_translation_notes(entry.context)
                f.write(f"{entry.offset},{entry.context},\"{entry.original_text}\",\"{entry.translated_text}\",{max_len},\"{notes}\"\n")

        logger.info(f"N64 translation project created: {project_dir}")
        return str(project_dir)

    def _get_n64_max_length(self, context: str) -> int:
        """Get maximum length for N64 context"""
        limits = {
            "quest64_dialogue": 60,
            "quest64_menu": 12,
            "hexen_menu": 15,
            "hexen_string_table": 40,
            "default": 50
        }
        return limits.get(context, limits["default"])

    def _get_n64_translation_notes(self, context: str) -> str:
        """Get translation notes for N64 context"""
        notes = {
            "quest64_dialogue": "Dialogue box limit. Use short sentences.",
            "quest64_menu": "Menu item. Keep very short.",
            "hexen_menu": "Menu text. Avoid special characters.",
            "hexen_string_table": "Game text. May affect gameplay.",
            "default": "Standard text. Check character limits."
        }
        return notes.get(context, notes["default"])

    def _create_n64_translation_guide(self, game_id: str, text_entries: List[TextEntry]) -> str:
        """Create N64-specific translation guide"""
        guide = f"""# N64 Translation Guide - {game_id}

## Game-Specific Information

### {"Hexen 64" if "HEXEN" in game_id else "Quest 64" if "QUEST" in game_id else "Unknown Game"}

#### Technical Limitations:
- **Character Set**: ASCII + basic Portuguese accents (á, é, í, ó, ú, ç, ã, õ)
- **Line Length**: Varies by context (see individual entries)
- **Special Characters**: Limited support
- **Encoding**: ASCII with null terminators

#### Translation Tips:
"""

        if "HEXEN" in game_id:
            guide += """
##### Hexen 64 Specific:
- **Menu Items**: Keep under 15 characters
- **Spell Names**: Maintain mystical/medieval tone
- **Character Classes**: FIGHTER, CLERIC, MAGE (traditional names)
- **Items**: Keep descriptive but concise
- **Hub Messages**: Location names should be atmospheric

##### Common Terms:
- Mana = Mana (keep as is)
- Health = Saúde
- Armor = Armadura
- Magic = Magia
- Weapon = Arma
"""

        elif "QUEST" in game_id:
            guide += """
##### Quest 64 Specific:
- **Dialogue**: Maximum 60 characters per line
- **Menu Items**: Maximum 12 characters
- **Spell Names**: Keep magical feel
- **Character Names**: Consider keeping original names
- **Location Names**: Celtic/Irish inspired names

##### Common Terms:
- Magic = Magia
- Spirit = Espírito
- Element = Elemento
- Earth = Terra
- Water = Água
- Fire = Fogo
- Wind = Vento
"""

        guide += f"""
## Statistics:
- **Total Entries**: {len(text_entries)}
- **Dialogue Entries**: {len([e for e in text_entries if 'dialogue' in e.context])}
- **Menu Entries**: {len([e for e in text_entries if 'menu' in e.context])}
- **System Entries**: {len([e for e in text_entries if 'system' in e.context])}

## Context Types Found:
"""

        contexts = set(entry.context for entry in text_entries)
        for context in sorted(contexts):
            count = len([e for e in text_entries if e.context == context])
            guide += f"- **{context}**: {count} entries\n"

        guide += """
## Translation Workflow:
1. Review this guide thoroughly
2. Edit `n64_translations.csv` with your translations
3. Test length limits for each context
4. Run the patcher to generate the translated ROM
5. Test on emulator (Project64 recommended)

## Quality Checklist:
- [ ] All translations fit within character limits
- [ ] No unsupported characters used
- [ ] Game-specific terminology maintained
- [ ] Context-appropriate language tone
- [ ] Null terminators preserved where needed
"""

        return guide_from_iso(file_data, rom_info)
        elif rom_info.rom_type == ROMType.ELF:
            text_entries = self._extract_from_elf(file_data, rom_info)
        else:
            text_entries = self._extract_generic_text(file_data, rom_info)

        logger.info(f"Extracted {len(text_entries)} text entries")
        return text_entries

    def _extract_from_iso(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Extract text from ISO files"""
        text_entries = []

        # PS2 ISO structure analysis
        # Look for text in common sections
        for offset in range(0, len(file_data) - 32, 2048):  # ISO sectors
            sector = file_data[offset:offset+2048]

            # Skip binary sectors
            if sector.count(b'\x00') > len(sector) * 0.8:
                continue

            # Extract strings
            strings = self._extract_strings_from_sector(sector, offset, rom_info.text_encoding)
            text_entries.extend(strings)

        return text_entries

    def _extract_from_elf(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Extract text from ELF files"""
        text_entries = []

        # ELF header analysis
        if len(file_data) < 52:
            return text_entries

        # Read ELF header
        e_shoff = struct.unpack('<I', file_data[32:36])[0]  # Section header offset
        e_shentsize = struct.unpack('<H', file_data[46:48])[0]  # Section header size
        e_shnum = struct.unpack('<H', file_data[48:50])[0]  # Number of sections

        # Extract from each section
        for i in range(e_shnum):
            sh_offset = e_shoff + i * e_shentsize
            if sh_offset + 40 > len(file_data):
                break

            # Read section header
            sh_type = struct.unpack('<I', file_data[sh_offset+4:sh_offset+8])[0]
            sh_addr = struct.unpack('<I', file_data[sh_offset+12:sh_offset+16])[0]
            sh_offset_data = struct.unpack('<I', file_data[sh_offset+16:sh_offset+20])[0]
            sh_size = struct.unpack('<I', file_data[sh_offset+20:sh_offset+24])[0]

            # Extract strings from relevant sections
            if sh_type in [3, 1]:  # STRTAB or PROGBITS
                if sh_offset_data + sh_size <= len(file_data):
                    section_data = file_data[sh_offset_data:sh_offset_data+sh_size]
                    strings = self._extract_strings_from_sector(section_data, sh_offset_data, rom_info.text_encoding)
                    text_entries.extend(strings)

        return text_entries

    def _extract_generic_text(self, file_data: bytes, rom_info: ROMInfo) -> List[TextEntry]:
        """Generic text extraction for unknown ROM types"""
        text_entries = []

        # Use sliding window approach
        window_size = 1024
        for offset in range(0, len(file_data) - window_size, window_size // 2):
            window = file_data[offset:offset + window_size]
            strings = self._extract_strings_from_sector(window, offset, rom_info.text_encoding)
            text_entries.extend(strings)

        return text_entries

    def _extract_strings_from_sector(self, sector_data: bytes, base_offset: int, encoding: TextEncoding) -> List[TextEntry]:
        """Extract strings from a data sector"""
        strings = []

        # Multiple string extraction methods
        methods = [
            self._extract_null_terminated_strings,
            self._extract_length_prefixed_strings,
            self._extract_pattern_based_strings
        ]

        for method in methods:
            try:
                method_strings = method(sector_data, base_offset, encoding)
                strings.extend(method_strings)
            except Exception as e:
                logger.debug(f"String extraction method failed: {e}")

        # Remove duplicates and filter
        unique_strings = []
        seen_offsets = set()

        for string_entry in strings:
            if string_entry.offset not in seen_offsets and self._is_valid_text(string_entry.original_text):
                unique_strings.append(string_entry)
                seen_offsets.add(string_entry.offset)

        return unique_strings

    def _extract_null_terminated_strings(self, data: bytes, base_offset: int, encoding: TextEncoding) -> List[TextEntry]:
        """Extract null-terminated strings"""
        strings = []
        current_string = b""
        string_start = 0

        for i, byte in enumerate(data):
            if byte == 0:
                if len(current_string) > 3:  # Minimum string length
                    try:
                        decoded = self._decode_text(current_string, encoding)
                        if decoded:
                            strings.append(TextEntry(
                                offset=base_offset + string_start,
                                original_text=decoded,
                                encoding=encoding,
                                length=len(current_string)
                            ))
                    except UnicodeDecodeError:
                        pass

                current_string = b""
                string_start = i + 1
            else:
                current_string += bytes([byte])

        return strings

    def _extract_length_prefixed_strings(self, data: bytes, base_offset: int, encoding: TextEncoding) -> List[TextEntry]:
        """Extract length-prefixed strings"""
        strings = []
        i = 0

        while i < len(data) - 2:
            # Try different length prefix sizes
            for prefix_size in [1, 2, 4]:
                if i + prefix_size >= len(data):
                    break

                # Read length
                if prefix_size == 1:
                    length = data[i]
                elif prefix_size == 2:
                    length = struct.unpack('<H', data[i:i+2])[0]
                else:  # prefix_size == 4
                    length = struct.unpack('<I', data[i:i+4])[0]

                # Validate length
                if 4 <= length <= 256 and i + prefix_size + length <= len(data):
                    string_data = data[i + prefix_size:i + prefix_size + length]
                    try:
                        decoded = self._decode_text(string_data, encoding)
                        if decoded and self._is_valid_text(decoded):
                            strings.append(TextEntry(
                                offset=base_offset + i,
                                original_text=decoded,
                                encoding=encoding,
                                length=length + prefix_size
                            ))
                            i += prefix_size + length
                            break
                    except UnicodeDecodeError:
                        pass
            else:
                i += 1

        return strings

    def _extract_pattern_based_strings(self, data: bytes, base_offset: int, encoding: TextEncoding) -> List[TextEntry]:
        """Extract strings based on common patterns"""
        strings = []

        # Look for common game text patterns
        patterns = [
            rb'[A-Za-z0-9 ]{4,}',  # ASCII text
            rb'[\x20-\x7E]{4,}',   # Printable ASCII
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, data)
            for match in matches:
                try:
                    decoded = self._decode_text(match.group(), encoding)
                    if decoded and self._is_valid_text(decoded):
                        strings.append(TextEntry(
                            offset=base_offset + match.start(),
                            original_text=decoded,
                            encoding=encoding,
                            length=len(match.group())
                        ))
                except UnicodeDecodeError:
                    pass

        return strings

    def _decode_text(self, text_bytes: bytes, encoding: TextEncoding) -> str:
        """Decode text bytes using specified encoding"""
        if encoding == TextEncoding.ASCII:
            return text_bytes.decode('ascii', errors='ignore')
        elif encoding == TextEncoding.SHIFT_JIS:
            return text_bytes.decode('shift_jis', errors='ignore')
        elif encoding == TextEncoding.UTF8:
            return text_bytes.decode('utf-8', errors='ignore')
        elif encoding == TextEncoding.UTF16:
            return text_bytes.decode('utf-16', errors='ignore')
        else:
            return text_bytes.decode('ascii', errors='ignore')

    def _is_valid_text(self, text: str) -> bool:
        """Check if extracted text is valid for translation"""
        if not text or len(text) < 3:
            return False

        # Filter out binary data disguised as text
        if text.count('\x00') > len(text) * 0.3:
            return False

        # Check for reasonable character distribution
        printable_chars = sum(1 for c in text if c.isprintable())
        if printable_chars < len(text) * 0.7:
            return False

        # Filter out strings that are likely not user-facing text
        skip_patterns = [
            r'^[0-9A-F]{8,}$',  # Hex strings
            r'^[0-9]+$',        # Pure numbers
            r'^[^a-zA-Z]*$',    # No letters
        ]

        for pattern in skip_patterns:
            if re.match(pattern, text):
                return False

        return True

    def translate_text(self, text_entries: List[TextEntry],
                      source_lang: str = "auto",
                      target_lang: str = "pt-BR") -> List[TextEntry]:
        """Translate extracted text entries"""
        logger.info(f"Translating {len(text_entries)} text entries to {target_lang}")

        for entry in text_entries:
            if not entry.original_text.strip():
                continue

            # Try AI translation first
            if HAS_OPENAI and self.openai_api_key:
                try:
                    translated = self._translate_with_openai(entry.original_text, source_lang, target_lang)
                    if translated:
                        entry.translated_text = translated
                        continue
                except Exception as e:
                    logger.debug(f"OpenAI translation failed: {e}")

            # Fallback to other translation methods
            if HAS_REQUESTS:
                try:
                    translated = self._translate_with_google(entry.original_text, source_lang, target_lang)
                    if translated:
                        entry.translated_text = translated
                        continue
                except Exception as e:
                    logger.debug(f"Google translation failed: {e}")

            # Last resort: mark as untranslated
            entry.translated_text = f"[UNTRANSLATED] {entry.original_text}"

        logger.info("Translation complete")
        return text_entries

    def _translate_with_openai(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using OpenAI API"""
        if not HAS_OPENAI:
            return ""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a professional video game translator. Translate the following text from {source_lang} to {target_lang}. Maintain the original tone and gaming context. Keep the translation natural and appropriate for gamers."},
                    {"role": "user", "content": text}
                ],
                max_tokens=150,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return ""

    def _translate_with_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Google Translate (unofficial API)"""
        if not HAS_REQUESTS:
            return ""

        try:
            # Simple Google Translate API call
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": text
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    return result[0][0][0]

        except Exception as e:
            logger.error(f"Google translation error: {e}")

        return ""

    def patch_rom(self, rom_path: str, text_entries: List[TextEntry], output_path: str) -> bool:
        """Patch ROM with translated text"""
        logger.info(f"Patching ROM: {rom_path} -> {output_path}")

        try:
            # Read original ROM
            with open(rom_path, 'rb') as f:
                rom_data = bytearray(f.read())

            # Apply translations
            patches_applied = 0
            for entry in text_entries:
                if not entry.translated_text or entry.translated_text.startswith("[UNTRANSLATED]"):
                    continue

                # Encode translated text
                try:
                    encoded_translation = entry.translated_text.encode(entry.encoding.value)

                    # Check if translation fits in original space
                    if len(encoded_translation) <= entry.length:
                        # Pad with null bytes if needed
                        padded_translation = encoded_translation.ljust(entry.length, b'\x00')

                        # Apply patch
                        end_offset = entry.offset + entry.length
                        if end_offset <= len(rom_data):
                            rom_data[entry.offset:end_offset] = padded_translation
                            patches_applied += 1

                except UnicodeEncodeError:
                    logger.warning(f"Could not encode translation: {entry.translated_text}")

            # Write patched ROM
            with open(output_path, 'wb') as f:
                f.write(rom_data)

            logger.info(f"Patching complete: {patches_applied} patches applied")
            return True

        except Exception as e:
            logger.error(f"ROM patching failed: {e}")
            return False

    def create_translation_project(self, rom_path: str, output_dir: str) -> str:
        """Create a complete translation project"""
        logger.info(f"Creating translation project for {rom_path}")

        # Create project directory
        project_dir = Path(output_dir) / f"translation_project_{Path(rom_path).stem}"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Analyze ROM
        rom_info = self.analyze_rom(rom_path)

        # Extract text
        text_entries = self.extract_text(rom_path, rom_info)

        # Save project data
        project_data = {
            "rom_info": {
                "filename": rom_info.filename,
                "file_size": rom_info.file_size,
                "md5_hash": rom_info.md5_hash,
                "rom_type": rom_info.rom_type.value,
                "text_encoding": rom_info.text_encoding.value,
                "game_id": rom_info.game_id,
                "region": rom_info.region
            },
            "text_entries": [
                {
                    "offset": entry.offset,
                    "original_text": entry.original_text,
                    "translated_text": entry.translated_text,
                    "encoding": entry.encoding.value,
                    "context": entry.context,
                    "length": entry.length
                }
                for entry in text_entries
            ]
        }

        # Save project file
        project_file = project_dir / "project.json"
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        # Create translation CSV for easy editing
        csv_file = project_dir / "translations.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Offset,Original,Translation,Context,Length\n")
            for entry in text_entries:
                f.write(f"{entry.offset},\"{entry.original_text}\",\"{entry.translated_text}\",\"{entry.context}\",{entry.length}\n")

        logger.info(f"Translation project created: {project_dir}")
        return str(project_dir)

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="PS2 ROM Translation Engine")
    parser.add_argument("rom_path", help="Path to PS2 ROM file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--target-lang", "-t", default="pt-BR", help="Target language")
    parser.add_argument("--openai-key", help="OpenAI API key for AI translation")
    parser.add_argument("--translate", action="store_true", help="Auto-translate extracted text")
    parser.add_argument("--patch", help="Patch ROM with translations from project file")

    args = parser.parse_args()

    # Initialize translator
    translator = PS2ROMTranslator(openai_api_key=args.openai_key)

    if args.patch:
        # Load project and patch ROM
        with open(args.patch, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        text_entries = [
            TextEntry(
                offset=entry["offset"],
                original_text=entry["original_text"],
                translated_text=entry["translated_text"],
                encoding=TextEncoding(entry["encoding"]),
                context=entry["context"],
                length=entry["length"]
            )
            for entry in project_data["text_entries"]
        ]

        output_rom = f"{args.output}/patched_rom.iso"
        translator.patch_rom(args.rom_path, text_entries, output_rom)
        print(f"Patched ROM saved to: {output_rom}")

    else:
        # Create translation project
        project_dir = translator.create_translation_project(args.rom_path, args.output)

        if args.translate:
            # Load project and translate
            with open(f"{project_dir}/project.json", 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            text_entries = [
                TextEntry(
                    offset=entry["offset"],
                    original_text=entry["original_text"],
                    encoding=TextEncoding(entry["encoding"]),
                    context=entry["context"],
                    length=entry["length"]
                )
                for entry in project_data["text_entries"]
            ]

            # Translate
            translated_entries = translator.translate_text(text_entries, target_lang=args.target_lang)

            # Update project
            project_data["text_entries"] = [
                {
                    "offset": entry.offset,
                    "original_text": entry.original_text,
                    "translated_text": entry.translated_text,
                    "encoding": entry.encoding.value,
                    "context": entry.context,
                    "length": entry.length
                }
                for entry in translated_entries
            ]

            with open(f"{project_dir}/project.json", 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            print(f"Translation complete! Project saved to: {project_dir}")

        print(f"Translation project created: {project_dir}")
        print("Next steps:")
        print("1. Review translations.csv and edit as needed")
        print("2. Run with --patch to create patched ROM")

if __name__ == "__main__":
    main()