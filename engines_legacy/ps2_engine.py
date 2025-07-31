#!/usr/bin/env python3
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
    """PlayStation 2 ROM file types"""
    ISO = "iso"
    BIN = "bin"
    ELF = "elf"
    IRX = "irx"
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

class PS2ROMTranslator:
    """Main PS2 ROM Translation Engine"""

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
        """Detect the type of ROM file"""
        extension = Path(rom_path).suffix.lower()

        if extension == '.iso':
            return ROMType.ISO
        elif extension == '.bin':
            return ROMType.BIN
        elif extension == '.elf':
            return ROMType.ELF
        elif extension == '.irx':
            return ROMType.IRX

        # Check file signatures
        if file_data[:4] == b'\x7fELF':
            return ROMType.ELF
        elif b'PlayStation' in file_data[:0x1000]:
            return ROMType.ISO

        return ROMType.UNKNOWN

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