"""
Base Engine for ROM Translation
===============================

This module provides the base classes for ROM translation engines.
All console-specific engines should inherit from these base classes.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

class BaseEngine(ABC):
    """
    Abstract base class for ROM translation engines.

    This class defines the interface that all ROM translation engines must implement.
    It provides common functionality and ensures consistency across different console engines.
    """

    def __init__(self, console_name: str = "Unknown"):
        """
        Initialize the base engine.

        Args:
            console_name (str): Name of the console this engine handles
        """
        self.console_name = console_name
        self.supported_formats = []
        self.rom_data = None
        self.text_regions = []

    @abstractmethod
    def load_rom(self, rom_path: str) -> bool:
        """
        Load ROM file and prepare it for translation.

        Args:
            rom_path (str): Path to the ROM file

        Returns:
            bool: True if ROM was loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def extract_text(self) -> Dict[str, str]:
        """
        Extract translatable text from the ROM.

        Returns:
            dict: Dictionary mapping text IDs to original text
        """
        pass

    @abstractmethod
    def insert_text(self, translations: Dict[str, str]) -> bool:
        """
        Insert translated text back into the ROM.

        Args:
            translations (dict): Dictionary mapping text IDs to translated text

        Returns:
            bool: True if text was inserted successfully, False otherwise
        """
        pass

    @abstractmethod
    def save_rom(self, output_path: str) -> bool:
        """
        Save the modified ROM to a file.

        Args:
            output_path (str): Path where the translated ROM should be saved

        Returns:
            bool: True if ROM was saved successfully, False otherwise
        """
        pass

    def validate_rom(self, rom_path: str) -> bool:
        """
        Validate if the ROM file is compatible with this engine.

        Args:
            rom_path (str): Path to the ROM file

        Returns:
            bool: True if ROM is compatible, False otherwise
        """
        if not os.path.exists(rom_path):
            return False

        # Check file extension
        _, ext = os.path.splitext(rom_path)
        return ext.lower() in self.supported_formats

    def get_console_info(self) -> Dict[str, Any]:
        """
        Get information about the console this engine handles.

        Returns:
            dict: Console information including name and supported formats
        """
        return {
            'name': self.console_name,
            'supported_formats': self.supported_formats
        }


class EnhancedBaseEngine(BaseEngine):
    """
    Enhanced base engine with additional common functionality.

    This class extends BaseEngine with commonly used features that
    multiple console engines might need.
    """

    def __init__(self, console_name: str = "Unknown"):
        """
        Initialize the enhanced base engine.

        Args:
            console_name (str): Name of the console this engine handles
        """
        super().__init__(console_name)
        self.text_encoding = 'utf-8'
        self.pointer_tables = {}
        self.text_cache = {}

    def set_text_encoding(self, encoding: str):
        """
        Set the text encoding for this ROM.

        Args:
            encoding (str): Text encoding (e.g., 'utf-8', 'shift_jis')
        """
        self.text_encoding = encoding

    def find_text_patterns(self, data: bytes, patterns: List[str]) -> List[Tuple[int, str]]:
        """
        Find text patterns in ROM data using regular expressions.

        Args:
            data (bytes): ROM data to search
            patterns (list): List of regex patterns to search for

        Returns:
            list: List of tuples (offset, matched_text)
        """
        matches = []

        for pattern in patterns:
            try:
                # Convert bytes to string for regex matching
                text_data = data.decode(self.text_encoding, errors='ignore')

                for match in re.finditer(pattern, text_data):
                    offset = match.start()
                    matched_text = match.group()
                    matches.append((offset, matched_text))

            except Exception as e:
                print(f"Error searching pattern '{pattern}': {e}")
                continue

        return matches

    def calculate_checksum(self, data: bytes) -> int:
        """
        Calculate checksum for ROM data.

        Args:
            data (bytes): ROM data

        Returns:
            int: Calculated checksum
        """
        return sum(data) & 0xFFFF

    def create_backup(self, rom_path: str) -> str:
        """
        Create a backup of the original ROM file.

        Args:
            rom_path (str): Path to the original ROM file

        Returns:
            str: Path to the backup file
        """
        backup_path = rom_path + '.backup'

        try:
            with open(rom_path, 'rb') as original:
                with open(backup_path, 'wb') as backup:
                    backup.write(original.read())
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None

    def get_text_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted text.

        Returns:
            dict: Statistics including text count, total length, etc.
        """
        if not self.text_cache:
            return {'text_count': 0, 'total_length': 0}

        total_length = sum(len(text) for text in self.text_cache.values())

        return {
            'text_count': len(self.text_cache),
            'total_length': total_length,
            'average_length': total_length / len(self.text_cache) if self.text_cache else 0
        }