#!/usr/bin/env python3
"""
Sega CD Translation Engine - Sistema Completo
Versão 4

Inspirado na arquitetura da Nintendo DS Translation Engine,
adaptado para as peculiaridades do Sega CD.

Suporta:
- Análise de estruturas CUE/BIN/ISO
- Detecção automática de texto em múltiplas codificações
- Extração de assets gráficos
- Processamento de áudio
- Aplicação de traduções com backup automático
- Suporte a múltiplos formatos de disco
"""

import os
import sys
import json
import hashlib
import subprocess
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
import shutil
import tempfile
from datetime import datetime

# Dependências externas necessárias
try:
    import chardet
    import Pillow as PIL
    from PIL import Image
    import numpy as np
    import pandas as pd
    import yaml
except ImportError as e:
    print(f"Erro: Dependência necessária não encontrada: {e}")
    print("Instale com: pip install chardet pillow numpy pandas pyyaml")
    sys.exit(1)

class SegaCDFormat(Enum):
    """Formatos de disco suportados pelo Sega CD"""
    BIN_CUE = "bin_cue"
    ISO = "iso"
    CHD = "chd"
    MDF_MDS = "mdf_mds"
    NRG = "nrg"

class TrackType(Enum):
    """Tipos de trilha em um CD"""
    MODE1_2048 = "MODE1/2048"    # Dados padrão
    MODE1_2352 = "MODE1/2352"    # Dados com subchannel
    MODE2_2336 = "MODE2/2336"    # Dados XA
    MODE2_2352 = "MODE2/2352"    # Dados XA com subchannel
    AUDIO = "AUDIO"              # Trilha de áudio

@dataclass
class SegaCDGameInfo:
    """Informações básicas do jogo Sega CD"""
    title: str
    region: str
    publisher: str
    year: int
    format: SegaCDFormat
    tracks: List[Dict]
    estimated_size_mb: float

class SegaCDROMAnalyzer:
    """Analisador de estruturas de ROM do Sega CD"""

    def __init__(self):
        self.supported_formats = ['.cue', '.bin', '.iso', '.chd', '.mdf', '.nrg']
        self.temp_dir = None

    def analyze_disc(self, rom_path: str) -> Dict:
        """Analisa um disco Sega CD e retorna informações estruturais"""
        rom_path = Path(rom_path)

        if not rom_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {rom_path}")

        # Determina o formato baseado na extensão
        format_type = self._detect_format(rom_path)

        # Análise específica por formato
        if format_type == SegaCDFormat.BIN_CUE:
            return self._analyze_bin_cue(rom_path)
        elif format_type == SegaCDFormat.ISO:
            return self._analyze_iso(rom_path)
        elif format_type == SegaCDFormat.CHD:
            return self._analyze_chd(rom_path)
        else:
            raise ValueError(f"Formato não suportado: {format_type}")

    def _detect_format(self, rom_path: Path) -> SegaCDFormat:
        """Detecta o formato do arquivo de disco"""
        extension = rom_path.suffix.lower()

        format_map = {
            '.cue': SegaCDFormat.BIN_CUE,
            '.bin': SegaCDFormat.BIN_CUE,
            '.iso': SegaCDFormat.ISO,
            '.chd': SegaCDFormat.CHD,
            '.mdf': SegaCDFormat.MDF_MDS,
            '.nrg': SegaCDFormat.NRG
        }

        return format_map.get(extension, SegaCDFormat.ISO)

    def _analyze_bin_cue(self, cue_path: Path) -> Dict:
        """Analisa um par BIN/CUE"""
        if cue_path.suffix.lower() == '.bin':
            # Se foi passado o .bin, procura o .cue correspondente
            cue_path = cue_path.with_suffix('.cue')

        if not cue_path.exists():
            raise FileNotFoundError(f"Arquivo CUE não encontrado: {cue_path}")

        # Parse do arquivo CUE
        track_info = self._parse_cue_file(cue_path)

        # Análise de cada trilha
        analysis = {
            'total_tracks': len(track_info),
            'data_tracks': [],
            'audio_tracks': [],
            'estimated_content': {}
        }

        for track_num, track_data in track_info.items():
            track_analysis = {
                'number': track_num,
                'type': track_data['type'],
                'file': track_data['file'],
                'size_mb': self._estimate_track_size(track_data)
            }

            if track_data['type'] == 'MODE1/2048':
                # Trilha de dados - provavelmente contém o jogo
                track_analysis['contains_game_data'] = True
                track_analysis['filesystem'] = self._analyze_data_track(track_data)
                analysis['data_tracks'].append(track_analysis)

            elif track_data['type'] == 'MODE2/2336':
                # Trilha XA - pode conter dados misturados com áudio
                track_analysis['contains_xa_data'] = True
                track_analysis['mixed_content'] = self._analyze_xa_track(track_data)
                analysis['data_tracks'].append(track_analysis)

            elif track_data['type'] == 'AUDIO':
                # Trilha de áudio puro
                track_analysis['duration_seconds'] = self._estimate_audio_duration(track_data)
                track_analysis['format'] = 'CD Audio (16-bit 44.1kHz)'
                analysis['audio_tracks'].append(track_analysis)

        # Estimativa do conteúdo principal
        if analysis['data_tracks']:
            main_track = analysis['data_tracks'][0]  # Primeira trilha de dados
            analysis['estimated_content'] = {
                'executable_size_mb': main_track['size_mb'] * 0.3,  # ~30% executável
                'graphics_size_mb': main_track['size_mb'] * 0.4,   # ~40% gráficos
                'audio_size_mb': main_track['size_mb'] * 0.2,      # ~20% áudio
                'text_size_mb': main_track['size_mb'] * 0.1        # ~10% texto
            }

        return analysis

    def _parse_cue_file(self, cue_path: Path) -> Dict:
        """Faz o parse de um arquivo CUE"""
        tracks = {}
        current_track = None
        current_file = None

        with open(cue_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                if line.startswith('FILE'):
                    # FILE "nome.bin" BINARY
                    parts = line.split('"')
                    if len(parts) >= 2:
                        current_file = parts[1]

                elif line.startswith('TRACK'):
                    # TRACK 01 MODE1/2048
                    parts = line.split()
                    if len(parts) >= 3:
                        track_num = int(parts[1])
                        track_type = parts[2]

                        current_track = {
                            'file': current_file,
                            'type': track_type,
                            'index': {}
                        }
                        tracks[track_num] = current_track

                elif line.startswith('INDEX') and current_track:
                    # INDEX 01 00:00:00
                    parts = line.split()
                    if len(parts) >= 3:
                        index_num = int(parts[1])
                        time_str = parts[2]
                        current_track['index'][index_num] = time_str

        return tracks

    def _estimate_track_size(self, track_data: Dict) -> float:
        """Estima o tamanho de uma trilha em MB"""
        if not track_data.get('file'):
            return 0.0

        # Busca o arquivo BIN correspondente
        bin_path = Path(track_data['file'])
        if bin_path.exists():
            size_bytes = bin_path.stat().st_size
            return size_bytes / (1024 * 1024)  # Convert to MB

        # Estimativa baseada no tipo de trilha
        track_type = track_data.get('type', '')
        if 'MODE1' in track_type:
            return 650.0  # Estimativa para CD padrão
        elif 'AUDIO' in track_type:
            return 74.0   # ~74 minutos de áudio

        return 0.0

    def _analyze_data_track(self, track_data: Dict) -> Dict:
        """Analisa uma trilha de dados para identificar o sistema de arquivos"""
        # Placeholder para análise de filesystem
        # Na prática, seria necessário implementar leitura de ISO 9660
        return {
            'type': 'iso9660',
            'files_estimated': 150,
            'directories_estimated': 20,
            'largest_file_mb': 25.0
        }

    def _analyze_xa_track(self, track_data: Dict) -> Dict:
        """Analisa uma trilha XA (dados misturados)"""
        return {
            'has_interleaved_audio': True,
            'estimated_sectors': 300000,
            'mixed_ratio': 0.6  # 60% dados, 40% áudio
        }

    def _estimate_audio_duration(self, track_data: Dict) -> float:
        """Estima a duração de uma trilha de áudio em segundos"""
        # Cálculo baseado em tamanho padrão de trilha CD
        size_mb = self._estimate_track_size(track_data)
        # CD Audio: ~10.5 MB por minuto
        return (size_mb / 10.5) * 60

    def _analyze_iso(self, iso_path: Path) -> Dict:
        """Analisa um arquivo ISO"""
        size_mb = iso_path.stat().st_size / (1024 * 1024)

        return {
            'total_tracks': 1,
            'data_tracks': [{
                'number': 1,
                'type': 'MODE1/2048',
                'file': str(iso_path),
                'size_mb': size_mb,
                'contains_game_data': True,
                'filesystem': self._analyze_data_track({'file': str(iso_path)})
            }],
            'audio_tracks': [],
            'estimated_content': {
                'executable_size_mb': size_mb * 0.3,
                'graphics_size_mb': size_mb * 0.4,
                'audio_size_mb': size_mb * 0.2,
                'text_size_mb': size_mb * 0.1
            }
        }

    def _analyze_chd(self, chd_path: Path) -> Dict:
        """Analisa um arquivo CHD (requer chdman)"""
        # Verificar se chdman está disponível
        try:
            result = subprocess.run(['chdman', 'info', '-i', str(chd_path)],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return self._parse_chd_info(result.stdout)
        except FileNotFoundError:
            print("Warning: chdman não encontrado. Análise limitada de CHD.")

        # Fallback para análise básica
        size_mb = chd_path.stat().st_size / (1024 * 1024)
        return {
            'total_tracks': 1,
            'data_tracks': [{
                'number': 1,
                'type': 'CHD_COMPRESSED',
                'file': str(chd_path),
                'size_mb': size_mb,
                'contains_game_data': True
            }],
            'audio_tracks': [],
            'estimated_content': {
                'executable_size_mb': size_mb * 0.3,
                'graphics_size_mb': size_mb * 0.4,
                'audio_size_mb': size_mb * 0.2,
                'text_size_mb': size_mb * 0.1
            }
        }

    def _parse_chd_info(self, chd_info: str) -> Dict:
        """Faz o parse da saída do chdman info"""
        tracks = []
        lines = chd_info.split('\n')

        for line in lines:
            if 'Track' in line and 'Type' in line:
                # Parse da informação de track
                # Formato exemplo: "Track 01: Type MODE1/2048, Frames 12345"
                parts = line.split(',')
                if len(parts) >= 2:
                    track_info = {
                        'number': len(tracks) + 1,
                        'type': 'MODE1/2048',  # Padrão
                        'frames': 0
                    }
                    tracks.append(track_info)

        return {
            'total_tracks': len(tracks) if tracks else 1,
            'data_tracks': tracks if tracks else [{'number': 1, 'type': 'CHD_DATA'}],
            'audio_tracks': [],
            'estimated_content': {}
        }

class SegaCDTextExtractor:
    """Extrator de texto especializado para ROMs do Sega CD"""

    def __init__(self):
        self.encodings = ['shift-jis', 'utf-8', 'latin-1', 'cp932', 'euc-jp']
        self.min_string_length = 4
        self.max_string_length = 200

    def extract_text_from_disc(self, disc_analysis: Dict, rom_path: str) -> Dict:
        """Extrai texto de um disco Sega CD analisado"""
        extracted_texts = {
            'japanese': [],
            'english': [],
            'other': [],
            'dialogue': [],
            'menus': [],
            'system_messages': []
        }

        # Processa cada trilha de dados
        for track in disc_analysis.get('data_tracks', []):
            track_texts = self._extract_from_track(track, rom_path)

            # Categoriza os textos encontrados
            for text_entry in track_texts:
                category = self._categorize_text(text_entry['text'])
                extracted_texts[category].append(text_entry)

        return extracted_texts

    def _extract_from_track(self, track_info: Dict, rom_path: str) -> List[Dict]:
        """Extrai texto de uma trilha específica"""
        texts = []

        # Determina o arquivo a ser processado
        if track_info.get('file'):
            track_file = Path(track_info['file'])
            if not track_file.is_absolute():
                track_file = Path(rom_path).parent / track_file
        else:
            track_file = Path(rom_path)

        if not track_file.exists():
            return texts

        # Lê o arquivo em chunks para eficiência
        chunk_size = 1024 * 1024  # 1MB chunks
        offset = 0

        with open(track_file, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Extrai strings do chunk
                chunk_texts = self._extract_strings_from_data(chunk, offset)
                texts.extend(chunk_texts)

                offset += len(chunk)

        return texts

    def _extract_strings_from_data(self, data: bytes, base_offset: int) -> List[Dict]:
        """Extrai strings de um bloco de dados"""
        strings = []

        # Procura por sequências de texto em diferentes encodings
        for encoding in self.encodings:
            try:
                # Decodifica os dados
                decoded = data.decode(encoding, errors='ignore')

                # Procura por strings válidas
                for match in re.finditer(r'[A-Za-z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\s\.\,\!\?\:\;]{4,}', decoded):
                    text = match.group().strip()

                    if self._is_valid_string(text):
                        strings.append({
                            'text': text,
                            'offset': base_offset + match.start(),
                            'length': len(text),
                            'encoding': encoding,
                            'confidence': self._calculate_confidence(text, encoding)
                        })

            except UnicodeDecodeError:
                continue

        # Remove duplicatas e ordena por confiança
        unique_strings = []
        seen_texts = set()

        for string_info in sorted(strings, key=lambda x: x['confidence'], reverse=True):
            if string_info['text'] not in seen_texts:
                unique_strings.append(string_info)
                seen_texts.add(string_info['text'])

        return unique_strings

    def _is_valid_string(self, text: str) -> bool:
        """Verifica se uma string é válida para tradução"""
        # Filtros básicos
        if len(text) < self.min_string_length or len(text) > self.max_string_length:
            return False

        # Evita strings que são apenas números ou símbolos
        if re.match(r'^[\d\s\.\,\!\?\:\;]+$', text):
            return False

        # Evita strings que parecem ser lixo
        if text.count('�') > len(text) * 0.1:  # Muitos caracteres de substituição
            return False

        return True

    def _calculate_confidence(self, text: str, encoding: str) -> float:
        """Calcula a confiança na extração de uma string"""
        confidence = 0.5  # Base

        # Bonus por encoding apropriado
        if encoding == 'shift-jis' and re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            confidence += 0.3
        elif encoding == 'utf-8' and not re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            confidence += 0.2

        # Bonus por tamanho razoável
        if 8 <= len(text) <= 50:
            confidence += 0.1

        # Bonus por conter palavras comuns
        common_words = ['the', 'and', 'you', 'are', 'です', 'ます', 'する', 'した']
        for word in common_words:
            if word in text.lower():
                confidence += 0.1
                break

        return min(confidence, 1.0)

    def _categorize_text(self, text: str) -> str:
        """Categoriza o texto extraído"""
        text_lower = text.lower()

        # Palavras-chave para diálogo
        dialogue_keywords = ['said', 'says', 'asked', 'replied', 'thought', 'whispered']
        if any(keyword in text_lower for keyword in dialogue_keywords):
            return 'dialogue'

        # Palavras-chave para menus
        menu_keywords = ['start', 'option', 'select', 'exit', 'menu', 'config', 'settings']
        if any(keyword in text_lower for keyword in menu_keywords):
            return 'menus'

        # Palavras-chave para mensagens de sistema
        system_keywords = ['error', 'loading', 'save', 'load', 'press', 'button']
        if any(keyword in text_lower for keyword in system_keywords):
            return 'system_messages'

        # Detecta japonês
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'

        # Detecta inglês
        if re.search(r'^[A-Za-z0-9\s\.\,\!\?\:\;]+$', text):
            return 'english'

        return 'other'

class SegaCDGraphicsExtractor:
    """Extrator de gráficos para ROMs do Sega CD"""

    def __init__(self):
        self.supported_formats = {
            'raw_8bit': self._extract_raw_8bit,
            'raw_16bit': self._extract_raw_16bit,
            'planar_4bit': self._extract_planar_4bit,
            'tile_8x8': self._extract_tile_8x8,
            'sprite_data': self._extract_sprite_data
        }

    def extract_graphics_from_disc(self, disc_analysis: Dict, rom_path: str,
                                 output_dir: str) -> Dict:
        """Extrai gráficos de um disco Sega CD"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted_graphics = {
            'palettes': [],
            'sprites': [],
            'backgrounds': [],
            'textures': [],
            'fonts': []
        }

        # Processa cada trilha de dados
        for track in disc_analysis.get('data_tracks', []):
            track_graphics = self._extract_graphics_from_track(track, rom_path, output_path)

            # Combina os resultados
            for category, items in track_graphics.items():
                extracted_graphics[category].extend(items)

        return extracted_graphics

    def _extract_graphics_from_track(self, track_info: Dict, rom_path: str,
                                   output_path: Path) -> Dict:
        """Extrai gráficos de uma trilha específica"""
        graphics = {
            'palettes': [],
            'sprites': [],
            'backgrounds': [],
            'textures': [],
            'fonts': []
        }

        # Determina o arquivo a ser processado
        if track_info.get('file'):
            track_file = Path(track_info['file'])
            if not track_file.is_absolute():
                track_file = Path(rom_path).parent / track_file
        else:
            track_file = Path(rom_path)

        if not track_file.exists():
            return graphics

        # Lê o arquivo
        with open(track_file, 'rb') as f:
            data = f.read()

        # Busca por padrões gráficos
        graphics['palettes'] = self._find_palettes(data)
        graphics['sprites'] = self._find_sprites(data, output_path)
        graphics['backgrounds'] = self._find_backgrounds(data, output_path)
        graphics['textures'] = self._find_textures(data, output_path)
        graphics['fonts'] = self._find_fonts(data, output_path)

        return graphics

    def _find_palettes(self, data: bytes) -> List[Dict]:
        """Busca por paletas de cores nos dados"""
        palettes = []

        # Procura por padrões de paleta (16 cores, 16-bit cada)
        for offset in range(0, len(data) - 32, 2):
            # Verifica se parece com uma paleta
            palette_data = data[offset:offset + 32]

            if self._is_likely_palette(palette_data):
                colors = []
                for i in range(0, 32, 2):
                    # Formato RGB565 do Sega CD
                    color_word = struct.unpack('<H', palette_data[i:i+2])[0]
                    r = (color_word >> 11) & 0x1F
                    g = (color_word >> 6) & 0x1F
                    b = color_word & 0x1F

                    # Converte para RGB888
                    r = (r << 3) | (r >> 2)
                    g = (g << 3) | (g >> 2)
                    b = (b << 3) | (b >> 2)

                    colors.append((r, g, b))

                palettes.append({
                    'offset': offset,
                    'colors': colors,
                    'format': 'RGB565'
                })

        return palettes

    def _is_likely_palette(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser uma paleta"""
        if len(data) < 32:
            return False

        # Verifica se há variação suficiente nas cores
        unique_colors = set()
        for i in range(0, 32, 2):
            color = struct.unpack('<H', data[i:i+2])[0]
            unique_colors.add(color)

        # Deve ter pelo menos 8 cores diferentes
        return len(unique_colors) >= 8

    def _find_sprites(self, data: bytes, output_path: Path) -> List[Dict]:
        """Busca por dados de sprite"""
        sprites = []

        # Procura por padrões de sprite 8x8 e 16x16
        for size in [8, 16, 32]:
            sprite_size = size * size // 2  # 4 bits por pixel

            for offset in range(0, len(data) - sprite_size, sprite_size):
                sprite_data = data[offset:offset + sprite_size]

                if self._is_likely_sprite(sprite_data):
                    # Tenta converter para imagem
                    try:
                        image = self._sprite_to_image(sprite_data, size)

                        # Salva a imagem
                        sprite_filename = f"sprite_{offset:08x}_{size}x{size}.png"
                        image_path = output_path / sprite_filename
                        image.save(image_path)

                        sprites.append({
                            'offset': offset,
                            'size': (size, size),
                            'file': sprite_filename,
                            'format': '4bpp_planar'
                        })
                    except Exception:
                        continue

        return sprites

    def _is_likely_sprite(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser um sprite"""
        if len(data) < 32:
            return False

        # Verifica se há padrões que sugerem dados gráficos
        # Sprites tendem a ter mais variação que dados aleatórios
        byte_frequency = {}
        for byte in data:
            byte_frequency[byte] = byte_frequency.get(byte, 0) + 1

        # Se há muitos bytes diferentes, provavelmente é gráfico
        return len(byte_frequency) > len(data) * 0.3

    def _sprite_to_image(self, sprite_data: bytes, size: int) -> Image.Image:
        """Converte dados de sprite para imagem PIL"""
        # Cria uma imagem em escala de cinza
        pixels = []

        for byte in sprite_data:
            # Cada byte contém 2 pixels de 4 bits
            pixel1 = (byte >> 4) & 0x0F
            pixel2 = byte & 0x0F

            # Converte para escala de cinza (0-255)
            pixels.append(pixel1 * 17)  # 17 = 255/15
            pixels.append(pixel2 * 17)

        # Cria a imagem
        image = Image.new('L', (size, size))
        image.putdata(pixels[:size * size])

        return image

    def _find_backgrounds(self, data: bytes, output_path: Path) -> List[Dict]:
        """Busca por dados de background/cenário"""
        backgrounds = []

        # Procura por padrões maiores que poderiam ser backgrounds
        for width, height in [(256, 256), (320, 240), (512, 256)]:
            bg_size = width * height // 2  # 4 bits por pixel

            for offset in range(0, len(data) - bg_size, bg_size):
                bg_data = data[offset:offset + bg_size]

                if self._is_likely_background(bg_data):
                    try:
                        image = self._background_to_image(bg_data, width, height)

                        bg_filename = f"background_{offset:08x}_{width}x{height}.png"
                        image_path = output_path / bg_filename
                        image.save(image_path)

                        backgrounds.append({
                            'offset': offset,
                            'size': (width, height),
                            'file': bg_filename,
                            'format': '4bpp_planar'
                        })
                    except Exception:
                        continue

        return backgrounds

    def _is_likely_background(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser um background"""
        if len(data) < 1024:
            return False

        # Backgrounds tendem a ter padrões repetitivos
        # Verifica se há sequências repetidas
        chunk_size = 64
       chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

        # Conta chunks únicos
        unique_chunks = len(set(chunks))
        total_chunks = len(chunks)

        # Se há muita repetição, provavelmente é background
        return unique_chunks < total_chunks * 0.7

    def _background_to_image(self, bg_data: bytes, width: int, height: int) -> Image.Image:
        """Converte dados de background para imagem PIL"""
        pixels = []

        for byte in bg_data:
            pixel1 = (byte >> 4) & 0x0F
            pixel2 = byte & 0x0F

            pixels.append(pixel1 * 17)
            pixels.append(pixel2 * 17)

        image = Image.new('L', (width, height))
        image.putdata(pixels[:width * height])

        return image

    def _find_textures(self, data: bytes, output_path: Path) -> List[Dict]:
        """Busca por texturas"""
        textures = []

        # Procura por padrões de textura (64x64, 128x128)
        for size in [64, 128]:
            texture_size = size * size // 2

            for offset in range(0, len(data) - texture_size, texture_size):
                texture_data = data[offset:offset + texture_size]

                if self._is_likely_texture(texture_data):
                    try:
                        image = self._texture_to_image(texture_data, size)

                        texture_filename = f"texture_{offset:08x}_{size}x{size}.png"
                        image_path = output_path / texture_filename
                        image.save(image_path)

                        textures.append({
                            'offset': offset,
                            'size': (size, size),
                            'file': texture_filename,
                            'format': '4bpp_planar'
                        })
                    except Exception:
                        continue

        return textures

    def _is_likely_texture(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser uma textura"""
        if len(data) < 512:
            return False

        # Texturas têm características específicas
        # Verifica variação gradual de pixels
        variations = 0
        for i in range(len(data) - 1):
            diff = abs(data[i] - data[i + 1])
            if diff < 4:  # Variação gradual
                variations += 1

        return variations > len(data) * 0.4

    def _texture_to_image(self, texture_data: bytes, size: int) -> Image.Image:
        """Converte dados de textura para imagem PIL"""
        pixels = []

        for byte in texture_data:
            pixel1 = (byte >> 4) & 0x0F
            pixel2 = byte & 0x0F

            pixels.append(pixel1 * 17)
            pixels.append(pixel2 * 17)

        image = Image.new('L', (size, size))
        image.putdata(pixels[:size * size])

        return image

    def _find_fonts(self, data: bytes, output_path: Path) -> List[Dict]:
        """Busca por fontes/caracteres"""
        fonts = []

        # Procura por padrões de fonte (8x8, 16x16 caracteres)
        for char_size in [8, 16]:
            font_char_size = char_size * char_size // 8  # 1 bit por pixel

            # Procura por conjuntos de caracteres
            for offset in range(0, len(data) - font_char_size * 95, font_char_size):
                font_data = data[offset:offset + font_char_size * 95]  # ASCII printable

                if self._is_likely_font(font_data, char_size):
                    try:
                        # Extrai caracteres individuais
                        font_chars = []
                        for i in range(95):
                            char_offset = i * font_char_size
                            char_data = font_data[char_offset:char_offset + font_char_size]

                            char_image = self._font_char_to_image(char_data, char_size)
                            char_filename = f"font_{offset:08x}_char_{i:02d}.png"
                            char_path = output_path / char_filename
                            char_image.save(char_path)

                            font_chars.append({
                                'ascii_code': i + 32,  # Começa em espaço
                                'file': char_filename
                            })

                        fonts.append({
                            'offset': offset,
                            'char_size': (char_size, char_size),
                            'char_count': 95,
                            'characters': font_chars,
                            'format': '1bpp_bitmap'
                        })
                    except Exception:
                        continue

        return fonts

    def _is_likely_font(self, data: bytes, char_size: int) -> bool:
        """Verifica se os dados parecem ser uma fonte"""
        if len(data) < char_size * char_size // 8:
            return False

        # Fontes têm características específicas
        # Verifica se há padrões que sugerem caracteres
        char_data_size = char_size * char_size // 8

        # Verifica alguns "caracteres" para ver se fazem sentido
        valid_chars = 0
        for i in range(min(10, len(data) // char_data_size)):
            char_data = data[i * char_data_size:(i + 1) * char_data_size]

            # Caracteres tendem a ter bordas (pixels ligados e desligados)
            transitions = 0
            for byte in char_data:
                for bit in range(7):
                    if ((byte >> bit) & 1) != ((byte >> (bit + 1)) & 1):
                        transitions += 1

            if transitions > 2:  # Pelo menos algumas transições
                valid_chars += 1

        return valid_chars > 3

    def _font_char_to_image(self, char_data: bytes, char_size: int) -> Image.Image:
        """Converte dados de caractere para imagem PIL"""
        pixels = []

        for byte in char_data:
            for bit in range(8):
                pixel = (byte >> (7 - bit)) & 1
                pixels.append(255 if pixel else 0)

        image = Image.new('L', (char_size, char_size))
        image.putdata(pixels[:char_size * char_size])

        return image

    # Métodos auxiliares para formatos específicos
    def _extract_raw_8bit(self, data: bytes, width: int, height: int) -> Image.Image:
        """Extrai gráfico em formato raw 8-bit"""
        size = width * height
        if len(data) < size:
            raise ValueError("Dados insuficientes para o tamanho especificado")

        image = Image.new('L', (width, height))
        image.putdata(data[:size])
        return image

    def _extract_raw_16bit(self, data: bytes, width: int, height: int) -> Image.Image:
        """Extrai gráfico em formato raw 16-bit (RGB565)"""
        size = width * height * 2
        if len(data) < size:
            raise ValueError("Dados insuficientes para o tamanho especificado")

        pixels = []
        for i in range(0, size, 2):
            color_word = struct.unpack('<H', data[i:i+2])[0]
            r = (color_word >> 11) & 0x1F
            g = (color_word >> 6) & 0x1F
            b = color_word & 0x1F

            # Converte para RGB888
            r = (r << 3) | (r >> 2)
            g = (g << 3) | (g >> 2)
            b = (b << 3) | (b >> 2)

            pixels.append((r, g, b))

        image = Image.new('RGB', (width, height))
        image.putdata(pixels)
        return image

    def _extract_planar_4bit(self, data: bytes, width: int, height: int) -> Image.Image:
        """Extrai gráfico em formato planar 4-bit"""
        size = width * height // 2
        if len(data) < size:
            raise ValueError("Dados insuficientes para o tamanho especificado")

        pixels = []
        for byte in data[:size]:
            pixel1 = (byte >> 4) & 0x0F
            pixel2 = byte & 0x0F

            pixels.append(pixel1 * 17)
            pixels.append(pixel2 * 17)

        image = Image.new('L', (width, height))
        image.putdata(pixels[:width * height])
        return image

    def _extract_tile_8x8(self, data: bytes, tile_count: int) -> List[Image.Image]:
        """Extrai tiles 8x8"""
        tiles = []
        tile_size = 32  # 8x8 pixels, 4 bits por pixel

        for i in range(tile_count):
            offset = i * tile_size
            if offset + tile_size > len(data):
                break

            tile_data = data[offset:offset + tile_size]
            tile_image = self._sprite_to_image(tile_data, 8)
            tiles.append(tile_image)

        return tiles

    def _extract_sprite_data(self, data: bytes, sprite_width: int, sprite_height: int) -> Image.Image:
        """Extrai dados de sprite com dimensões específicas"""
        return self._extract_planar_4bit(data, sprite_width, sprite_height)

class SegaCDAudioExtractor:
    """Extrator de áudio para ROMs do Sega CD"""

    def __init__(self):
        self.supported_formats = ['cdda', 'pcm', 'xa_adpcm']
        self.temp_dir = None

    def extract_audio_from_disc(self, disc_analysis: Dict, rom_path: str,
                              output_dir: str) -> Dict:
        """Extrai áudio de um disco Sega CD"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted_audio = {
            'cd_tracks': [],
            'sfx_samples': [],
            'voice_samples': [],
            'music_data': []
        }

        # Extrai trilhas de áudio CD
        for track in disc_analysis.get('audio_tracks', []):
            cd_track = self._extract_cd_track(track, rom_path, output_path)
            if cd_track:
                extracted_audio['cd_tracks'].append(cd_track)

        # Extrai samples de áudio das trilhas de dados
        for track in disc_analysis.get('data_tracks', []):
            track_audio = self._extract_audio_from_data_track(track, rom_path, output_path)

            extracted_audio['sfx_samples'].extend(track_audio.get('sfx_samples', []))
            extracted_audio['voice_samples'].extend(track_audio.get('voice_samples', []))
            extracted_audio['music_data'].extend(track_audio.get('music_data', []))

        return extracted_audio

    def _extract_cd_track(self, track_info: Dict, rom_path: str, output_path: Path) -> Optional[Dict]:
        """Extrai uma trilha de áudio CD"""
        try:
            # Determina o arquivo fonte
            if track_info.get('file'):
                source_file = Path(track_info['file'])
                if not source_file.is_absolute():
                    source_file = Path(rom_path).parent / source_file
            else:
                source_file = Path(rom_path)

            if not source_file.exists():
                return None

            # Nome do arquivo de saída
            track_num = track_info.get('number', 1)
            output_filename = f"cdda_track_{track_num:02d}.wav"
            output_file = output_path / output_filename

            # Extrai usando ferramentas externas se disponíveis
            if self._extract_with_external_tools(source_file, output_file, track_info):
                return {
                    'track_number': track_num,
                    'file': output_filename,
                    'duration_seconds': track_info.get('duration_seconds', 0),
                    'format': 'CD Audio',
                    'sample_rate': 44100,
                    'channels': 2,
                    'bit_depth': 16
                }
            else:
                # Fallback para extração manual
                return self._extract_cd_track_manual(track_info, source_file, output_file)

        except Exception as e:
            print(f"Erro ao extrair trilha de áudio {track_info.get('number', '?')}: {e}")
            return None

    def _extract_with_external_tools(self, source_file: Path, output_file: Path,
                                   track_info: Dict) -> bool:
        """Tenta extrair áudio usando ferramentas externas"""
        # Tenta usar bchunk para BIN/CUE
        if source_file.suffix.lower() in ['.bin', '.cue']:
            try:
                cue_file = source_file.with_suffix('.cue')
                if cue_file.exists():
                    result = subprocess.run([
                        'bchunk', str(source_file), str(cue_file),
                        str(output_file.with_suffix(''))
                    ], capture_output=True, text=True)

                    if result.returncode == 0:
                        return True
            except FileNotFoundError:
                pass

        # Tenta usar cdrdao
        try:
            result = subprocess.run([
                'cdrdao', 'read-cd', '--device', str(source_file),
                '--datafile', str(output_file.with_suffix('.bin')),
                str(output_file.with_suffix('.toc'))
            ], capture_output=True, text=True)

            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass

        return False

    def _extract_cd_track_manual(self, track_info: Dict, source_file: Path,
                               output_file: Path) -> Optional[Dict]:
        """Extração manual de trilha de áudio CD"""
        try:
            # Lê dados da trilha
            with open(source_file, 'rb') as f:
                # Para trilhas de áudio CD, os dados são raw PCM
                # 2352 bytes por setor, sendo 2048 dados + 304 headers/ECC

                # Procura pelo início da trilha de áudio
                # (implementação simplificada)
                audio_data = f.read()

                # Converte para WAV
                self._save_as_wav(audio_data, output_file)

                return {
                    'track_number': track_info.get('number', 1),
                    'file': output_file.name,
                    'duration_seconds': len(audio_data) / (44100 * 2 * 2),  # Estimativa
                    'format': 'CD Audio (extracted)',
                    'sample_rate': 44100,
                    'channels': 2,
                    'bit_depth': 16
                }

        except Exception as e:
            print(f"Erro na extração manual: {e}")
            return None

    def _save_as_wav(self, audio_data: bytes, output_file: Path):
        """Salva dados de áudio como arquivo WAV"""
        with open(output_file, 'wb') as f:
            # Header WAV
            f.write(b'RIFF')
            f.write(struct.pack('<I', len(audio_data) + 36))
            f.write(b'WAVE')

            # Chunk fmt
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Tamanho do chunk
            f.write(struct.pack('<H', 1))   # Formato PCM
            f.write(struct.pack('<H', 2))   # Canais
            f.write(struct.pack('<I', 44100))  # Sample rate
            f.write(struct.pack('<I', 44100 * 2 * 2))  # Byte rate
            f.write(struct.pack('<H', 4))   # Block align
            f.write(struct.pack('<H', 16))  # Bits per sample

            # Chunk data
            f.write(b'data')
            f.write(struct.pack('<I', len(audio_data)))
            f.write(audio_data)

    def _extract_audio_from_data_track(self, track_info: Dict, rom_path: str,
                                     output_path: Path) -> Dict:
        """Extrai samples de áudio de uma trilha de dados"""
        audio_data = {
            'sfx_samples': [],
            'voice_samples': [],
            'music_data': []
        }

        # Determina o arquivo da trilha
        if track_info.get('file'):
            track_file = Path(track_info['file'])
            if not track_file.is_absolute():
                track_file = Path(rom_path).parent / track_file
        else:
            track_file = Path(rom_path)

        if not track_file.exists():
            return audio_data

        # Lê o arquivo em chunks
        with open(track_file, 'rb') as f:
            data = f.read()

        # Procura por padrões de áudio
        audio_data['sfx_samples'] = self._find_sfx_samples(data, output_path)
        audio_data['voice_samples'] = self._find_voice_samples(data, output_path)
        audio_data['music_data'] = self._find_music_data(data, output_path)

        return audio_data

    def _find_sfx_samples(self, data: bytes, output_path: Path) -> List[Dict]:
        """Procura por samples de efeitos sonoros"""
        sfx_samples = []

        # Procura por padrões de PCM de diferentes tamanhos
        for sample_size in [1024, 2048, 4096, 8192]:
            for offset in range(0, len(data) - sample_size, sample_size):
                sample_data = data[offset:offset + sample_size]

                if self._is_likely_audio_sample(sample_data):
                    try:
                        # Salva como WAV
                        sample_filename = f"sfx_sample_{offset:08x}.wav"
                        sample_path = output_path / sample_filename
                        self._save_as_wav(sample_data, sample_path)

                        sfx_samples.append({
                            'offset': offset,
                            'size': sample_size,
                            'file': sample_filename,
                            'estimated_duration': sample_size / (22050 * 2),  # Estimativa
                            'format': 'PCM'
                        })
                    except Exception:
                        continue

        return sfx_samples

    def _is_likely_audio_sample(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser um sample de áudio"""
        if len(data) < 512:
            return False

        # Áudio PCM tem características específicas
        # Verifica se há variação suficiente (não é silêncio)
        variations = 0
        for i in range(len(data) - 1):
            if abs(data[i] - data[i + 1]) > 5:
                variations += 1

        # Verifica se não há valores extremos demais (ruído)
        extreme_values = sum(1 for b in data if b < 10 or b > 245)

        return (variations > len(data) * 0.1 and
                extreme_values < len(data) * 0.05)

    def _find_voice_samples(self, data: bytes, output_path: Path) -> List[Dict]:
        """Procura por samples de voz"""
        voice_samples = []

        # Samples de voz tendem a ser maiores
        for sample_size in [8192, 16384, 32768]:
            for offset in range(0, len(data) - sample_size, sample_size):
                sample_data = data[offset:offset + sample_size]

                if self._is_likely_voice_sample(sample_data):
                    try:
                        sample_filename = f"voice_sample_{offset:08x}.wav"
                        sample_path = output_path / sample_filename
                        self._save_as_wav(sample_data, sample_path)

                        voice_samples.append({
                            'offset': offset,
                            'size': sample_size,
                            'file': sample_filename,
                            'estimated_duration': sample_size / (22050 * 2),
                            'format': 'PCM'
                        })
                    except Exception:
                        continue

        return voice_samples

    def _is_likely_voice_sample(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser um sample de voz"""
        if len(data) < 8192:
            return False

        # Voz tem padrões específicos
        # Verifica se há variação gradual (formantes)
        gradual_changes = 0
        for i in range(len(data) - 10):
            avg_before = sum(data[i:i+5]) / 5
            avg_after = sum(data[i+5:i+10]) / 5

            if abs(avg_before - avg_after) < 20:  # Mudança gradual
                gradual_changes += 1

        return gradual_changes > len(data) * 0.3

    def _find_music_data(self, data: bytes, output_path: Path) -> List[Dict]:
        """Procura por dados de música/sequências"""
        music_data = []

        # Procura por padrões que podem ser dados de música
        # (MIDI, sequências, etc.)
        for offset in range(0, len(data) - 1024, 1024):
            chunk = data[offset:offset + 1024]

            if self._is_likely_music_sequence(chunk):
                try:
                    music_filename = f"music_data_{offset:08x}.bin"
                    music_path = output_path / music_filename

                    with open(music_path, 'wb') as f:
                        f.write(chunk)

                    music_data.append({
                        'offset': offset,
                        'size': 1024,
                        'file': music_filename,
                        'format': 'Unknown sequence',
                        'type': 'music_sequence'
                    })
                except Exception:
                    continue

        return music_data

    def _is_likely_music_sequence(self, data: bytes) -> bool:
        """Verifica se os dados parecem ser uma sequência musical"""
        if len(data) < 256:
            return False

        # Sequências musicais têm padrões específicos
        # Verifica se há valores que parecem ser notas/comandos
        note_like_values = 0
        for byte in data:
            if 0x20 <= byte <= 0x7F:  # Faixa típica de notas MIDI
                note_like_values += 1

        return note_like_values > len(data) * 0.4

class SegaCDTranslationManager:
    """Gerenciador principal de traduções para Sega CD"""

    def __init__(self, rom_path: str, output_dir: str):
        self.rom_path = Path(rom_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Componentes da engine
        self.rom_analyzer = SegaCDROMAnalyzer()
        self.text_extractor = SegaCDTextExtractor()
        self.graphics_extractor = SegaCDGraphicsExtractor()
        self.audio_extractor = SegaCDAudioExtractor()

        # Dados da análise
        self.disc_analysis = None
        self.extracted_texts = None
        self.extracted_graphics = None
        self.extracted_audio = None

        # Configuração
        self.config = {
            'extract_text': True,
            'extract_graphics': True,
            'extract_audio': True,
            'create_backups': True,
            'output_format': 'json'
        }

    def analyze_rom(self) -> Dict:
        """Analisa completamente a ROM do Sega CD"""
        print(f"Iniciando análise de: {self.rom_path}")

        # Análise da estrutura do disco
        print("Analisando estrutura do disco...")
        self.disc_analysis = self.rom_analyzer.analyze_disc(str(self.rom_path))

        # Extração de texto
        if self.config['extract_text']:
            print("Extraindo textos...")
            self.extracted_texts = self.text_extractor.extract_text_from_disc(
                self.disc_analysis, str(self.rom_path)
            )

        # Extração de gráficos
        if self.config['extract_graphics']:
            print("Extraindo gráficos...")
            graphics_dir = self.output_dir / "graphics"
            self.extracted_graphics = self.graphics_extractor.extract_graphics_from_disc(
                self.disc_analysis, str(self.rom_path), str(graphics_dir)
            )

        # Extração de áudio
        if self.config['extract_audio']:
            print("Extraindo áudio...")
            audio_dir = self.output_dir / "audio"
            self.extracted_audio = self.audio_extractor.extract_audio_from_disc(
                self.disc_analysis, str(self.rom_path), str(audio_dir)
            )

        # Compila resultado final
        analysis_result = {
            'rom_info': {
                'path': str(self.rom_path),
                'size_mb': self.rom_path.stat().st_size / (1024 * 1024),
                'format': self.disc_analysis.get('format', 'unknown')
            },
            'disc_structure': self.disc_analysis,
            'extracted_texts': self.extracted_texts,
            'extracted_graphics': self.extracted_graphics,
            'extracted_audio': self.extracted_audio,
            'analysis_timestamp': datetime.now().isoformat()
        }

        # Salva resultado
        self._save_analysis_result(analysis_result)

        return analysis_result

    def _save_analysis_result(self, result: Dict):
        """Salva o resultado da análise"""
        if self.config['output_format'] == 'json':
            output_file = self.output_dir / "analysis_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        elif self.config['output_format'] == 'yaml':
            output_file = self.output_dir / "analysis_result.yaml"
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)

        print(f"Resultado salvo em: {output_file}")

    def create_translation_template(self) -> Dict:
        """Cria um template para tradução"""
        if not self.extracted_texts:
            raise ValueError("Textos não foram extraídos. Execute analyze_rom() primeiro.")

        template = {
            'metadata': {
                'rom_name': self.rom_path.stem,
                'original_language': 'japanese',
                'target_language': 'portuguese',
                'translator': '',
                'version': '1.0',
                'date_created': datetime.now().isoformat()
            },
            'translation_entries': []
        }

        # Converte textos extraídos em entradas de tradução
        entry_id = 1
        for category, texts in self.extracted_texts.items():
            for text_info in texts:
                template['translation_entries'].append({
                    'id': entry_id,
                    'category': category,
                    'original_text': text_info['text'],
                    'translated_text': '',
                    'context': f"Offset: 0x{text_info['offset']:08x}",
                    'encoding': text_info['encoding'],
                    'confidence': text_info['confidence'],
                    'notes': '',
                    'status': 'untranslated'
                })
                entry_id += 1

        # Salva template
        template_file = self.output_dir / "translation_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        print(f"Template de tradução criado: {template_file}")
        return template

    def apply_translation(self, translation_file: str, output_rom: str):
        """Aplica uma tradução à ROM"""
        print(f"Aplicando tradução: {translation_file}")

        # Carrega o arquivo de tradução
        with open(translation_file, 'r', encoding='utf-8') as f:
            translation_data = json.load(f)

      # Cria backup da ROM original
        if self.config['create_backups']:
            backup_path = self.output_dir / f"{self.rom_path.stem}_backup{self.rom_path.suffix}"
            shutil.copy2(self.rom_path, backup_path)
            self.logger.info(f"Backup criado: {backup_path}")

        # Aplica as traduções
        self.logger.info("Aplicando traduções...")

        # Ordena patches por offset para aplicação sequencial
        patches_ordenados = sorted(self.patches, key=lambda p: p.offset)

        aplicados = 0
        for patch in patches_ordenados:
            try:
                # Verifica se o offset é válido
                if patch.offset >= len(self.rom.data):
                    self.logger.warning(f"Offset inválido: {patch.offset:08X}")
                    continue

                # Aplica o patch
                texto_bytes = patch.texto_traduzido.encode(self.config['encoding'])

                # Verifica se o texto cabe no espaço disponível
                if len(texto_bytes) > len(patch.texto_original.encode(self.config['encoding'])):
                    self.logger.warning(f"Texto muito longo no offset {patch.offset:08X}: '{patch.texto_traduzido}'")
                    continue

                # Aplica o patch na ROM
                self.rom.data[patch.offset:patch.offset + len(texto_bytes)] = texto_bytes

                # Preenche o espaço restante com zeros se necessário
                espaco_restante = len(patch.texto_original.encode(self.config['encoding'])) - len(texto_bytes)
                if espaco_restante > 0:
                    self.rom.data[patch.offset + len(texto_bytes):patch.offset + len(patch.texto_original.encode(self.config['encoding']))] = b'\x00' * espaco_restante

                aplicados += 1

            except Exception as e:
                self.logger.error(f"Erro ao aplicar patch no offset {patch.offset:08X}: {str(e)}")
                continue

        self.logger.info(f"Patches aplicados: {aplicados}/{len(patches_ordenados)}")

        # Salva a ROM traduzida
        output_path = self.output_dir / f"{self.rom_path.stem}_traduzida{self.rom_path.suffix}"

        try:
            with open(output_path, 'wb') as f:
                f.write(self.rom.data)

            self.logger.info(f"ROM traduzida salva: {output_path}")

            # Gera relatório
            self._gerar_relatorio(output_path, aplicados, len(patches_ordenados))

            return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar ROM traduzida: {str(e)}")
            return False

    def _gerar_relatorio(self, output_path: Path, aplicados: int, total: int):
        """Gera relatório detalhado da tradução"""
        relatorio_path = self.output_dir / f"{self.rom_path.stem}_relatorio.txt"

        try:
            with open(relatorio_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("RELATÓRIO DE TRADUÇÃO - SEGA CD TRANSLATION ENGINE\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"ROM Original: {self.rom_path.name}\n")
                f.write(f"ROM Traduzida: {output_path.name}\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

                f.write(f"Patches Aplicados: {aplicados}/{total}\n")
                f.write(f"Taxa de Sucesso: {(aplicados/total)*100:.1f}%\n\n")

                f.write("CONFIGURAÇÕES UTILIZADAS:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.config.items():
                    f.write(f"{key}: {value}\n")

                f.write("\nDETALHES DOS PATCHES:\n")
                f.write("-" * 30 + "\n")

                for i, patch in enumerate(self.patches, 1):
                    f.write(f"\nPatch {i:03d}:\n")
                    f.write(f"  Offset: {patch.offset:08X}\n")
                    f.write(f"  Original: {patch.texto_original[:50]}{'...' if len(patch.texto_original) > 50 else ''}\n")
                    f.write(f"  Traduzido: {patch.texto_traduzido[:50]}{'...' if len(patch.texto_traduzido) > 50 else ''}\n")
                    f.write(f"  Status: {'✓ Aplicado' if patch.aplicado else '✗ Falhou'}\n")

            self.logger.info(f"Relatório gerado: {relatorio_path}")

        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {str(e)}")

class AITranslator:
    """Sistema de tradução usando IA"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.cache = {}

        # Inicializa o cliente da IA baseado na configuração
        self._init_ai_client()

    def _init_ai_client(self):
        """Inicializa o cliente da IA"""
        try:
            if self.config['ai_provider'] == 'openai':
                import openai
                self.client = openai.OpenAI(api_key=self.config['api_key'])
            elif self.config['ai_provider'] == 'anthropic':
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config['api_key'])
            else:
                raise ValueError(f"Provedor de IA não suportado: {self.config['ai_provider']}")

            self.logger.info(f"Cliente da IA inicializado: {self.config['ai_provider']}")

        except Exception as e:
            self.logger.error(f"Erro ao inicializar cliente da IA: {str(e)}")
            raise

    def traduzir_batch(self, textos: List[str]) -> List[str]:
        """Traduz uma lista de textos em lote"""
        if not textos:
            return []

        # Filtra textos já traduzidos do cache
        textos_para_traduzir = []
        resultados = {}

        for i, texto in enumerate(textos):
            if texto in self.cache:
                resultados[i] = self.cache[texto]
            else:
                textos_para_traduzir.append((i, texto))

        if not textos_para_traduzir:
            return [resultados[i] for i in range(len(textos))]

        # Traduz textos não encontrados no cache
        try:
            batch_text = "\n---SEPARATOR---\n".join([texto for _, texto in textos_para_traduzir])

            traducao = self._traduzir_texto(batch_text)

            # Divide as traduções
            traducoes = traducao.split("---SEPARATOR---")

            # Atualiza cache e resultados
            for (i, texto_original), traducao in zip(textos_para_traduzir, traducoes):
                traducao_limpa = traducao.strip()
                self.cache[texto_original] = traducao_limpa
                resultados[i] = traducao_limpa

            return [resultados[i] for i in range(len(textos))]

        except Exception as e:
            self.logger.error(f"Erro na tradução em lote: {str(e)}")
            # Retorna textos originais em caso de erro
            return textos

    def _traduzir_texto(self, texto: str) -> str:
        """Traduz um texto usando a IA configurada"""
        try:
            prompt = self._criar_prompt(texto)

            if self.config['ai_provider'] == 'openai':
                response = self.client.chat.completions.create(
                    model=self.config['model'],
                    messages=[
                        {"role": "system", "content": self.config['system_prompt']},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature']
                )
                return response.choices[0].message.content.strip()

            elif self.config['ai_provider'] == 'anthropic':
                response = self.client.messages.create(
                    model=self.config['model'],
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature'],
                    system=self.config['system_prompt'],
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()

        except Exception as e:
            self.logger.error(f"Erro na tradução: {str(e)}")
            return texto

    def _criar_prompt(self, texto: str) -> str:
        """Cria o prompt para tradução"""
        return f"""
Traduza o seguinte texto de jogo do Sega CD do {self.config['idioma_origem']} para {self.config['idioma_destino']}.

INSTRUÇÕES IMPORTANTES:
- Mantenha o contexto de jogos retrô dos anos 90
- Preserve códigos especiais, números e símbolos
- Mantenha o comprimento similar ao original
- Use linguagem apropriada para games da época
- Se houver separadores ---SEPARATOR---, mantenha-os exatamente

TEXTO PARA TRADUZIR:
{texto}

TRADUÇÃO:"""

class ConfigManager:
    """Gerenciador de configurações"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        self.load_config()

    def _load_default_config(self) -> Dict:
        """Carrega configuração padrão"""
        return {
            # Configurações gerais
            'create_backups': True,
            'encoding': 'shift_jis',
            'output_dir': 'output',
            'log_level': 'INFO',

            # Configurações de tradução
            'ai_provider': 'openai',  # 'openai' ou 'anthropic'
            'model': 'gpt-4',
            'api_key': '',
            'idioma_origem': 'japonês',
            'idioma_destino': 'português brasileiro',
            'max_tokens': 2000,
            'temperature': 0.3,
            'system_prompt': 'Você é um especialista em tradução de jogos retrô.',

            # Configurações de extração
            'min_text_length': 3,
            'max_text_length': 200,
            'skip_binary_data': True,
            'extract_patterns': [
                # Padrões comuns em jogos Sega CD
                r'[\x20-\x7E]+',  # ASCII printable
                r'[\x81-\x9F\xE0-\xEF][\x40-\x7E\x80-\xFC]+',  # Shift-JIS
            ],

            # Configurações de patches
            'patch_format': 'ips',  # 'ips', 'ups', 'bps'
            'verify_patches': True,
            'auto_apply': False,
        }

    def load_config(self):
        """Carrega configuração do arquivo"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)

                print(f"Configuração carregada: {self.config_path}")
            else:
                self.save_config()
                print(f"Arquivo de configuração criado: {self.config_path}")

        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            print("Usando configuração padrão...")

    def save_config(self):
        """Salva configuração no arquivo"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")

    def get(self, key: str, default=None):
        """Obtém valor da configuração"""
        return self.config.get(key, default)

    def set(self, key: str, value):
        """Define valor na configuração"""
        self.config[key] = value
        self.save_config()

class SegaCDTranslationEngine:
    """Engine principal do sistema de tradução"""

    def __init__(self, config_path: str = "config.json"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Configura logging
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('translation_engine.log'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

        # Inicializa componentes
        self.translator = AITranslator(self.config) if self.config['api_key'] else None
        self.current_rom = None

        self.logger.info("Sega CD Translation Engine iniciado")

    def processar_rom(self, rom_path: str, output_dir: str = None) -> bool:
        """Processa uma ROM completa"""
        try:
            rom_path = Path(rom_path)

            if not rom_path.exists():
                self.logger.error(f"ROM não encontrada: {rom_path}")
                return False

            # Define diretório de saída
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = Path(self.config['output_dir'])

            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Processando ROM: {rom_path}")

            # Extrai textos
            extractor = TextExtractor(self.config)
            textos = extractor.extrair_textos(rom_path)

            if not textos:
                self.logger.warning("Nenhum texto encontrado na ROM")
                return False

            # Traduz textos se tradutor disponível
            if self.translator:
                self.logger.info("Iniciando tradução com IA...")
                textos_originais = [t.texto for t in textos]
                textos_traduzidos = self.translator.traduzir_batch(textos_originais)

                # Atualiza objetos de texto com traduções
                for texto_obj, traducao in zip(textos, textos_traduzidos):
                    texto_obj.texto_traduzido = traducao
            else:
                self.logger.warning("Tradutor não disponível - pulando tradução")
                for texto_obj in textos:
                    texto_obj.texto_traduzido = texto_obj.texto

            # Converte para patches
            patches = [TextPatch(t.offset, t.texto, t.texto_traduzido) for t in textos]

            # Aplica patches
            patcher = ROMPatcher(self.config)
            sucesso = patcher.aplicar_patches(rom_path, patches, output_dir)

            if sucesso:
                self.logger.info("ROM processada com sucesso!")
                return True
            else:
                self.logger.error("Erro ao processar ROM")
                return False

        except Exception as e:
            self.logger.error(f"Erro no processamento: {str(e)}")
            return False

    def modo_interativo(self):
        """Modo interativo para uso manual"""
        print("\n" + "="*60)
        print("SEGA CD TRANSLATION ENGINE - MODO INTERATIVO")
        print("="*60)

        while True:
            print("\nOpções:")
            print("1. Processar ROM")
            print("2. Configurar tradutor")
            print("3. Ver configurações")
            print("4. Sair")

            try:
                escolha = input("\nEscolha uma opção: ").strip()

                if escolha == '1':
                    self._processar_rom_interativo()
                elif escolha == '2':
                    self._configurar_tradutor()
                elif escolha == '3':
                    self._mostrar_configuracoes()
                elif escolha == '4':
                    print("Saindo...")
                    break
                else:
                    print("Opção inválida!")

            except KeyboardInterrupt:
                print("\nSaindo...")
                break
            except Exception as e:
                print(f"Erro: {e}")

    def _processar_rom_interativo(self):
        """Processa ROM no modo interativo"""
        rom_path = input("Caminho da ROM: ").strip().strip('"')

        if not rom_path:
            print("Caminho inválido!")
            return

        output_dir = input("Diretório de saída (Enter para padrão): ").strip()
        if not output_dir:
            output_dir = None

        print("Processando...")
        sucesso = self.processar_rom(rom_path, output_dir)

        if sucesso:
            print("✓ ROM processada com sucesso!")
        else:
            print("✗ Erro ao processar ROM")

    def _configurar_tradutor(self):
        """Configura o tradutor"""
        print("\nConfigurando tradutor...")

        provider = input(f"Provedor de IA (atual: {self.config['ai_provider']}): ").strip()
        if provider:
            self.config_manager.set('ai_provider', provider)

        api_key = input("API Key (deixe em branco para manter atual): ").strip()
        if api_key:
            self.config_manager.set('api_key', api_key)

        model = input(f"Modelo (atual: {self.config['model']}): ").strip()
        if model:
            self.config_manager.set('model', model)

        print("Configurações atualizadas!")

        # Reinicializa o tradutor
        if self.config_manager.get('api_key'):
            self.translator = AITranslator(self.config_manager.config)

    def _mostrar_configuracoes(self):
        """Mostra configurações atuais"""
        print("\nConfigurações atuais:")
        print("-" * 40)

        for key, value in self.config.items():
            if 'api_key' in key.lower() and value:
                # Oculta API key por segurança
                print(f"{key}: {'*' * 10}")
            else:
                print(f"{key}: {value}")

def main():
    """Função principal"""
    import argparse

    parser = argparse.ArgumentParser(description='Sega CD Translation Engine')
    parser.add_argument('rom_path', nargs='?', help='Caminho da ROM para processar')
    parser.add_argument('-o', '--output', help='Diretório de saída')
    parser.add_argument('-c', '--config', default='config.json', help='Arquivo de configuração')
    parser.add_argument('-i', '--interactive', action='store_true', help='Modo interativo')

    args = parser.parse_args()

    # Inicializa engine
    engine = SegaCDTranslationEngine(args.config)

    if args.interactive or not args.rom_path:
        # Modo interativo
        engine.modo_interativo()
    else:
        # Modo linha de comando
        sucesso = engine.processar_rom(args.rom_path, args.output)
        exit(0 if sucesso else 1)

if __name__ == "__main__":
    main()