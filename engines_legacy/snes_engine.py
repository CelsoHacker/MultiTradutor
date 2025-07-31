import re
import pandas as pd
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import hashlib
from collections import defaultdict
import struct

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextEntry:
    """Representa um texto extraído da ROM"""
    offset: int
    original_text: str
    translated_text: str = ""
    category: str = ""
    confidence: float = 0.0
    context: str = ""
    is_proper_noun: bool = False
    length: int = 0
    encoding: str = "ascii"

@dataclass
class ROMPatch:
    """Representa um patch a ser aplicado na ROM"""
    offset: int
    original_bytes: bytes
    new_bytes: bytes
    description: str = ""

class Lufia2IntelligentScanner:
    """Scanner inteligente para extração, tradução e patch de textos da ROM Lufia 2"""

    def __init__(self, rom_path: str = None):
        self.rom_path = rom_path
        self.rom_data = None
        self.text_entries: List[TextEntry] = []
        self.patches: List[ROMPatch] = []
        self.translation_cache: Dict[str, str] = {}

        # Configurações de API de tradução
        self.translation_apis = {
            'google': {
                'url': 'https://translate.googleapis.com/translate_a/single',
                'params': {'client': 'gtx', 'sl': 'en', 'tl': 'pt', 'dt': 't'}
            },
            'libre': {
                'url': 'https://libretranslate.com/translate',
                'headers': {'Content-Type': 'application/json'}
            }
        }

        # Keywords expandidas para classificação semântica
        self.keywords = {
            'battle': [
                'attack', 'magic', 'damage', 'hp', 'mp', 'defend', 'item', 'spell', 'cast', 'hit', 'miss',
                'critical', 'poison', 'sleep', 'paralysis', 'curse', 'weapon', 'armor', 'shield',
                'bow', 'sword', 'staff', 'ring', 'potion', 'elixir', 'antidote', 'revival',
                'fire', 'water', 'earth', 'wind', 'light', 'dark', 'thunder', 'ice'
            ],
            'menu': [
                'start', 'select', 'option', 'config', 'save', 'load', 'quit', 'continue', 'new game',
                'settings', 'sound', 'music', 'controls', 'difficulty', 'reset', 'exit',
                'inventory', 'equipment', 'status', 'party', 'formation'
            ],
            'dialogue': [
                'yes', 'no', 'hello', 'thanks', 'sorry', 'help', 'what', 'where', 'how', 'why',
                'welcome', 'goodbye', 'please', 'excuse me', 'certainly', 'maybe', 'probably',
                'wonderful', 'terrible', 'amazing', 'beautiful', 'strange', 'mysterious'
            ],
            'pattern': [
                'level', 'exp', 'gold', 'strength', 'defense', 'speed', 'luck', 'agility',
                'intelligence', 'wisdom', 'charisma', 'constitution', 'dexterity',
                'points', 'bonus', 'penalty', 'modifier', 'stat', 'attribute'
            ],
            'location': [
                'town', 'city', 'village', 'castle', 'dungeon', 'cave', 'forest', 'mountain',
                'tower', 'temple', 'shrine', 'palace', 'house', 'shop', 'inn', 'tavern',
                'north', 'south', 'east', 'west', 'entrance', 'exit', 'door', 'stairs'
            ]
        }

        # Nomes próprios conhecidos de Lufia 2
        self.proper_nouns = {
            'characters': [
                'Maxim', 'Selan', 'Guy', 'Artea', 'Tia', 'Dekar', 'Lexis', 'Iris',
                'Doom Island', 'Gratze', 'Parcelyte', 'Elcid', 'Chaed', 'Forfeit',
                'Gades', 'Amon', 'Daos', 'Erim', 'Dual Blade', 'Sinistrals'
            ],
            'places': [
                'Parcelyte', 'Elcid', 'Chaed', 'Forfeit', 'Gratze', 'Clamento',
                'Merix', 'Dankirk', 'Bound', 'Gruberik', 'Karlloon', 'Treble',
                'Portoa', 'Jyad', 'Pico', 'Esturk', 'Doom Island'
            ],
            'items': [
                'Dual Blade', 'Zircon Blade', 'Excalibur', 'Masamune', 'Miracle',
                'Gades Blade', 'Amon Blade', 'Daos Blade', 'Erim Blade'
            ]
        }

        # Dicionário de tradução expandido
        self.translation_dict = {
            # Menu/Sistema
            'start': 'iniciar', 'select': 'selecionar', 'option': 'opção', 'config': 'configurar',
            'save': 'salvar', 'load': 'carregar', 'quit': 'sair', 'continue': 'continuar',
            'new game': 'novo jogo', 'settings': 'configurações', 'sound': 'som',
            'music': 'música', 'controls': 'controles', 'difficulty': 'dificuldade',
            'reset': 'reiniciar', 'exit': 'sair', 'inventory': 'inventário',
            'equipment': 'equipamento', 'status': 'estado', 'party': 'grupo',
            'formation': 'formação',

            # Batalha
            'attack': 'atacar', 'magic': 'magia', 'damage': 'dano', 'defend': 'defender',
            'item': 'item', 'spell': 'feitiço', 'cast': 'conjurar', 'hit': 'acerto',
            'miss': 'erro', 'critical': 'crítico', 'poison': 'veneno', 'sleep': 'sono',
            'paralysis': 'paralisia', 'curse': 'maldição', 'weapon': 'arma',
            'armor': 'armadura', 'shield': 'escudo', 'bow': 'arco', 'sword': 'espada',
            'staff': 'cajado', 'ring': 'anel', 'potion': 'poção', 'elixir': 'elixir',
            'antidote': 'antídoto', 'revival': 'ressurreição',

            # Elementos
            'fire': 'fogo', 'water': 'água', 'earth': 'terra', 'wind': 'vento',
            'light': 'luz', 'dark': 'trevas', 'thunder': 'trovão', 'ice': 'gelo',

            # Diálogo
            'yes': 'sim', 'no': 'não', 'hello': 'olá', 'thanks': 'obrigado',
            'sorry': 'desculpe', 'help': 'ajuda', 'what': 'o que', 'where': 'onde',
            'how': 'como', 'why': 'por que', 'welcome': 'bem-vindo', 'goodbye': 'tchau',
            'please': 'por favor', 'excuse me': 'com licença', 'certainly': 'certamente',
            'maybe': 'talvez', 'probably': 'provavelmente', 'wonderful': 'maravilhoso',
            'terrible': 'terrível', 'amazing': 'incrível', 'beautiful': 'bonito',
            'strange': 'estranho', 'mysterious': 'misterioso',

            # Atributos
            'level': 'nível', 'exp': 'experiência', 'gold': 'ouro', 'strength': 'força',
            'defense': 'defesa', 'speed': 'velocidade', 'luck': 'sorte',
            'agility': 'agilidade', 'intelligence': 'inteligência', 'wisdom': 'sabedoria',
            'charisma': 'carisma', 'constitution': 'constituição', 'dexterity': 'destreza',
            'points': 'pontos', 'bonus': 'bônus', 'penalty': 'penalidade',
            'modifier': 'modificador', 'stat': 'atributo', 'attribute': 'atributo',

            # Localização
            'town': 'cidade', 'city': 'cidade', 'village': 'vila', 'castle': 'castelo',
            'dungeon': 'masmorra', 'cave': 'caverna', 'forest': 'floresta',
            'mountain': 'montanha', 'tower': 'torre', 'temple': 'templo',
            'shrine': 'santuário', 'palace': 'palácio', 'house': 'casa',
            'shop': 'loja', 'inn': 'pousada', 'tavern': 'taverna',
            'north': 'norte', 'south': 'sul', 'east': 'leste', 'west': 'oeste',
            'entrance': 'entrada', 'exit': 'saída', 'door': 'porta', 'stairs': 'escadas'
        }

    def load_rom(self, rom_path: str) -> bool:
        """Carrega dados da ROM com validação de header SNES"""
        try:
            self.rom_path = rom_path
            with open(rom_path, 'rb') as f:
                self.rom_data = bytearray(f.read())  # Usar bytearray para modificações

            # Validação básica de ROM SNES
            if len(self.rom_data) < 512:
                logger.error("ROM muito pequena para ser válida")
                return False

            # Detecta se tem header SMC
            has_header = len(self.rom_data) % 1024 == 512
            if has_header:
                logger.info("Header SMC detectado - removendo...")
                self.rom_data = self.rom_data[512:]

            logger.info(f"ROM carregada: {len(self.rom_data)} bytes")
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar ROM: {e}")
            return False

    def calculate_text_probability(self, data: bytes) -> float:
        """Algoritmo aprimorado para detecção de texto"""
        if not data or len(data) < 2:
            return 0.0

        # Contadores para análise estatística
        printable_count = 0
        letter_count = 0
        space_count = 0
        punctuation_count = 0
        digit_count = 0
        consecutive_letters = 0
        max_consecutive = 0

        for i, byte in enumerate(data):
            # Verifica se é printável (ASCII 32-126)
            if 32 <= byte <= 126:
                printable_count += 1

                # Analisa tipos de caracteres
                if (65 <= byte <= 90) or (97 <= byte <= 122):  # Letras
                    letter_count += 1
                    consecutive_letters += 1
                    max_consecutive = max(max_consecutive, consecutive_letters)
                elif byte == 32:  # Espaço
                    space_count += 1
                    consecutive_letters = 0
                elif byte in [46, 44, 33, 63, 59, 58]:  # Pontuação
                    punctuation_count += 1
                    consecutive_letters = 0
                elif 48 <= byte <= 57:  # Dígitos
                    digit_count += 1
                    consecutive_letters = 0
                else:
                    consecutive_letters = 0

        total_bytes = len(data)

        # Ratios para análise
        printable_ratio = printable_count / total_bytes
        letter_ratio = letter_count / total_bytes
        space_ratio = space_count / total_bytes
        punct_ratio = punctuation_count / total_bytes
        digit_ratio = digit_count / total_bytes

        # Algoritmo de pontuação aprimorado
        score = 0.0

        # Peso por tipo de caractere
        score += printable_ratio * 0.3
        score += letter_ratio * 0.4
        score += min(space_ratio * 3, 0.15)  # Espaços são importantes mas não demais
        score += min(punct_ratio * 2, 0.1)   # Pontuação indica texto natural
        score += min(digit_ratio * 1, 0.05)  # Alguns números são OK

        # Bônus por sequências de letras (palavras)
        if max_consecutive >= 3:
            score += 0.1
        if max_consecutive >= 5:
            score += 0.1

        # Penalidades
        if letter_ratio < 0.3:  # Muito poucas letras
            score *= 0.5
        if space_ratio > 0.5:   # Muitos espaços
            score *= 0.3
        if printable_ratio < 0.8:  # Muitos caracteres não-printáveis
            score *= 0.2

        return min(score, 1.0)

    def is_proper_noun(self, text: str) -> bool:
        """Detecta se o texto é um nome próprio"""
        text_clean = text.strip()

        # Verifica se está na lista de nomes conhecidos
        for category, names in self.proper_nouns.items():
            if text_clean in names:
                return True

        # Heurísticas para detecção
        # 1. Começa com maiúscula e não tem palavras comuns
        if (text_clean[0].isupper() and
            len(text_clean) > 2 and
            text_clean.lower() not in self.translation_dict):
            return True

        # 2. Todas as palavras começam com maiúscula
        words = text_clean.split()
        if len(words) > 1 and all(word[0].isupper() for word in words if word):
            return True

        return False

    def classify_text(self, text: str) -> str:
        """Classificação aprimorada com detecção de contexto"""
        text_lower = text.lower()

        # Verifica se é nome próprio primeiro
        if self.is_proper_noun(text):
            return 'proper_noun'

        # Contadores por categoria
        category_scores = defaultdict(int)

        # Análise por keywords
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Peso maior para match exato
                    if keyword == text_lower:
                        category_scores[category] += 3
                    # Peso menor para match parcial
                    else:
                        category_scores[category] += 1

        # Heurísticas adicionais
        # Textos curtos com números provavelmente são stats
        if len(text) < 10 and any(c.isdigit() for c in text):
            category_scores['pattern'] += 2

        # Textos com interrogação são diálogos
        if '?' in text:
            category_scores['dialogue'] += 2

        # Textos com comandos são menus
        if any(word in text_lower for word in ['press', 'button', 'key']):
            category_scores['menu'] += 2

        # Retorna categoria com maior pontuação
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'unknown'

    def scan_rom(self, min_length: int = 3, max_length: int = 100) -> List[TextEntry]:
        """Scanner otimizado com análise contextual"""
        if not self.rom_data:
            logger.error("ROM não carregada")
            return []

        logger.info("Iniciando escaneamento inteligente...")
        self.text_entries = []
        processed_offsets = set()

        # Primeira passagem: busca por strings terminadas em null
        logger.info("Fase 1: Buscando strings null-terminated...")
        for offset in range(len(self.rom_data) - max_length):
            if offset in processed_offsets:
                continue

            # Procura por sequência terminada em 0x00
            end_offset = offset
            while (end_offset < len(self.rom_data) and
                   self.rom_data[end_offset] != 0 and
                   end_offset - offset < max_length):
                end_offset += 1

            length = end_offset - offset
            if min_length <= length <= max_length:
                chunk = self.rom_data[offset:end_offset]
                probability = self.calculate_text_probability(chunk)

                if probability > 0.4:  # Threshold mais alto para null-terminated
                    self._process_text_chunk(offset, chunk, probability, processed_offsets)

        # Segunda passagem: busca por chunks de tamanho fixo
        logger.info("Fase 2: Buscando chunks de tamanho fixo...")
        for offset in range(0, len(self.rom_data) - max_length, 2):  # Step de 2 para otimizar
            if offset in processed_offsets:
                continue

            for length in range(min_length, max_length + 1):
                if offset + length > len(self.rom_data):
                    break

                chunk = self.rom_data[offset:offset + length]
                probability = self.calculate_text_probability(chunk)

                if probability > 0.35:  # Threshold para chunks fixos
                    if self._process_text_chunk(offset, chunk, probability, processed_offsets):
                        break  # Encontrou texto válido, pula para próximo offset

        # Pós-processamento: remove duplicatas e ordena
        self._post_process_entries()

        logger.info(f"Escaneamento concluído: {len(self.text_entries)} textos únicos")
        return self.text_entries

    def _process_text_chunk(self, offset: int, chunk: bytes, probability: float, processed_offsets: set) -> bool:
        """Processa um chunk de texto encontrado"""
        try:
            # Tenta diferentes encodings
            for encoding in ['ascii', 'latin1', 'utf-8']:
                try:
                    text = chunk.decode(encoding, errors='ignore').strip()

                    # Validações
                    if (len(text) >= 3 and
                        re.match(r'^[a-zA-Z0-9\s\.,!?\-\'\"]+$', text) and
                        not text.isspace() and
                        not text.replace(' ', '').isdigit()):

                        # Verifica se não é muito repetitivo
                        if not self._is_repetitive_text(text):
                            category = self.classify_text(text)
                            is_proper = self.is_proper_noun(text)

                            entry = TextEntry(
                                offset=offset,
                                original_text=text,
                                category=category,
                                confidence=probability,
                                is_proper_noun=is_proper,
                                length=len(chunk),
                                encoding=encoding
                            )

                            self.text_entries.append(entry)

                            # Marca região como processada
                            for i in range(offset, offset + len(chunk)):
                                processed_offsets.add(i)

                            return True

                except UnicodeDecodeError:
                    continue

            return False

        except Exception as e:
            logger.debug(f"Erro ao processar chunk em {offset:08X}: {e}")
            return False

    def _is_repetitive_text(self, text: str) -> bool:
        """Detecta textos repetitivos ou inválidos"""
        # Muito repetitivo (mesmo caractere)
        if len(set(text.replace(' ', ''))) < 3:
            return True

        # Padrões inválidos
        invalid_patterns = [
            r'^[A-Z]{4,}$',  # Só maiúsculas
            r'^[a-z]{10,}$', # Só minúsculas longas
            r'^[\d\s]+$',    # Só números e espaços
            r'^[^\w\s]+$'    # Só símbolos
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, text):
                return True

        return False

    def _post_process_entries(self):
        """Pós-processamento dos textos encontrados"""
        # Remove duplicatas exatas
        seen_texts = set()
        unique_entries = []

        for entry in self.text_entries:
            if entry.original_text not in seen_texts:
                unique_entries.append(entry)
                seen_texts.add(entry.original_text)

        # Remove substrings de textos maiores
        filtered_entries = []
        for entry in unique_entries:
            is_substring = False
            for other in unique_entries:
                if (entry != other and
                    entry.original_text in other.original_text and
                    len(entry.original_text) < len(other.original_text)):
                    is_substring = True
                    break

            if not is_substring:
                filtered_entries.append(entry)

        # Ordena por offset
        self.text_entries = sorted(filtered_entries, key=lambda x: x.offset)

    def translate_with_api(self, text: str, api: str = 'google', max_retries: int = 3) -> str:
        """Traduz texto usando API externa"""
        # Verifica cache primeiro
        cache_key = f"{api}:{text}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        # Não traduz nomes próprios
        if self.is_proper_noun(text):
            return text

        for attempt in range(max_retries):
            try:
                if api == 'google':
                    response = self._translate_google(text)
                elif api == 'libre':
                    response = self._translate_libre(text)
                else:
                    return text

                if response and response != text:
                    self.translation_cache[cache_key] = response
                    return response

            except Exception as e:
                logger.warning(f"Erro na tradução (tentativa {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Aguarda antes de tentar novamente

        return text  # Retorna original se falhar

    def _translate_google(self, text: str) -> str:
        """Traduz usando Google Translate (não oficial)"""
        params = self.translation_apis['google']['params'].copy()
        params['q'] = text

        response = requests.get(
            self.translation_apis['google']['url'],
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if result and result[0] and result[0][0]:
                return result[0][0][0]

        return text

    def _translate_libre(self, text: str) -> str:
        """Traduz usando LibreTranslate"""
        data = {
            'q': text,
            'source': 'en',
            'target': 'pt',
            'format': 'text'
        }

        response = requests.post(
            self.translation_apis['libre']['url'],
            json=data,
            headers=self.translation_apis['libre']['headers'],
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if 'translatedText' in result:
                return result['translatedText']

        return text

    def translate_text(self, text: str, use_api: bool = True) -> str:
        """Traduz texto usando múltiplas estratégias"""
        # 1. Verifica se é nome próprio
        if self.is_proper_noun(text):
            return text

        # 2. Busca no dicionário local
        text_lower = text.lower()
        if text_lower in self.translation_dict:
            return self.translation_dict[text_lower]

        # 3. Traduz palavras individuais
        words = text.split()
        if len(words) > 1:
            translated_words = []
            for word in words:
                clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
                if clean_word in self.translation_dict:
                    translated_words.append(self.translation_dict[clean_word])
                else:
                    translated_words.append(word)

            partial_translation = ' '.join(translated_words)
            if partial_translation != text:
                return partial_translation

        # 4. Usa API se disponível
        if use_api and len(text) > 2:
            api_translation = self.translate_with_api(text)
            if api_translation != text:
                return api_translation

        return text  # Retorna original se não conseguir traduzir

    def auto_translate(self, use_api: bool = True, batch_size: int = 50) -> int:
        """Traduz automaticamente todos os textos com progresso"""
        translated_count = 0
        total_texts = len(self.text_entries)

        logger.info(f"Iniciando tradução automática de {total_texts} textos...")

        for i, entry in enumerate(self.text_entries):
            if not entry.translated_text:
                translation = self.translate_text(entry.original_text, use_api)
                if translation != entry.original_text:
                    entry.translated_text = translation
                    translated_count += 1

            # Log de progresso
            if (i + 1) % batch_size == 0:
                progress = (i + 1) / total_texts * 100
                logger.info(f"Progresso: {progress:.1f}% ({i + 1}/{total_texts})")

        logger.info(f"Tradução concluída: {translated_count} textos traduzidos")
        return translated_count

    def create_patch(self, entry: TextEntry) -> ROMPatch:
        """Cria um patch para aplicar tradução na ROM"""
        original_bytes = entry.original_text.encode(entry.encoding)
        translated_bytes = entry.translated_text.encode(entry.encoding)

        # Ajusta tamanho se necessário
        if len(translated_bytes) > len(original_bytes):
            # Trunca se for muito longo
            translated_bytes = translated_bytes[:len(original_bytes)]
        elif len(translated_bytes) < len(original_bytes):
            # Preenche com zeros
            translated_bytes += b'\x00' * (len(original_bytes) - len(translated_bytes))

        patch = ROMPatch(
            offset=entry.offset,
            original_bytes=original_bytes,
            new_bytes=translated_bytes,
            description=f"Tradução: '{entry.original_text}' -> '{entry.translated_text}'"
        )

        return patch

    def generate_patches(self) -> List[ROMPatch]:
        """Gera patches para todos os textos traduzidos"""
        self.patches = []

        for entry in self.text_entries:
            if entry.translated_text and entry.translated_text != entry.original_text:
                patch = self.create_patch(entry)
                self.patches.append(patch)

        logger.info(f"Gerados {len(self.patches)} patches")
        return self.patches

    def apply_patches(self, output_path: str = None) -> bool:
        """Aplica patches na ROM e salva nova versão"""
        if not self.patches:
            logger.warning("Nenhum patch para aplicar")
            return False

        if not output_path:
            output_path = self.rom_path.replace('.smc', '_traduzido.smc')

        try:
            # Cria cópia da ROM original
            patched_rom = bytearray(self.rom_data)

            # Aplica patches
            applied_count = 0
            for patch in self.patches:
                if patch.offset + len(patch.new_bytes) <= len(patched_rom):
                    # Verifica se o original ainda está lá
                    current_bytes = patched_rom[patch.offset:patch.offset + len(patch.original_bytes)]
                    if current_bytes == patch.original_bytes:
                        patched_rom[patch.offset:patch.offset + len(patch.new_bytes)] = patch.new_bytes
                        applied_count += 1
                    else:
                        logger.warning(f"Patch em {patch.offset:08X} não aplicado - dados alterados")
                else:
                    logger.warning(f"Patch em {patch.offset:08X} fora dos limites da ROM")

            # Salva ROM traduzida
            with open(output_path, 'wb') as f:
                f.write(patched_rom)

            logger.info(f"ROM traduzida salva: {output_path}")
            logger.info(f"Patches aplicados: {applied_count}/{len(self.patches)}")
            return True

        except Exception as e:
            logger.error(f"Erro ao aplicar patches: {e}")
            return False

    def generate_translation_template(self, output_path: str = "lufia2_translation.json") -> bool:
        """Gera template JSON completo com metadados"""
        try:
            # Calcula estatísticas
            stats = self.get_statistics()

            template_data = {
                "metadata": {
                    "rom_path": self.rom_path,
                    "rom_size": len(self.rom_data) if self.rom_data else 0,
                    "scan_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_texts": len(self.text_entries),
                    "translated_count": stats.get('translated', 0),
                    "translation_percentage": stats.get('translation_percentage', 0),
                    "categories": stats.get('categories', {}),
                    "average_confidence": stats.get('average_confidence', 0),
                    "proper_nouns": stats.get('proper_nouns', 0)
                },
                "configuration": {
                    "min_confidence": 0.35,
                    "encodings_used": ["ascii", "latin1", "utf-8"],
                    "api_translation_enabled": True,
                    "preserve_proper_nouns": True
                },
                "texts": []
            }

            # Adiciona textos com contexto completo
            for entry in self.text_entries:
                text_data = {
                    "id": len(template_data["texts"]) + 1,
                    "offset": f"0x{entry.offset:08X}",
                    "offset_decimal": entry.offset,
                    "original": entry.original_text,
                    "translated": entry.translated_text,
                    "category": entry.category,
                    "confidence": round(entry.confidence, 3),
                    "is_proper_noun": entry.is_proper_noun,
                    "length": entry.length,
                    "encoding": entry.encoding,
                    "status": "translated" if entry.translated_text else "pending",
                    "translation_method": self._get_translation_method(entry),
                    "context_before": self._get_context(entry.offset - 20, 20),
                    "context_after": self._get_context(entry.offset + entry.length, 20)
                }
                template_data["texts"].append(text_data)

            # Salva template
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Template completo gerado: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao gerar template: {e}")
            return False

    def _get_translation_method(self, entry: TextEntry) -> str:
        """Determina método usado para traduzir"""
        if not entry.translated_text:
            return "none"
        elif entry.is_proper_noun:
            return "preserved"
        elif entry.original_text.lower() in self.translation_dict:
            return "dictionary"
        elif entry.translated_text in self.translation_cache.values():
            return "api"
        else:
            return "partial"

    def _get_context(self, offset: int, length: int) -> str:
        """Obtém contexto binário ao redor do texto"""
        if not self.rom_data or offset < 0 or offset >= len(self.rom_data):
            return ""

        end_offset = min(offset + length, len(self.rom_data))
        context_bytes = self.rom_data[offset:end_offset]

        # Converte para hex string legível
        hex_str = ' '.join(f'{b:02X}' for b in context_bytes)
        return hex_str

    def export_to_excel(self, output_path: str = "lufia2_translation.xlsx") -> bool:
        """Exporta dados para Excel com múltiplas abas"""
        try:
            # Prepara dados para diferentes abas

            # Aba 1: Textos principais
            main_data = []
            for entry in self.text_entries:
                main_data.append({
                    'ID': f"T{entry.offset:08X}",
                    'Offset': f"0x{entry.offset:08X}",
                    'Original': entry.original_text,
                    'Traduzido': entry.translated_text,
                    'Categoria': entry.category,
                    'Confiança': f"{entry.confidence:.3f}",
                    'Nome Próprio': 'Sim' if entry.is_proper_noun else 'Não',
                    'Tamanho': entry.length,
                    'Status': 'Traduzido' if entry.translated_text else 'Pendente',
                    'Método': self._get_translation_method(entry),
                    'Encoding': entry.encoding
                })

            # Aba 2: Estatísticas por categoria
            stats = self.get_statistics()
            category_data = []
            for category, count in stats.get('categories', {}).items():
                translated_in_cat = sum(1 for entry in self.text_entries
                                      if entry.category == category and entry.translated_text)
                category_data.append({
                    'Categoria': category,
                    'Total': count,
                    'Traduzidos': translated_in_cat,
                    'Pendentes': count - translated_in_cat,
                    'Progresso': f"{translated_in_cat/count*100:.1f}%" if count > 0 else "0%"
                })

            # Aba 3: Nomes próprios
            proper_nouns_data = []
            for entry in self.text_entries:
                if entry.is_proper_noun:
                    proper_nouns_data.append({
                        'Nome': entry.original_text,
                        'Offset': f"0x{entry.offset:08X}",
                        'Categoria': entry.category,
                        'Mantido': 'Sim' if entry.translated_text == entry.original_text else 'Não',
                        'Tradução': entry.translated_text if entry.translated_text != entry.original_text else '-'
                    })

            # Aba 4: Patches gerados
            patch_data = []
            if self.patches:
                for i, patch in enumerate(self.patches):
                    patch_data.append({
                        'ID': f"P{i+1:04d}",
                        'Offset': f"0x{patch.offset:08X}",
                        'Tamanho Original': len(patch.original_bytes),
                        'Tamanho Novo': len(patch.new_bytes),
                        'Bytes Originais': ' '.join(f'{b:02X}' for b in patch.original_bytes[:20]),
                        'Bytes Novos': ' '.join(f'{b:02X}' for b in patch.new_bytes[:20]),
                        'Descrição': patch.description
                    })

            # Cria Excel com múltiplas abas
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Aba principal
                pd.DataFrame(main_data).to_excel(writer, sheet_name='Textos', index=False)

                # Aba de estatísticas
                pd.DataFrame(category_data).to_excel(writer, sheet_name='Estatísticas', index=False)

                # Aba de nomes próprios
                if proper_nouns_data:
                    pd.DataFrame(proper_nouns_data).to_excel(writer, sheet_name='Nomes Próprios', index=False)

                # Aba de patches
                if patch_data:
                    pd.DataFrame(patch_data).to_excel(writer, sheet_name='Patches', index=False)

                # Aba de resumo
                summary_data = [{
                    'Métrica': 'Total de Textos',
                    'Valor': len(self.text_entries)
                }, {
                    'Métrica': 'Textos Traduzidos',
                    'Valor': stats.get('translated', 0)
                }, {
                    'Métrica': 'Progresso Geral',
                    'Valor': f"{stats.get('translation_percentage', 0):.1f}%"
                }, {
                    'Métrica': 'Confiança Média',
                    'Valor': f"{stats.get('average_confidence', 0):.3f}"
                }, {
                    'Métrica': 'Nomes Próprios',
                    'Valor': stats.get('proper_nouns', 0)
                }, {
                    'Métrica': 'Patches Gerados',
                    'Valor': len(self.patches)
                }]
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumo', index=False)

            logger.info(f"Exportado para Excel: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao exportar Excel: {e}")
            return False

    def export_patches(self, output_path: str = "lufia2_patches.txt") -> bool:
        """Exporta patches em formato texto legível"""
        try:
            if not self.patches:
                logger.warning("Nenhum patch para exportar")
                return False

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== PATCHES PARA LUFIA 2 ===\n")
                f.write(f"Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de patches: {len(self.patches)}\n\n")

                for i, patch in enumerate(self.patches):
                    f.write(f"PATCH #{i+1:04d}\n")
                    f.write(f"Offset: 0x{patch.offset:08X} ({patch.offset})\n")
                    f.write(f"Descrição: {patch.description}\n")
                    f.write(f"Bytes originais: {' '.join(f'{b:02X}' for b in patch.original_bytes)}\n")
                    f.write(f"Bytes novos: {' '.join(f'{b:02X}' for b in patch.new_bytes)}\n")
                    f.write(f"Tamanho: {len(patch.original_bytes)} -> {len(patch.new_bytes)} bytes\n")
                    f.write("-" * 50 + "\n\n")

            logger.info(f"Patches exportados: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao exportar patches: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Retorna estatísticas completas do escaneamento"""
        if not self.text_entries:
            return {}

        translated_count = sum(1 for entry in self.text_entries if entry.translated_text)
        proper_nouns_count = sum(1 for entry in self.text_entries if entry.is_proper_noun)

        stats = {
            'total_texts': len(self.text_entries),
            'translated': translated_count,
            'pending': len(self.text_entries) - translated_count,
            'translation_percentage': (translated_count / len(self.text_entries)) * 100,
            'proper_nouns': proper_nouns_count,
            'categories': defaultdict(int),
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'average_confidence': sum(entry.confidence for entry in self.text_entries) / len(self.text_entries),
            'encodings_used': set(),
            'length_distribution': {'short': 0, 'medium': 0, 'long': 0}
        }

        # Estatísticas por categoria
        for entry in self.text_entries:
            stats['categories'][entry.category] += 1
            stats['encodings_used'].add(entry.encoding)

            # Distribuição de confiança
            if entry.confidence >= 0.7:
                stats['confidence_distribution']['high'] += 1
            elif entry.confidence >= 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1

            # Distribuição de tamanho
            if len(entry.original_text) <= 10:
                stats['length_distribution']['short'] += 1
            elif len(entry.original_text) <= 30:
                stats['length_distribution']['medium'] += 1
            else:
                stats['length_distribution']['long'] += 1

        # Converte set para lista para serialização JSON
        stats['encodings_used'] = list(stats['encodings_used'])
        stats['categories'] = dict(stats['categories'])

        return stats

    def save_translation_cache(self, cache_path: str = "translation_cache.json") -> bool:
        """Salva cache de traduções para reutilização"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Cache de traduções salvo: {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
            return False

    def load_translation_cache(self, cache_path: str = "translation_cache.json") -> bool:
        """Carrega cache de traduções salvo"""
        try:
            if Path(cache_path).exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.translation_cache = json.load(f)
                logger.info(f"Cache de traduções carregado: {len(self.translation_cache)} entradas")
                return True
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {e}")
        return False

    def validate_translation(self, entry: TextEntry) -> Dict[str, any]:
        """Valida qualidade da tradução"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'score': 1.0
        }

        if not entry.translated_text:
            validation['errors'].append("Texto não traduzido")
            validation['valid'] = False
            validation['score'] = 0.0
            return validation

        # Verifica se é muito diferente do original
        if len(entry.translated_text) > len(entry.original_text) * 1.5:
            validation['warnings'].append("Tradução muito longa")
            validation['score'] -= 0.2

        # Verifica se mantém formatação
        if entry.original_text.isupper() and not entry.translated_text.isupper():
            validation['warnings'].append("Formatação de maiúsculas perdida")
            validation['score'] -= 0.1

        # Verifica se tradução é igual ao original
        if entry.translated_text == entry.original_text and not entry.is_proper_noun:
            validation['warnings'].append("Texto não foi traduzido")
            validation['score'] -= 0.3

        # Verifica caracteres especiais
        if re.search(r'[^\w\s\.,!?-]', entry.translated_text):
            validation['warnings'].append("Caracteres especiais detectados")
            validation['score'] -= 0.1

        return validation

    def generate_report(self, output_path: str = "lufia2_report.html") -> bool:
        """Gera relatório HTML completo"""
        try:
            stats = self.get_statistics()

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Relatório de Tradução - Lufia 2</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; justify-content: space-between; margin: 20px 0; }}
        .stat-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
        .progress-bar {{ width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; }}
        .progress-fill {{ height: 100%; background: #27ae60; border-radius: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        .category-menu {{ color: #e74c3c; }}
        .category-battle {{ color: #e67e22; }}
        .category-dialogue {{ color: #9b59b6; }}
        .category-pattern {{ color: #2ecc71; }}
        .category-proper_noun {{ color: #f39c12; }}
        .category-unknown {{ color: #95a5a6; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 Relatório de Tradução - Lufia 2</h1>
        <p>Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <h3>{stats['total_texts']}</h3>
            <p>Textos Encontrados</p>
        </div>
        <div class="stat-box">
            <h3>{stats['translated']}</h3>
            <p>Traduzidos</p>
        </div>
        <div class="stat-box">
            <h3>{stats['translation_percentage']:.1f}%</h3>
            <p>Progresso</p>
        </div>
        <div class="stat-box">
            <h3>{stats['average_confidence']:.3f}</h3>
            <p>Confiança Média</p>
        </div>
    </div>

    <div class="progress-bar">
        <div class="progress-fill" style="width: {stats['translation_percentage']}%"></div>
    </div>

    <h2>📊 Estatísticas por Categoria</h2>
    <table>
        <tr><th>Categoria</th><th>Total</th><th>Traduzidos</th><th>Progresso</th></tr>
"""

            # Adiciona estatísticas por categoria
            for category, count in stats['categories'].items():
                translated_in_cat = sum(1 for entry in self.text_entries
                                      if entry.category == category and entry.translated_text)
                progress = (translated_in_cat / count * 100) if count > 0 else 0

                html_content += f"""
        <tr>
            <td class="category-{category}">{category}</td>
            <td>{count}</td>
            <td>{translated_in_cat}</td>
            <td>{progress:.1f}%</td>
        </tr>
"""

            html_content += """
    </table>

    <h2>📝 Textos Encontrados</h2>
    <table>
        <tr><th>Offset</th><th>Original</th><th>Traduzido</th><th>Categoria</th><th>Confiança</th></tr>
"""

            # Adiciona textos (limitado a 100 para não ficar muito grande)
            for entry in self.text_entries[:100]:
                status_icon = "✅" if entry.translated_text else "⏳"
                html_content += f"""
        <tr>
            <td>0x{entry.offset:08X}</td>
            <td>{entry.original_text}</td>
            <td>{entry.translated_text or '-'}</td>
            <td class="category-{entry.category}">{entry.category}</td>
            <td>{entry.confidence:.3f}</td>
        </tr>
"""

            html_content += """
    </table>

    <p><em>Relatório gerado pelo Lufia2IntelligentScanner v2.0</em></p>
</body>
</html>
"""

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Relatório HTML gerado: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return False

            # Bloco de código para colar na classe Lufia2IntelligentScanner
    # =============================================================

    def auto_translate(self, use_api: bool = True, batch_size: int = 50) -> int:
        """Traduz automaticamente todos os textos com progresso"""
        translated_count = 0
        total_texts = len(self.text_entries)

        logger.info(f"Iniciando tradução automática de {total_texts} textos...")

        for i, entry in enumerate(self.text_entries):
            if not entry.translated_text:
                translation = self.translate_text(entry.original_text, use_api)
                if translation != entry.original_text:
                    entry.translated_text = translation
                    translated_count += 1

            # Log de progresso
            if (i + 1) % batch_size == 0 or (i + 1) == total_texts:
                progress = (i + 1) / total_texts * 100
                logger.info(f"Progresso: {progress:.1f}% ({i + 1}/{total_texts})")

        logger.info(f"Tradução concluída: {translated_count} textos traduzidos")
        return translated_count

    def export_texts(self, output_path: str = None) -> bool:
        """Exporta textos para um arquivo .txt, priorizando traduções quando disponíveis"""
        try:
            if not output_path:
                base_name = self.rom_path.replace('.smc', '').replace('.sfc', '')
                output_path = f"{base_name}_texto_extraido.txt"

            logger.info(f"Exportando textos para: {output_path}")

            translated_count = 0
            original_count = 0

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"TEXTOS EXTRAÍDOS E TRADUZIDOS DA ROM: {Path(self.rom_path).name}\n")
                f.write("=" * 80 + "\n\n")

                for i, entry in enumerate(self.text_entries):
                    if entry.translated_text and entry.translated_text.strip() and entry.translated_text != entry.original_text:
                        final_text = entry.translated_text
                        status = "[TRADUZIDO]"
                        translated_count += 1
                    else:
                        final_text = entry.original_text
                        status = "[ORIGINAL]"
                        original_count += 1

                    f.write(f"--- TEXTO {i+1:04d} {status} ---\n")
                    f.write(f"Posição (Offset): 0x{entry.offset:08X}\n")

                    if status == "[TRADUZIDO]":
                        f.write(f"Original: {entry.original_text}\n")
                        f.write(f"Tradução: {final_text}\n")
                    else:
                        f.write(f"Texto: {final_text}\n")

                    f.write("-" * 40 + "\n\n")

            logger.info(f"Exportação para '{output_path}' concluída!")
            logger.info(f"Total de textos: {len(self.text_entries)} | Traduzidos: {translated_count} | Originais: {original_count}")

            return True

        except Exception as e:
            logger.error(f"Erro na exportação para .txt: {e}")
            return False

    def save_translated_rom(self, output_path: str = None) -> bool:
        """Salva a ROM com os textos traduzidos aplicados diretamente"""
        if not output_path:
            base_name = Path(self.rom_path).stem
            output_path = f"{base_name}_traduzido.smc"

        logger.info(f"Criando ROM traduzida em: {output_path}")

        # Usa os dados da ROM já carregados em self.rom_data
        patched_rom = bytearray(self.rom_data)
        applied_count = 0

        for entry in self.text_entries:
            if entry.translated_text and entry.translated_text.strip() and entry.translated_text != entry.original_text:
                try:
                    original_bytes = entry.original_text.encode(entry.encoding)
                    translated_bytes = entry.translated_text.encode(entry.encoding)

                    # Verifica se o texto traduzido cabe no espaço do original
                    if len(translated_bytes) <= len(original_bytes):
                        # Preenche com zeros (null bytes) se a tradução for menor
                        padding_needed = len(original_bytes) - len(translated_bytes)
                        final_bytes = translated_bytes + (b'\x00' * padding_needed)

                        # Aplica a alteração
                        start = entry.offset
                        end = start + len(original_bytes)
                        patched_rom[start:end] = final_bytes
                        applied_count += 1
                    else:
                        logger.warning(f"Tradução em 0x{entry.offset:08X} é longa demais ('{entry.translated_text}') e foi ignorada.")

                except Exception as e:
                    logger.error(f"Erro ao aplicar patch em 0x{entry.offset:08X}: {e}")

        with open(output_path, 'wb') as f:
            f.write(patched_rom)

        logger.info(f"ROM traduzida salva com sucesso! {applied_count} textos aplicados.")
        return True

    def get_display_text(self, entry) -> str:
        """Retorna o texto a ser exibido (traduzido ou original)"""
        if hasattr(entry, 'translated_text') and entry.translated_text and entry.translated_text.strip():
            return entry.translated_text
        return entry.original_text

    def get_translation_stats(self) -> dict:
        """Retorna estatísticas da tradução"""
        stats = {
            'total_texts': len(self.text_entries),
            'translated_texts': 0,
            'original_texts': 0,
            'translation_percentage': 0.0
        }

        for entry in self.text_entries:
            if hasattr(entry, 'translated_text') and entry.translated_text and entry.translated_text.strip() and entry.translated_text != entry.original_text:
                stats['translated_texts'] += 1
            else:
                stats['original_texts'] += 1

        if stats['total_texts'] > 0:
            stats['translation_percentage'] = (stats['translated_texts'] / stats['total_texts']) * 100

        return stats

    def apply_patches(self, output_path: str = None) -> bool:
        """Aplica as traduções diretamente na ROM e a salva."""
        try:
            logger.info("Iniciando processo de aplicação de patches (traduções)...")

            stats = self.get_translation_stats()
            if stats['translated_texts'] == 0:
                logger.warning("Nenhuma tradução disponível para aplicar na ROM.")
                return False

            # Chama a função para salvar a ROM modificada
            success = self.save_translated_rom(output_path)

            if success:
                logger.info(f"Processo de patching concluído. {stats['translated_texts']} traduções foram aplicadas.")

            return success

        except Exception as e:
            logger.error(f"Falha crítica ao aplicar patches: {e}")
            return False

    # =============================================================

# Exemplo de uso completo
if __name__ == "__main__":
    # Inicializa o scanner
    scanner = Lufia2IntelligentScanner()

    # Carrega cache de traduções anteriores
    scanner.load_translation_cache()

    # Carrega ROM
    rom_path = "lufia2.smc"  # Substitua pelo caminho real

    if scanner.load_rom(rom_path):
        print("🎮 ROM carregada com sucesso!")

        # Fase 1: Escaneamento
        print("\n📡 Fase 1: Escaneamento inteligente...")
        texts = scanner.scan_rom()
        print(f"📝 Encontrados {len(texts)} textos únicos")

        # Fase 2: Tradução automática
        print("\n🔄 Fase 2: Tradução automática...")
        translated = scanner.auto_translate(use_api=True)
        print(f"✅ Traduzidos automaticamente: {translated}")

        # Fase 3: Geração de patches
        print("\n🔧 Fase 3: Geração de patches...")
        patches = scanner.generate_patches()
        print(f"📦 Patches gerados: {len(patches)}")

        # Fase 4: Aplicação na ROM
        print("\n🎯 Fase 4: Aplicação na ROM...")
        if scanner.apply_patches("lufia2_traduzido.smc"):
            print("✅ ROM traduzida gerada com sucesso!")

        # Fase 5: Relatórios
        print("\n📊 Fase 5: Gerando relatórios...")
        scanner.generate_translation_template()
        scanner.export_to_excel()
        scanner.export_patches()
        scanner.generate_report()
        scanner.save_translation_cache()

        # Estatísticas finais
        stats = scanner.get_statistics()
        print(f"\n🎉 TRADUÇÃO CONCLUÍDA!")
        print(f"Total: {stats['total_texts']} textos")
        print(f"Traduzidos: {stats['translated']} ({stats['translation_percentage']:.1f}%)")
        print(f"Nomes próprios preservados: {stats['proper_nouns']}")
        print(f"Patches aplicados: {len(patches)}")
        print(f"Confiança média: {stats['average_confidence']:.3f}")

        # Mostra distribuição por categoria
        print("\n📋 Distribuição por categoria:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count} textos")

        print(f"\n📁 Arquivos gerados:")
        print(f"  - lufia2_traduzido.smc (ROM traduzida)")
        print(f"  - lufia2_translation.json (template)")
        print(f"  - lufia2_translation.xlsx (planilha)")
        print(f"  - lufia2_patches.txt (patches)")
        print(f"  - lufia2_report.html (relatório)")
        print(f"  - translation_cache.json (cache)")

    else:
        print("❌ Erro ao carregar ROM")
        print("Certifique-se de que o arquivo 'lufia2.smc' existe no diretório atual")