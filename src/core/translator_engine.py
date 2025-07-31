pip install -q -U google-generativeai```

---

### Passo 2: O Código Completo (`translator.py`)

Crie o arquivo `src/core/translator.py` e cole o código completo abaixo.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translation System - Sistema de Tradução com Multiple Providers
================================================================
Suporte para Gemini, Google Translate, DeepL e outros provedores
"""

import time
import logging
import requests
import google.generativeai as genai
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import json
import re

# Configuração básica de logging para vermos o que está acontecendo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseTranslator(ABC):
    """Classe base abstrata para todos os provedores de tradução."""
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key

    @abstractmethod
    def translate(self, text: str, target_language: str, source_language: Optional[str] = "auto") -> Optional[str]:
        """
        Método principal de tradução. Deve ser implementado por cada subclasse.

        :param text: O texto a ser traduzido.
        :param target_language: O idioma de destino (ex: 'pt-br').
        :param source_language: O idioma de origem (ex: 'en'). 'auto' para detecção automática.
        :return: O texto traduzido ou None em caso de erro.
        """
        pass

    def translate_batch(self, texts: List[str], target_language: str, source_language: Optional[str] = "auto") -> List[Optional[str]]:
        """
        Traduz uma lista de textos. A implementação padrão faz isso um por um,
        mas pode ser otimizada por subclasses se a API suportar lotes.
        """
        self.logger.info(f"Iniciando tradução em lote de {len(texts)} textos...")
        translated_texts = []
        for i, text in enumerate(texts):
            try:
                translated = self.translate(text, target_language, source_language)
                translated_texts.append(translated)
                # Pausa para não sobrecarregar APIs gratuitas
                if i % 10 == 0 and i > 0:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Erro ao traduzir o texto '{text[:30]}...': {e}")
                translated_texts.append(None)
        self.logger.info("Tradução em lote concluída.")
        return translated_texts

class GeminiTranslator(BaseTranslator):
    """Implementação do tradutor usando a API do Google Gemini."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if not self.api_key:
            raise ValueError("A chave de API do Gemini é obrigatória.")

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.logger.info("Modelo Gemini configurado com sucesso.")
        except Exception as e:
            self.logger.error(f"Falha ao configurar o Gemini. Verifique sua chave de API. Erro: {e}")
            raise

    def translate(self, text: str, target_language: str, source_language: Optional[str] = "auto") -> Optional[str]:
        if not text.strip():
            return text

        # O Gemini funciona melhor com prompts claros
        prompt = f"Translate the following text from '{source_language}' to '{target_language}'. Only return the translated text, without any introductory phrases: '{text}'"

        try:
            response = self.model.generate_content(prompt)
            translated_text = response.text.strip()
            return translated_text
        except Exception as e:
            self.logger.error(f"Erro na API do Gemini: {e}")
            return None

class GoogleTranslateTranslator(BaseTranslator):
    """
    Implementação usando a API pública (não-oficial) do Google Translate.
    Ótimo para uso leve e testes, sem necessidade de chave de API.
    """
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.base_url = "https://translate.googleapis.com/translate_a/single"

    def translate(self, text: str, target_language: str, source_language: Optional[str] = "auto") -> Optional[str]:
        if not text.strip():
            return text

        params = {
            'client': 'gtx',
            'sl': source_language,
            'tl': target_language,
            'dt': 't',
            'q': text
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status() # Lança um erro para status HTTP 4xx/5xx

            # O resultado é um JSON aninhado, a tradução principal está no primeiro elemento
            result_json = response.json()
            translated_text = "".join([sentence[0] for sentence in result_json[0]])
            return translated_text.strip()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro de conexão com o Google Translate: {e}")
            return None
        except (json.JSONDecodeError, IndexError) as e:
            self.logger.error(f"Erro ao processar a resposta do Google Translate: {e}")
            return None

# --- Fábrica de Tradutores ---

def get_translator(provider: str, api_key: Optional[str] = None) -> Optional[BaseTranslator]:
    """
    Fábrica que cria e retorna a instância do tradutor desejado.

    :param provider: O nome do provedor (ex: 'gemini', 'google').
    :param api_key: A chave de API, necessária para alguns provedores.
    :return: Uma instância de um tradutor ou None se o provedor for inválido.
    """
    provider = provider.lower()
    if provider == 'gemini':
        try:
            return GeminiTranslator(api_key=api_key)
        except (ValueError, Exception) as e:
            logging.error(f"Não foi possível criar o tradutor Gemini: {e}")
            return None

    elif provider == 'google':
        return GoogleTranslateTranslator()

    # Adicione outros provedores aqui (ex: 'deepl', 'azure')
    # elif provider == 'deepl':
    #     return DeepLTranslator(api_key=api_key)

    else:
        logging.error(f"Provedor de tradução '{provider}' não suportado.")
        return None

# --- Exemplo de Uso ---

if __name__ == "__main__":
    # Para testar, você pode executar este arquivo diretamente.

    print("--- Testando Google Translate (sem chave) ---")
    google_translator = get_translator('google')
    if google_translator:
        texto_original_1 = "Hello, world! This is a test."
        traducao_1 = google_translator.translate(texto_original_1, 'pt-br', 'en')
        print(f"Original: {texto_original_1}")
        print(f"Tradução: {traducao_1}\n")

    print("\n--- Testando Gemini (com chave de API) ---")
    # **IMPORTANTE**: Substitua pela sua chave de API do Gemini AI Studio
    # Crie um arquivo chamado 'config.json' com {"GEMINI_API_KEY": "SUA_CHAVE_AQUI"}
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        GEMINI_API_KEY = config.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError):
        GEMINI_API_KEY = None
        print("AVISO: Chave de API do Gemini não encontrada. Crie um 'config.json'.")

    if GEMINI_API_KEY:
        gemini_translator = get_translator('gemini', api_key=GEMINI_API_KEY)
        if gemini_translator:
            texto_original_2 = "This is a more complex sentence to test the power of a large language model."
            traducao_2 = gemini_translator.translate(texto_original_2, 'pt-br', 'en')
            print(f"Original: {texto_original_2}")
            print(f"Tradução: {traducao_2}\n")

            # Teste de lote
            textos_para_traduzir = [
                "The quick brown fox jumps over the lazy dog.",
                "Programming is the art of telling a computer what to do.",
                "Artificial intelligence will shape our future."
            ]
            traducoes_lote = gemini_translator.translate_batch(textos_para_traduzir, 'pt-br', 'en')
            for original, traduzido in zip(textos_para_traduzir, traducoes_lote):
                print(f"Original: {original} -> Traduzido: {traduzido}")
    else:
        print("Teste do Gemini pulado.")