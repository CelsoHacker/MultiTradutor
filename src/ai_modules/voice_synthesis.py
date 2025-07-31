#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Dubbing System - Sistema de Dublagem com Multiple Providers
================================================================
Suporte para gTTS, Azure Speech, AWS Polly e TTS offline
"""

import os
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile

class BaseTTSProvider(ABC):
    """Classe base para providers de Text-to-Speech"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def generate_audio(self, text: str, output_path: str, language: str = "pt-BR") -> bool:
        """Gera arquivo de áudio a partir do texto"""
        pass

    @abstractmethod
    def get_available_voices(self, language: str = "pt-BR") -> List[str]:
        """Retorna vozes disponíveis para o idioma"""
        pass


class GTTSProvider(BaseTTSProvider):
    """Provider usando Google Text-to-Speech (gTTS)"""

    def __init__(self):
        super().__init__()
        try:
            from gtts import gTTS
            self.gtts = gTTS
            self.available = True
            self.logger.info("✅ gTTS disponível")
        except ImportError:
            self.available = False
            self.logger.warning("❌ gTTS não instalado (pip install gtts)")

    def generate_audio(self, text: str, output_path: str, language: str = "pt-BR") -> bool:
        """Gera áudio usando gTTS"""
        if not self.available or not text.strip():
            return False

        try:
            # Mapeia códigos de idioma
            lang_map = {
                "pt-BR": "pt",
                "en-US": "en",
                "es-ES": "es",
                "fr-FR": "fr",
                "de-DE": "de",
                "it-IT": "it",
                "ja-JP": "ja",
                "ko-KR": "ko",
                "zh-CN": "zh"
            }

            lang_code = lang_map.get(language, "pt")

            # Cria diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Gera o áudio
            tts = self.gtts(text=text, lang=lang_code, slow=False)
            tts.save(output_path)

            self.logger.debug(f"Áudio gerado: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Erro no gTTS: {str(e)}")
            return False

    def get_available_voices(self, language: str = "pt-BR") -> List[str]:
        """gTTS tem apenas uma voz por idioma"""
        return ["default"]


class AzureTTSProvider(BaseTTSProvider):
    """Provider usando Azure Cognitive Services Speech"""

    def __init__(self, subscription_key: str, region: str):
        super().__init__()
        self.subscription_key = subscription_key
        self.region = region

        try:
            import azure.cognitiveservices.speech as speechsdk
            self.speechsdk = speechsdk
            self.available = True
            self.logger.info("✅ Azure Speech SDK disponível")
        except ImportError:
            self.available = False
            self.logger.warning("❌ Azure Speech SDK não instalado")

    def generate_audio(self, text: str, output_path: str, language: str = "pt-BR") -> bool:
        """Gera áudio usando Azure Speech"""
        if not self.available or not text.strip():
            return False

        try:
            # Configuração da voz por idioma
            voice_map = {
                "pt-BR": "pt-BR-FranciscaNeural",
                "en-US": "en-US-JennyNeural",
                "es-ES": "es-ES-ElviraNeural",
                "fr-FR": "fr-FR-DeniseNeural",
                "de-DE": "de-DE-KatjaNeural",
                "it-IT": "it-IT-ElsaNeural"
            }

            voice_name = voice_map.get(language, "pt-BR-FranciscaNeural")

            # Configuração do Azure Speech
            speech_config = self.speechsdk.SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            speech_config.speech_synthesis_voice_name = voice_name

            # Cria diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Configuração do arquivo de saída
            audio_config = self.speechsdk.audio.AudioOutputConfig(filename=output_path)

            # Cria o sintetizador
            synthesizer = self.speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Gera o áudio
            result = synthesizer.speak_text_async(text).get()

            if result.reason == self.speechsdk.ResultReason.SynthesizingAudioCompleted:
                self.logger.debug(f"Áudio Azure gerado: {output_path}")
                return True
            else:
                self.logger.error(f"Erro Azure TTS: {result.reason}")
                return False

        except Exception as e:
            self.logger.error(f"Erro no Azure TTS: {str(e)}")
            return False

    def get_available_voices(self, language: str = "pt-BR") -> List[str]:
        """Retorna vozes disponíveis para o idioma"""
        voices_map = {
            "pt-BR": ["FranciscaNeural", "AntonioNeural", "BrendaNeural"],
            "en-US": ["JennyNeural", "GuyNeural", "AriaNeural"],
            "es-ES": ["ElviraNeural", "AlvaroNeural", "EstrellaNeural"]
        }
        return voices_map.get(language, ["default"])


class PyttsxProvider(BaseTTSProvider):
    """Provider offline usando pyttsx3"""

    def __init__(self):
        super().__init__()
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.available = True
            self.logger.info("✅ pyttsx3 disponível")

            # Configurações básicas
            self.engine.setProperty('rate', 150)  # Velocidade da fala
            self.engine.setProperty('volume', 0.9)  # Volume

        except ImportError:
            self.available = False
            self.logger.warning("❌ pyttsx3 não instalado (pip install pyttsx3)")
        except Exception as e:
            self.available = False
            self.logger.error(f"Erro ao inicializar pyttsx3: {e}")

    def generate_audio(self, text: str, output_path: str, language: str = "pt-BR") -> bool:
        """Gera áudio usando pyttsx3"""
        if not self.available or not text.strip():
            return False

        try:
            # Cria diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Configura a voz (se disponível)
            voices = self.engine.getProperty('voices')
            if voices:
                # Tenta encontrar uma voz apropriada para o idioma
                for voice in voices:
                    if language.startswith('pt') and ('brasil' in voice.name.lower() or 'port' in voice.name.lower()):
                        self.engine.setProperty('voice', voice.id)
                        break
                    elif language.startswith('en') and 'english' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break

            # Salva o áudio
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()

            self.logger.debug(f"Áudio pyttsx3 gerado: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Erro no pyttsx3: {str(e)}")
            return False

    def get_available_voices(self, language: str = "pt-BR") -> List[str]:
        """Retorna vozes disponíveis do sistema"""
        if not self.available:
            return []

        try:
            voices = self.engine.getProperty('voices')
            return [voice.name for voice in voices] if voices else ["default"]
        except:
            return ["default"]


class VoiceDubber:
    """Gerenciador principal de dublagem com múltiplos providers"""

    def __init__(self, azure_key: str = None, azure_region: str = None):
        self.logger = logging.getLogger(__name__)
        self.providers = {}

        # Inicializa providers disponíveis

        # gTTS (sempre tentar carregar - é grátis)
        gtts_provider = GTTSProvider()
        if gtts_provider.available:
            self.providers["gtts"] = gtts_provider
            self.primary_provider = "gtts"

        # Azure TTS (se configurado)
        if azure_key and azure_region:
            try:
                azure_provider = AzureTTSProvider(azure_key, azure_region)
                if azure_provider.available:
                    self.providers["azure"] = azure_provider
                    self.primary_provider = "azure"  # Azure tem melhor qualidade
                    self.logger.info("✅ Azure TTS configurado como primário")
            except Exception as e:
                self.logger.error(f"❌ Erro ao configurar Azure TTS: {e}")

        # pyttsx3 (offline fallback)
        pyttsx_provider = PyttsxProvider()
        if pyttsx_provider.available:
            self.providers["pyttsx"] = pyttsx_provider
            if not hasattr(self, 'primary_provider'):
                self.primary_provider = "pyttsx"

        if not self.providers:
            self.logger.error("❌ Nenhum provider de TTS disponível!")
            self.primary_provider = None
        else:
            self.logger.info(f"Providers TTS disponíveis: {list(self.providers.keys())}")
            self.logger.info(f"Provider primário: {self.primary_provider}")

    def generate_audio(self, text: str, output_path: str, language: str = "pt-BR") -> bool:
        """Gera áudio usando o provider primário com fallback"""
        if not text.strip():
            self.logger.warning("Texto vazio, pulando geração de áudio")
            return False

        # Lista de providers em ordem de prioridade
        provider_order = [self.primary_provider] if self.primary_provider else []

        # Adiciona outros como fallback
        for name in ["azure", "gtts", "pyttsx"]:
            if name != self.primary_provider and name in self.providers:
                provider_order.append(name)

        # Tenta gerar com cada provider
        for provider_name in provider_order:
            if provider_name in self.providers:
                try:
                    provider = self.providers[provider_name]
                    success = provider.generate_audio(text, output_path, language)

                    if success and os.path.exists(output_path):
                        self.logger.debug(f"Áudio gerado com {provider_name}: {os.path.basename(output_path)}")
                        return True

                except Exception as e:
                    self.logger.error(f"Erro em {provider_name}: {e}")
                    continue

        # Se todos falharam
        self.logger.error(f"Falha em todos os providers TTS para: {text[:50]}...")
        return False

    def batch_generate_audio(self, texts: List[str], output_dir: str,
                           language: str = "pt-BR",
                           filename_prefix: str = "audio",
                           progress_callback=None) -> List[str]:
        """Gera múltiplos arquivos de áudio"""
        generated_files = []
        total = len(texts)

        # Cria diretório de saída
        os.makedirs(output_dir, exist_ok=True)

        for i, text in enumerate(texts):
            if text.strip():  # Só gera áudio para textos não vazios
                filename = f"{filename_prefix}_{i:03d}.wav"
                output_path = os.path.join(output_dir, filename)

                success = self.generate_audio(text, output_path, language)

                if success:
                    generated_files.append(output_path)
                    self.logger.info(f"Áudio {i+1}/{total} gerado: {filename}")
                else:
                    generated_files.append(None)
                    self.logger.warning(f"Falha ao gerar áudio {i+1}/{total}")
            else:
                generated_files.append(None)

            # Callback de progresso
            if progress_callback:
                progress = int((i + 1) / total * 100)
                progress_callback(f"Gerando áudio {i+1}/{total}", progress)

        successful = len([f for f in generated_files if f is not None])
        self.logger.info(f"✅ Gerados {successful}/{total} arquivos de áudio")

        return generated_files

    def get_available_providers(self) -> List[str]:
        """Retorna lista de providers disponíveis"""
        return list(self.providers.keys())

    def switch_primary_provider(self, provider_name: str) -> bool:
        """Muda o provider primário"""
        if provider_name in self.providers:
            self.primary_provider = provider_name
            self.logger.info(f"Provider TTS primário alterado para: {provider_name}")
            return True
        return False

    def get_available_voices(self, language: str = "pt-BR") -> Dict[str, List[str]]:
        """Retorna vozes disponíveis por provider"""
        voices = {}
        for name, provider in self.providers.items():
            try:
                voices[name] = provider.get_available_voices(language)
            except:
                voices[name] = ["default"]
        return voices

    def test_provider(self, provider_name: str) -> bool:
        """Testa se um provider está funcionando"""
        if provider_name not in self.providers:
            return False

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            provider = self.providers[provider_name]
            success = provider.generate_audio("Teste", tmp_path, "pt-BR")

            # Limpa arquivo temporário
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            return success
        except:
            return False

    def estimate_audio_duration(self, text: str, wpm: int = 150) -> float:
        """Estima duração do áudio em segundos baseado no texto"""
        if not text.strip():
            return 0.0