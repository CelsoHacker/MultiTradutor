#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine Manager - Gerenciador de Engines de Console
=================================================
Responsável por carregar e gerenciar engines específicas de cada console
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class EngineManager:
    """Gerenciador centralizado das engines de console"""

    def __init__(self, engines_path: str = "engines_legacy"):
        self.engines_path = Path(engines_path)
        self.loaded_engines: Dict[str, Any] = {}
        self.available_engines: List[str] = []
        self.logger = logging.getLogger(__name__)

        self._discover_engines()

    def _discover_engines(self):
        """Descobre automaticamente engines disponíveis"""
        if not self.engines_path.exists():
            self.logger.warning(f"Diretório de engines não encontrado: {self.engines_path}")
            return

        self.available_engines = []

        # Procura por arquivos Python que terminam com '_engine.py'
        for engine_file in self.engines_path.glob("*_engine.py"):
            engine_name = engine_file.stem.replace("_engine", "").upper()
            self.available_engines.append(engine_name)
            self.logger.info(f"Engine descoberta: {engine_name} ({engine_file.name})")

        self.logger.info(f"Total de engines encontradas: {len(self.available_engines)}")

    def get_available_engines(self) -> List[str]:
        """Retorna lista de engines disponíveis"""
        return self.available_engines.copy()

    def load_engine(self, engine_name: str) -> Optional[Any]:
        """
        Carrega uma engine específica

        Args:
            engine_name: Nome da engine (ex: 'SNES', 'PS1')

        Returns:
            Instância da engine carregada ou None se falhar
        """
        engine_name = engine_name.upper()

        # Verifica se já está carregada
        if engine_name in self.loaded_engines:
            self.logger.info(f"Engine {engine_name} já carregada")
            return self.loaded_engines[engine_name]

        # Procura o arquivo da engine
        engine_file = self.engines_path / f"{engine_name.lower()}_engine.py"

        if not engine_file.exists():
            self.logger.error(f"Arquivo da engine não encontrado: {engine_file}")
            return None

        try:
            # Carrega o módulo dinamicamente
            spec = importlib.util.spec_from_file_location(
                f"{engine_name.lower()}_engine",
                engine_file
            )

            if spec is None:
                raise ImportError(f"Não foi possível carregar spec para {engine_file}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Procura pela classe principal da engine
            engine_class = self._find_engine_class(module, engine_name)

            if engine_class is None:
                raise ImportError(f"Classe da engine não encontrada em {engine_file}")

            # Instancia a engine
            engine_instance = engine_class()
            self.loaded_engines[engine_name] = engine_instance

            self.logger.info(f"✅ Engine {engine_name} carregada com sucesso")
            return engine_instance

        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar engine {engine_name}: {str(e)}")
            return None

    def _find_engine_class(self, module, engine_name: str):
        """Encontra a classe principal da engine no módulo"""
        # Possíveis nomes de classe
        possible_names = [
            f"{engine_name}Engine",
            f"{engine_name}Rom",
            f"{engine_name}Handler",
            "Engine",
            "RomEngine",
            "MainEngine"
        ]

        for class_name in possible_names:
            if hasattr(module, class_name):
                return getattr(module, class_name)

        # Se não encontrou, procura por qualquer classe que pareça ser uma engine
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                not attr_name.startswith('_') and
                (hasattr(attr, 'analyze_file') or
                 hasattr(attr, 'extract_texts') or
                 hasattr(attr, 'insert_texts'))):
                return attr

        return None

    def unload_engine(self, engine_name: str):
        """Descarrega uma engine da memória"""
        engine_name = engine_name.upper()
        if engine_name in self.loaded_engines:
            del self.loaded_engines[engine_name]
            self.logger.info(f"Engine {engine_name} descarregada")

    def unload_all_engines(self):
        """Descarrega todas as engines"""
        self.loaded_engines.clear()
        self.logger.info("Todas as engines foram descarregadas")

    def get_engine_info(self, engine_name: str) -> Dict[str, Any]:
        """
        Retorna informações sobre uma engine

        Args:
            engine_name: Nome da engine

        Returns:
            Dicionário com informações da engine
        """
        engine_name = engine_name.upper()
        info = {
            "name": engine_name,
            "loaded": engine_name in self.loaded_engines,
            "file_exists": False,
            "supported_formats": [],
            "description": ""
        }

        # Verifica se arquivo existe
        engine_file = self.engines_path / f"{engine_name.lower()}_engine.py"
        info["file_exists"] = engine_file.exists()

        # Se carregada, obtém informações adicionais
        if engine_name in self.loaded_engines:
            engine = self.loaded_engines[engine_name]

            # Tenta obter formatos suportados
            if hasattr(engine, 'SUPPORTED_FORMATS'):
                info["supported_formats"] = engine.SUPPORTED_FORMATS
            elif hasattr(engine, 'supported_formats'):
                info["supported_formats"] = engine.supported_formats

            # Tenta obter descrição
            if hasattr(engine, 'DESCRIPTION'):
                info["description"] = engine.DESCRIPTION
            elif hasattr(engine, 'description'):
                info["description"] = engine.description
            elif engine.__doc__:
                info["description"] = engine.__doc__.strip()

        return info


class BaseEngine:
    """
    Classe base para todas as engines
    Define a interface padrão que suas engines existentes devem implementar
    """

    SUPPORTED_FORMATS = []
    DESCRIPTION = ""

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analisa um arquivo ROM/programa

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Dicionário com informações da análise
        """
        raise NotImplementedError("Método analyze_file deve ser implementado pela engine")

    def extract_texts(self, file_path: str) -> List[str]:
        """
        Extrai textos do arquivo

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Lista de strings com os textos extraídos
        """
        raise NotImplementedError("Método extract_texts deve ser implementado pela engine")

    def insert_texts(self, input_path: str, output_path: str, texts: List[str]) -> bool:
        """
        Insere textos traduzidos no arquivo

        Args:
            input_path: Arquivo original
            output_path: Arquivo de saída
            texts: Lista de textos traduzidos

        Returns:
            True se sucesso, False caso contrário
        """
        raise NotImplementedError("Método insert_texts deve ser implementado pela engine")

    def validate_file(self, file_path: str) -> bool:
        """
        Valida se o arquivo é compatível com esta engine

        Args:
            file_path: Caminho para o arquivo

        Returns:
            True se compatível, False caso contrário
        """
        if not os.path.exists(file_path):
            return False

        # Validação básica por extensão
        if self.SUPPORTED_FORMATS:
            file_ext = Path(file_path).suffix.lower()
            return file_ext in [fmt.lower() for fmt in self.SUPPORTED_FORMATS]

        return True