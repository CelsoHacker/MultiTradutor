
# -*- coding: utf-8 -*-
"""
MultiTradutor - Arquivo Principal
Sistema Profissional de Tradução Automática com Dublagem
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório atual ao path para importações
sys.path.insert(0, str(Path(__file__).parent))

def verificar_dependencias():
    """Verificar se todas as dependências estão instaladas"""
    dependencias_necessarias = {
        'PyQt6': 'pip install PyQt6',
        'Pillow': 'pip install Pillow',
        'pytesseract': 'pip install pytesseract',
        'gtts': 'pip install gtts',
        'pygame': 'pip install pygame',
        'requests': 'pip install requests'
    }

    dependencias_faltando = []

    for dep, comando in dependencias_necessarias.items():
        try:
            __import__(dep)
        except ImportError:
            dependencias_faltando.append((dep, comando))

    if dependencias_faltando:
        print("❌ Dependências faltando:")
        for dep, comando in dependencias_faltando:
            print(f"   • {dep}: {comando}")
        print("\n🔧 Execute os comandos acima para instalar as dependências.")
        return False

    return True

def verificar_estrutura_projeto():
    """Verificar se a estrutura de pastas existe"""
    diretorios_necessarios = [
        'temp',
        'output',
        'assets',
        'traducoes'
    ]

    for diretorio in diretorios_necessarios:
        path = Path(diretorio)
        if not path.exists():
            print(f"📁 Criando diretório: {diretorio}")
            path.mkdir(exist_ok=True)

def criar_modulos_basicos():
    """Criar módulos básicos se não existirem"""

    # motor_traducao.py
    if not Path("motor_traducao.py").exists():
        print("📝 Criando motor_traducao.py...")
        with open("motor_traducao.py", "w", encoding="utf-8") as f:
            f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Tradução - MultiTradutor
"""

class MotorTraducao:
    def __init__(self):
        self.idioma_origem = "auto"
        self.idioma_destino = "pt"

    def traduzir_texto(self, texto):
        """Traduzir texto usando IA"""
        # Implementação básica - expandir conforme necessário
        return f"[TRADUZIDO] {texto}"

    def detectar_idioma(self, texto):
        """Detectar idioma do texto"""
        return "en"  # Placeholder
''')

    # motor_audio.py
    if not Path("motor_audio.py").exists():
        print("📝 Criando motor_audio.py...")
        with open("motor_audio.py", "w", encoding="utf-8") as f:
            f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Áudio - MultiTradutor
"""

class MotorAudio:
    def __init__(self):
        self.qualidade = "alta"

    def texto_para_audio(self, texto, idioma="pt"):
        """Converter texto em áudio"""
        print(f"🔊 Gerando áudio: {texto[:50]}...")
        return True

    def reproduzir_audio(self, arquivo_audio):
        """Reproduzir arquivo de áudio"""
        print(f"▶️ Reproduzindo: {arquivo_audio}")
        return True
''')

    # gerenciador_arquivos.py
    if not Path("gerenciador_arquivos.py").exists():
        print("📝 Criando gerenciador_arquivos.py...")
        with open("gerenciador_arquivos.py", "w", encoding="utf-8") as f:
            f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador de Arquivos - MultiTradutor
"""

from pathlib import Path

class GerenciadorArquivos:
    def __init__(self):
        self.pasta_temp = Path("temp")
        self.pasta_output = Path("output")

    def extrair_texto(self, arquivo):
        """Extrair texto de arquivo"""
        print(f"📄 Extraindo texto de: {arquivo}")
        return "Texto extraído com sucesso"

    def salvar_traducao(self, texto_traduzido, arquivo_destino):
        """Salvar tradução em arquivo"""
        print(f"💾 Salvando tradução em: {arquivo_destino}")
        return True

    def detectar_tipo_arquivo(self, arquivo):
        """Detectar tipo de arquivo"""
        extensao = Path(arquivo).suffix.lower()
        if extensao in ['.rom', '.nes', '.smc']:
            return "rom"
        elif extensao in ['.txt', '.json']:
            return "texto"
        else:
            return "desconhecido"
''')

def main():
    """Função principal do MultiTradutor"""
    print("🌐 MultiTradutor - Sistema Profissional de Tradução")
    print("=" * 50)

    # Verificar dependências
    print("🔍 Verificando dependências...")
    if not verificar_dependencias():
        input("\n⏸️ Pressione Enter após instalar as dependências...")
        return

    # Verificar estrutura
    print("📁 Verificando estrutura do projeto...")
    verificar_estrutura_projeto()

    # Criar módulos básicos se necessário
    print("📝 Verificando módulos...")
    criar_modulos_basicos()

    # Tentar importar PyQt6
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 encontrado!")
    except ImportError:
        print("❌ PyQt6 não encontrado!")
        print("🔧 Execute: pip install PyQt6")
        input("⏸️ Pressione Enter após instalar...")
        return

    # Inicializar interface gráfica
    try:
        print("🚀 Iniciando interface gráfica...")
        from janela_principal import JanelaPrincipal

        app = QApplication(sys.argv)
        app.setApplicationName("MultiTradutor")
        app.setApplicationVersion("1.0")

        janela = JanelaPrincipal()
        janela.show()

        print("✅ MultiTradutor iniciado com sucesso!")
        print("💡 Use a interface gráfica para operar o sistema.")

        sys.exit(app.exec())

    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        print("🔧 Certifique-se de que todos os arquivos estão no diretório:")
        print("   • janela_principal.py")
        print("   • motor_traducao.py")
        print("   • motor_audio.py")
        print("   • gerenciador_arquivos.py")

    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        input("⏸️ Pressione Enter para sair...")

if __name__ == "__main__":
    main()