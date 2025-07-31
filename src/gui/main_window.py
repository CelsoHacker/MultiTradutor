# -*- coding: utf-8 -*-
"""
MultiTradutor - Interface Gráfica Principal
Janela principal com PyQt6 para o sistema de tradução automática
"""
# Importa o gerenciador do núcleo
# Importa a FUNÇÃO que escolhe o motor, do arquivo correto
from src.core.engine_manager import get_engine_for_source
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QGridLayout, QPushButton, QLabel,
                            QTextEdit, QFileDialog, QProgressBar, QFrame,
                            QGroupBox, QMessageBox, QStatusBar, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor

# Importar os módulos já criados
try:
    from motor_traducao import MotorTraducao
    from motor_audio import MotorAudio
    from gerenciador_arquivos import GerenciadorArquivos
except ImportError:
    print("⚠️ Módulos não encontrados. Certifique-se de que estão no mesmo diretório.")


class WorkerThread(QThread):
    """Thread para operações em background"""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, operation, *args):
        super().__init__()
        self.operation = operation
        self.args = args

    def run(self):
        try:
            if self.operation == "extrair_texto":
                self.extrair_texto_processo()
            elif self.operation == "traduzir":
                self.traduzir_processo()
            elif self.operation == "inserir_traducao":
                self.inserir_traducao_processo()
        except Exception as e:
            self.error.emit(f"Erro: {str(e)}")
        finally:
            self.finished.emit()

    def extrair_texto_processo(self):
        self.log_message.emit("🔍 Iniciando extração de texto...")
        self.progress.emit(25)
        # Simular processo
        import time
        time.sleep(1)
        self.progress.emit(50)
        self.log_message.emit("📄 Texto extraído com sucesso!")
        self.progress.emit(100)

    def traduzir_processo(self):
        self.log_message.emit("🤖 Iniciando tradução com IA...")
        self.progress.emit(33)
        import time
        time.sleep(1.5)
        self.progress.emit(66)
        self.log_message.emit("✅ Tradução concluída!")
        self.progress.emit(100)

    def inserir_traducao_processo(self):
        self.log_message.emit("📝 Inserindo tradução no arquivo...")
        self.progress.emit(50)
        import time
        time.sleep(1)
        self.log_message.emit("💾 Tradução inserida com sucesso!")
        self.progress.emit(100)


class JanelaPrincipal(QMainWindow):
    """Janela principal do MultiTradutor"""

    def __init__(self):
        super().__init__()
        self.arquivo_selecionado = None
        self.worker_thread = None

        # Inicializar módulos
        self.gerenciador = GerenciadorArquivos()
        self.motor_traducao = MotorTraducao()
        self.motor_audio = MotorAudio()

        self.configurar_janela()
        self.criar_interface()
        self.aplicar_estilos()
        self.conectar_sinais()

    def configurar_janela(self):
        """Configurações básicas da janela"""
        self.setWindowTitle("MultiTradutor")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)

        # Ícone da janela (se disponível)
        try:
            self.setWindowIcon(QIcon("assets/icon.png"))
        except:
            pass

    def criar_interface(self):
        """Criar todos os elementos da interface"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        self.criar_header(main_layout)

        # Splitter para dividir em duas colunas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Coluna esquerda - Controles
        self.criar_coluna_controles(splitter)

        # Coluna direita - Console e detector
        self.criar_coluna_output(splitter)

        # Barra de status
        self.criar_barra_status()

        # Definir proporções do splitter
        splitter.setSizes([400, 600])

    def criar_header(self, parent_layout):
        """Criar o cabeçalho da aplicação"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.Box)
        header_frame.setFixedHeight(80)

        header_layout = QHBoxLayout(header_frame)

        # Título
        titulo = QLabel("🌐 MultiTradutor")
        titulo.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        titulo.setStyleSheet("color: #2E86AB; margin: 10px;")

        # Subtítulo
        subtitulo = QLabel("Sistema Profissional de Tradução Automática")
        subtitulo.setFont(QFont("Segoe UI", 11))
        subtitulo.setStyleSheet("color: #666; margin-left: 15px;")

        titulo_layout = QVBoxLayout()
        titulo_layout.addWidget(titulo)
        titulo_layout.addWidget(subtitulo)

        header_layout.addLayout(titulo_layout)
        header_layout.addStretch()

        parent_layout.addWidget(header_frame)

    def criar_coluna_controles(self, parent_splitter):
        """Criar a coluna com controles principais"""
        controles_widget = QWidget()
        controles_layout = QVBoxLayout(controles_widget)

        # Grupo: Seleção de Arquivo
        self.criar_grupo_arquivo(controles_layout)

        # Grupo: Operações
        self.criar_grupo_operacoes(controles_layout)

        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 6px;
            }
        """)
        controles_layout.addWidget(self.progress_bar)

        controles_layout.addStretch()
        parent_splitter.addWidget(controles_widget)

    def criar_grupo_arquivo(self, parent_layout):
        """Criar grupo de seleção de arquivo"""
        grupo_arquivo = QGroupBox("📁 Seleção de Arquivo")
        grupo_arquivo.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout_arquivo = QVBoxLayout(grupo_arquivo)

        # Label do arquivo selecionado
        self.label_arquivo = QLabel("Nenhum arquivo selecionado")
        self.label_arquivo.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f8f9fa;
                border: 2px dashed #ddd;
                border-radius: 8px;
                color: #666;
            }
        """)
        self.label_arquivo.setWordWrap(True)

        # Botão selecionar arquivo
        self.btn_selecionar = QPushButton("📂 Selecionar Arquivo")
        self.btn_selecionar.setFixedHeight(45)

        layout_arquivo.addWidget(self.label_arquivo)
        layout_arquivo.addWidget(self.btn_selecionar)

        parent_layout.addWidget(grupo_arquivo)

    def criar_grupo_operacoes(self, parent_layout):
        """Criar grupo de operações principais"""
        grupo_ops = QGroupBox("⚙️ Operações")
        grupo_ops.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout_ops = QVBoxLayout(grupo_ops)

        # Botões de operação
        self.btn_extrair = QPushButton("🔍 Extrair Texto")
        self.btn_traduzir = QPushButton("🤖 Traduzir com IA")
        self.btn_inserir = QPushButton("📝 Inserir Tradução")

        # Configurar botões
        botoes = [self.btn_extrair, self.btn_traduzir, self.btn_inserir]
        for btn in botoes:
            btn.setFixedHeight(45)
            btn.setEnabled(False)
            layout_ops.addWidget(btn)

        parent_layout.addWidget(grupo_ops)

    def criar_coluna_output(self, parent_splitter):
        """Criar coluna com output e console"""
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)

        # Console de logs
        self.criar_console_logs(output_layout)

        # Detector de ROMs
        self.criar_detector_roms(output_layout)

        parent_splitter.addWidget(output_widget)

    def criar_console_logs(self, parent_layout):
        """Criar console de logs"""
        grupo_console = QGroupBox("📋 Console de Logs")
        grupo_console.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout_console = QVBoxLayout(grupo_console)

        self.console_logs = QTextEdit()
        self.console_logs.setFont(QFont("Consolas", 10))
        self.console_logs.setReadOnly(True)
        self.console_logs.setMinimumHeight(200)

        # Mensagem inicial
        self.console_logs.append("🚀 MultiTradutor iniciado com sucesso!")
        self.console_logs.append("💡 Selecione um arquivo para começar...")
        self.console_logs.append("="*50)

        layout_console.addWidget(self.console_logs)
        parent_layout.addWidget(grupo_console)

    def criar_detector_roms(self, parent_layout):
        """Criar detector de ROMs"""
        grupo_detector = QGroupBox("🕹️ Detector de ROMs")
        grupo_detector.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        grupo_detector.setMaximumHeight(150)

        layout_detector = QVBoxLayout(grupo_detector)

        self.detector_output = QTextEdit()
        self.detector_output.setFont(QFont("Consolas", 9))
        self.detector_output.setReadOnly(True)
        self.detector_output.setMaximumHeight(80)
        self.detector_output.append("Aguardando arquivo para detecção...")

        layout_detector.addWidget(self.detector_output)
        parent_layout.addWidget(grupo_detector)

    def criar_barra_status(self):
        """Criar barra de status"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Pronto para uso")

    def aplicar_estilos(self):
        """Aplicar estilos visuais"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2E86AB;
            }

            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: #0056b3;
            }

            QPushButton:pressed {
                background-color: #004085;
            }

            QPushButton:disabled {
                background-color: #6c757d;
                color: #fff;
            }

            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: #2d3748;
                color: #e2e8f0;
                font-family: 'Consolas', 'Monaco', monospace;
                padding: 8px;
            }

            QFrame {
                background-color: white;
                border-radius: 10px;
            }
        """)

    def conectar_sinais(self):
        """Conectar sinais dos botões"""
        self.btn_selecionar.clicked.connect(self.selecionar_arquivo)
        self.btn_extrair.clicked.connect(self.extrair_texto)
        self.btn_traduzir.clicked.connect(self.traduzir_texto)
        self.btn_inserir.clicked.connect(self.inserir_traducao)

    def selecionar_arquivo(self):
        """Selecionar arquivo para tradução"""
        arquivo, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar arquivo para tradução",
            "",
            "Todos os arquivos (*);;ROMs (*.rom *.nes *.smc);;Textos (*.txt *.json)"
        )

        if arquivo:
            self.arquivo_selecionado = arquivo
            nome_arquivo = Path(arquivo).name
            self.label_arquivo.setText(f"📄 {nome_arquivo}")
            self.label_arquivo.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #d4edda;
                    border: 2px solid #c3e6cb;
                    border-radius: 8px;
                    color: #155724;
                }
            """)

            # Habilitar botões
            self.btn_extrair.setEnabled(True)

            # Log
            self.adicionar_log(f"✅ Arquivo selecionado: {nome_arquivo}")
            self.status_bar.showMessage(f"Arquivo carregado: {nome_arquivo}")

            # Detectar tipo de arquivo
            self.detectar_arquivo()

    def detectar_arquivo(self):
        """Detectar tipo de arquivo selecionado"""
        if not self.arquivo_selecionado:
            return

        try:
            arquivo_path = Path(self.arquivo_selecionado)
            extensao = arquivo_path.suffix.lower()
            tamanho = arquivo_path.stat().st_size

            self.detector_output.clear()
            self.detector_output.append(f"📁 Arquivo: {arquivo_path.name}")
            self.detector_output.append(f"📊 Tamanho: {tamanho:,} bytes")
            self.detector_output.append(f"🔖 Extensão: {extensao}")

            if extensao in ['.rom', '.nes', '.smc']:
                self.detector_output.append("🕹️ Tipo: ROM de videogame detectada")
                self.detector_output.append("✅ Suporte a OCR ativado")
            elif extensao in ['.txt', '.json']:
                self.detector_output.append("📄 Tipo: Arquivo de texto")
                self.detector_output.append("✅ Processamento direto disponível")
            else:
                self.detector_output.append("❓ Tipo: Arquivo genérico")
                self.detector_output.append("⚠️ Tentativa de detecção automática")

        except Exception as e:
            self.detector_output.append(f"❌ Erro na detecção: {str(e)}")

    def extrair_texto(self):
        """Extrair texto do arquivo"""
        if not self.arquivo_selecionado:
            QMessageBox.warning(self, "Aviso", "Selecione um arquivo primeiro!")
            return

        self.iniciar_operacao("extrair_texto")

    def traduzir_texto(self):
        """Traduzir texto extraído"""
        self.iniciar_operacao("traduzir")

    def inserir_traducao(self):
        """Inserir tradução no arquivo"""
        self.iniciar_operacao("inserir_traducao")

    def iniciar_operacao(self, operacao):
        """Iniciar uma operação em background"""
        if self.worker_thread and self.worker_thread.isRunning():
            return

        # Configurar interface para operação
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.desabilitar_botoes(True)

        # Criar e iniciar thread
        self.worker_thread = WorkerThread(operacao, self.arquivo_selecionado)
        self.worker_thread.finished.connect(self.operacao_finalizada)
        self.worker_thread.progress.connect(self.progress_bar.setValue)
        self.worker_thread.log_message.connect(self.adicionar_log)
        self.worker_thread.error.connect(self.mostrar_erro)
        self.worker_thread.start()

    def operacao_finalizada(self):
        """Callback quando operação é finalizada"""
        self.progress_bar.setVisible(False)
        self.desabilitar_botoes(False)

        # Habilitar próximos botões conforme necessário
        if self.btn_extrair.text().startswith("🔍"):
            self.btn_traduzir.setEnabled(True)
        if self.btn_traduzir.text().startswith("🤖"):
            self.btn_inserir.setEnabled(True)

        self.status_bar.showMessage("Operação concluída")

    def desabilitar_botoes(self, desabilitar):
        """Habilitar/desabilitar botões durante operações"""
        self.btn_extrair.setEnabled(not desabilitar)
        self.btn_traduzir.setEnabled(not desabilitar and self.arquivo_selecionado is not None)
        self.btn_inserir.setEnabled(not desabilitar and self.arquivo_selecionado is not None)
        self.btn_selecionar.setEnabled(not desabilitar)

    def adicionar_log(self, mensagem):
        """Adicionar mensagem ao console de logs"""
        self.console_logs.append(f"[{self.obter_timestamp()}] {mensagem}")
        self.console_logs.ensureCursorVisible()

    def mostrar_erro(self, erro):
        """Mostrar mensagem de erro"""
        self.adicionar_log(f"❌ {erro}")
        QMessageBox.critical(self, "Erro", erro)

    def obter_timestamp(self):
        """Obter timestamp atual"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def closeEvent(self, event):
        """Evento de fechamento da janela"""
        if self.worker_thread and self.worker_thread.isRunning():
            resposta = QMessageBox.question(
                self,
                "Confirmar saída",
                "Uma operação está em andamento. Deseja realmente sair?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if resposta == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.worker_thread.terminate()
            self.worker_thread.wait()

        event.accept()


def main():
    """Função principal"""
    app = QApplication(sys.argv)

    # Configurar aplicação
    app.setApplicationName("MultiTradutor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("MultiTradutor Team")

    # Criar e mostrar janela
    janela = JanelaPrincipal()
    janela.show()

    # Executar aplicação
    sys.exit(app.exec())


if __name__ == "__main__":
    main()