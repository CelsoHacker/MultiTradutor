# src/utils/backup_manager.py
"""
BackupManager - Sistema de backup automático
Autor: MultiTradutor Team
Versão: 1.0

Sistema de backup que NUNCA deixa você perder um arquivo original.
Regra de ouro do ROM hacking: SEMPRE faça backup antes de patchar!
"""

import os
import shutil
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class BackupManager:
    """
    Gerenciador de backup automático

    É como ter um sistema de save states - você pode sempre voltar
    ao estado original se algo der errado!
    """

    def __init__(self, backup_dir: str = "backups"):
        """
        Inicializa o gerenciador de backup

        Args:
            backup_dir: Diretório onde os backups serão armazenados
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

        # Arquivo de índice dos backups
        self.index_file = self.backup_dir / "backup_index.json"
        self.backup_index = self._load_backup_index()

    def _load_backup_index(self) -> Dict:
        """Carrega índice de backups existentes"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        return {
            'version': '1.0',
            'backups': {},
            'created': datetime.now().isoformat()
        }

    def _save_backup_index(self):
        """Salva índice de backups"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.backup_index, f, indent=2, ensure_ascii=False)

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash MD5 do arquivo para verificação de integridade"""
        hash_md5 = hashlib.md5()

        with open(file_path, 'rb') as f:
            # Lê em chunks para arquivos grandes
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def create_backup(self, source_path: str, backup_name: str = None) -> str:
        """
        Cria backup de um arquivo

        Args:
            source_path: Caminho do arquivo original
            backup_name: Nome personalizado do backup (opcional)

        Returns:
            str: Caminho do backup criado

        Raises:
            FileNotFoundError: Se arquivo original não existe
            PermissionError: Se não tem permissão para criar backup
        """
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {source_path}")

        if not source_path.is_file():
            raise ValueError(f"Caminho não é um arquivo: {source_path}")

        # Gera nome do backup
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"

        backup_path = self.backup_dir / backup_name

        # Garante que não sobrescreve backup existente
        counter = 1
        original_backup_path = backup_path
        while backup_path.exists():
            backup_path = original_backup_path.with_stem(f"{original_backup_path.stem}_{counter}")
            counter += 1

        try:
            # Copia arquivo
            shutil.copy2(source_path, backup_path)

            # Calcula hash para verificação
            file_hash = self._calculate_file_hash(str(source_path))

            # Registra no índice
            backup_info = {
                'original_path': str(source_path.absolute()),
                'backup_path': str(backup_path.absolute()),
                'original_name': source_path.name,
                'backup_name': backup_path.name,
                'created_at': datetime.now().isoformat(),
                'file_size': source_path.stat().st_size,
                'file_hash': file_hash,
                'backup_type': 'manual'
            }

            self.backup_index['backups'][str(backup_path)] = backup_info
            self._save_backup_index()

            print(f"💾 Backup criado: {backup_path.name}")
            return str(backup_path)

        except Exception as e:
            # Remove backup parcial se algo der errado
            if backup_path.exists():
                backup_path.unlink()
            raise e

    def create_auto_backup(self, source_path: str, operation: str = "translation") -> str:
        """
        Cria backup automático antes de operação

        Args:
            source_path: Arquivo a ser modificado
            operation: Tipo de operação (para organização)

        Returns:
            str: Caminho do backup
        """
        source_path = Path(source_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_name = f"{source_path.stem}_auto_{operation}_{timestamp}{source_path.suffix}"
        backup_path = self.create_backup(str(source_path), backup_name)

        # Marca como backup automático
        self.backup_index['backups'][backup_path]['backup_type'] = 'automatic'
        self.backup_index['backups'][backup_path]['operation'] = operation
        self._save_backup_index()

        return backup_path

    def restore_backup(self, backup_path: str, target_path: str = None) -> bool:
        """
        Restaura um backup

        Args:
            backup_path: Caminho do backup
            target_path: Onde restaurar (usa original se None)

        Returns:
            bool: True se sucesso
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            print(f"❌ Backup não encontrado: {backup_path}")
            return False

        # Determina destino
        if target_path:
            target_path = Path(target_path)
        else:
            # Usa caminho original do índice
            backup_info = self.backup_index['backups'].get(str(backup_path))
            if backup_info:
                target_path = Path(backup_info['original_path'])
            else:
                print(f"❌ Informações do backup não encontradas no índice")
                return False

        try:
            # Cria diretório se necessário
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copia backup para destino
            shutil.copy2(backup_path, target_path)

            print(f"✅ Backup restaurado: {target_path}")
            return True

        except Exception as e:
            print(f"💥 Erro ao restaurar backup: {e}")
            return False

    def list_backups(self, original_file: str = None) -> List[Dict]:
        """
        Lista backups disponíveis

        Args:
            original_file: Filtrar por arquivo original (opcional)

        Returns:
            List[Dict]: Lista de informações dos backups
        """
        backups = []

        for backup_path, info in self.backup_index['backups'].items():
            # Filtra por arquivo original se especificado
            if original_file:
                original_file_path = Path(original_file).absolute()
                backup_original_path = Path(info['original_path']).absolute()

                if original_file_path != backup_original_path:
                    continue

            # Verifica se backup ainda existe
            if Path(backup_path).exists():
                backups.append({
                    'backup_path': backup_path,
                    'original_path': info['original_path'],
                    'backup_name': info['backup_name'],
                    'created_at': info['created_at'],
                    'file_size': info['file_size'],
                    'backup_type': info.get('backup_type', 'unknown'),
                    'operation': info.get('operation', 'unknown')
                })

        # Ordena por data de criação (mais recente primeiro)
        backups.sort(key=lambda x: x['created_at'], reverse=True)

        return backups

    def cleanup_old_backups(self, keep_count: int = 10, keep_days: int = 30):
        """
        Remove backups antigos para economizar espaço

        Args:
            keep_count: Quantos backups manter por arquivo
            keep_days: Manter backups dos últimos N dias
        """
        print(f"🧹 Limpando backups antigos (manter {keep_count} por arquivo, {keep_days} dias)...")

        from collections import defaultdict

        # Agrupa backups por arquivo original
        backups_by_file = defaultdict(list)

        for backup_path, info in list(self.backup_index['backups'].items()):
            backup_file = Path(backup_path)

            # Remove entrada se backup não existe mais
            if not backup_file.exists():
                del self.backup_index['backups'][backup_path]
                continue

            backups_by_file[info['original_path']].append({
                'path': backup_path,
                'info': info,
                'created_at': datetime.fromisoformat(info['created_at'])
            })

        removed_count = 0
        cutoff_date = datetime.now() - timedelta(days=keep_days)

        for original_file, backups in backups_by_file.items():
            # Ordena por data (mais recente primeiro)
            backups.sort(key=lambda x: x['created_at'], reverse=True)

            for i, backup in enumerate(backups):
                should_remove = False

                # Remove se exceder contagem máxima
                if i >= keep_count:
                    should_remove = True

                # Remove se muito antigo (exceto os primeiros keep_count)
                if i >= keep_count and backup['created_at'] < cutoff_date:
                    should_remove = True

                if should_remove:
                    try:
                        Path(backup['path']).unlink()
                        del self.backup_index['backups'][backup['path']]
                        removed_count += 1
                        print(f"   🗑️ Removido: {Path(backup['path']).name}")
                    except Exception as e:
                        print(f"   ⚠️ Erro ao remover {backup['path']}: {e}")

        self._save_backup_index()
        print(f"✅ Limpeza concluída: {removed_count} backups removidos")

    def verify_backup_integrity(self, backup_path: str) -> bool:
        """
        Verifica integridade de um backup

        Args:
            backup_path: Caminho do backup

        Returns:
            bool: True se íntegro
        """
        backup_info = self.backup_index['backups'].get(backup_path)

        if not backup_info:
            print(f"❌ Backup não encontrado no índice: {backup_path}")
            return False

        if not Path(backup_path).exists():
            print(f"❌ Arquivo de backup não existe: {backup_path}")
            return False

        try:
            # Calcula hash atual
            current_hash = self._calculate_file_hash(backup_path)
            stored_hash = backup_info.get('file_hash')

            if current_hash == stored_hash:
                print(f"✅ Backup íntegro: {Path(backup_path).name}")
                return True
            else:
                print(f"⚠️ Backup corrompido: {Path(backup_path).name}")
                return False

        except Exception as e:
            print(f"💥 Erro ao verificar backup: {e}")
            return False

    def get_backup_stats(self) -> Dict:
        """Retorna estatísticas dos backups"""
        total_backups = len(self.backup_index['backups'])
        total_size = 0
        backup_types = {'automatic': 0, 'manual': 0, 'unknown': 0}

        for backup_path, info in self.backup_index['backups'].items():
            if Path(backup_path).exists():
                total_size += info.get('file_size', 0)
                backup_type = info.get('backup_type', 'unknown')
                backup_types[backup_type] = backup_types.get(backup_type, 0) + 1

        return {
            'total_backups': total_backups,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'backup_types': backup_types,
            'backup_directory': str(self.backup_dir)
        }


# Utilitários para facilitar uso
def quick_backup(file_path: str, backup_dir: str = "backups") -> str:
    """
    Função utilitária para backup rápido

    Args:
        file_path: Arquivo para backup
        backup_dir: Diretório de backup

    Returns:
        str: Caminho do backup criado
    """
    manager = BackupManager(backup_dir)
    return manager.create_backup(file_path)


def auto_backup_context(file_path: str, operation: str = "modification"):
    """
    Context manager para backup automático

    Usage:
        with auto_backup_context("meu_arquivo.exe", "translation"):
            # Modifica arquivo aqui
            modify_file("meu_arquivo.exe")
    """
    class AutoBackupContext:
        def __init__(self, file_path: str, operation: str):
            self.file_path = file_path
            self.operation = operation
            self.manager = BackupManager()
            self.backup_path = None

        def __enter__(self):
            self.backup_path = self.manager.create_auto_backup(self.file_path, self.operation)
            return self.backup_path

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                print(f"⚠️ Erro durante operação, backup disponível em: {self.backup_path}")
            else:
                print(f"✅ Operação concluída, backup em: {self.backup_path}")

    return AutoBackupContext(file_path, operation)


# Exemplo de uso
if __name__ == "__main__":
    manager = BackupManager()

    # Exemplo básico
    test_file = "exemplo.txt"

    # Cria arquivo de teste
    with open(test_file, 'w') as f:
        f.write("Conteúdo original")

    # Cria backup
    backup_path = manager.create_backup(test_file)

    # Lista backups
    backups = manager.list_backups()
    print(f"📊 Total de backups: {len(backups)}")

    # Mostra estatísticas
    stats = manager.get_backup_stats()
    print(f"📈 Estatísticas: {stats}")

    # Limpa arquivo de teste
    os.remove(test_file)
    print("🧪 Teste concluído!")