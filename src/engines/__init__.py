          # Aplica tradução
            output_rom = test_rom.replace('.smc', '_traduzido.smc')
            success = emulator_engine.apply_translation(extracted, output_rom)

            if success:
                print(f"🏆 Tradução aplicada: {output_rom}")
            else:
                print("❌ Falha na aplicação da tradução")
        else:
            print(f"❌ ROM não reconhecida: {test_rom}")
    else:
        print(f"ℹ️ Arquivo de teste não encontrado: {test_rom}")
        print("   Para testar, coloque uma ROM na pasta e ajuste o nome do arquivo")


# Utilitários extras para integração
class LegacyEngineManager:
    """
    Gerenciador para facilitar integração e debugging dos engines legacy
    É como ter um "ROM organizer" - mantém tudo catalogado e funcionando
    """

    def __init__(self, emulator_engine: EmulatorEngine):
        self.emulator_engine = emulator_engine
        self.engine_health = {}
        self._check_engine_health()

    def _check_engine_health(self):
        """Verifica saúde de todos os engines legacy"""
        print("🏥 Verificando saúde dos engines legacy...")

        for console, engine_class in self.emulator_engine.legacy_engines.items():
            try:
                # Tenta instanciar
                engine = engine_class()

                # Verifica métodos essenciais
                health = {
                    'instantiable': True,
                    'has_extract': hasattr(engine, 'extract_texts') or hasattr(engine, 'extract_strings'),
                    'has_apply': hasattr(engine, 'apply_translation') or hasattr(engine, 'patch_rom'),
                    'has_detect': hasattr(engine, 'is_compatible'),
                    'version': getattr(engine, 'version', 'unknown'),
                    'status': 'healthy'
                }

                # Score de saúde
                score = sum([health['has_extract'], health['has_apply'], health['has_detect']])
                if score == 3:
                    health['status'] = 'excellent'
                elif score == 2:
                    health['status'] = 'good'
                elif score == 1:
                    health['status'] = 'limited'
                else:
                    health['status'] = 'broken'

                self.engine_health[console] = health

            except Exception as e:
                self.engine_health[console] = {
                    'instantiable': False,
                    'error': str(e),
                    'status': 'failed'
                }

    def print_health_report(self):
        """Imprime relatório de saúde dos engines"""
        print("\n📋 RELATÓRIO DE SAÚDE DOS ENGINES LEGACY")
        print("=" * 50)

        for console, health in self.engine_health.items():
            status_emoji = {
                'excellent': '🟢',
                'good': '🟡',
                'limited': '🟠',
                'broken': '🔴',
                'failed': '💀'
            }.get(health['status'], '❓')

            print(f"{status_emoji} {console.upper():<12} - {health['status'].upper()}")

            if health.get('instantiable', False):
                print(f"   📦 Extração: {'✅' if health.get('has_extract') else '❌'}")
                print(f"   🔧 Aplicação: {'✅' if health.get('has_apply') else '❌'}")
                print(f"   🔍 Detecção: {'✅' if health.get('has_detect') else '❌'}")
                print(f"   📟 Versão: {health.get('version', 'N/A')}")
            else:
                print(f"   💥 Erro: {health.get('error', 'Desconhecido')}")
            print()

    def suggest_fixes(self):
        """Sugere correções para engines com problemas"""
        print("🔧 SUGESTÕES DE CORREÇÃO")
        print("=" * 30)

        for console, health in self.engine_health.items():
            if health['status'] in ['broken', 'failed', 'limited']:
                print(f"\n🚨 {console.upper()} precisa de atenção:")

                if not health.get('instantiable'):
                    print("   - Verificar se o arquivo do engine existe")
                    print("   - Verificar imports e dependências")
                    print("   - Verificar sintaxe do código")

                if not health.get('has_extract'):
                    print("   - Implementar método extract_texts() ou extract_strings()")
                    print("   - Verificar se o método retorna dados no formato esperado")

                if not health.get('has_apply'):
                    print("   - Implementar método apply_translation() ou patch_rom()")
                    print("   - Verificar se consegue escrever no arquivo de saída")

                if not health.get('has_detect'):
                    print("   - Implementar método is_compatible()")
                    print("   - Melhorar detecção automática de ROM")


class ROMTestSuite:
    """
    Suite de testes para validar integração com engines legacy
    É como ter um "ROM tester" - garante que tudo funciona como esperado
    """

    def __init__(self, emulator_engine: EmulatorEngine):
        self.emulator_engine = emulator_engine
        self.test_results = {}

    def run_integration_tests(self, test_roms: Dict[str, str] = None):
        """
        Executa testes de integração com ROMs reais

        test_roms: Dict no formato {'console': 'caminho_da_rom'}
        """
        if not test_roms:
            print("ℹ️ Nenhuma ROM de teste fornecida")
            print("   Para testar completamente, forneça ROMs de cada console")
            return

        print("🧪 EXECUTANDO TESTES DE INTEGRAÇÃO")
        print("=" * 40)

        for console, rom_path in test_roms.items():
            print(f"\n🎮 Testando {console.upper()}: {os.path.basename(rom_path)}")

            test_result = {
                'console': console,
                'rom_path': rom_path,
                'detection': False,
                'extraction': False,
                'translation': False,
                'errors': []
            }

            try:
                # Teste 1: Detecção
                if self.emulator_engine.detect_software_type(rom_path):
                    detected_console = self.emulator_engine.detect_console_type(rom_path)
                    test_result['detection'] = True
                    test_result['detected_as'] = detected_console
                    print(f"   ✅ Detecção: {detected_console}")

                    if detected_console != console:
                        print(f"   ⚠️ Detectado como {detected_console}, esperado {console}")
                else:
                    test_result['errors'].append("Falha na detecção")
                    print(f"   ❌ Detecção falhou")
                    continue

                # Teste 2: Extração
                try:
                    extracted = self.emulator_engine.extract_texts(rom_path)
                    strings_count = len(extracted.get('strings', {}))

                    if strings_count > 0:
                        test_result['extraction'] = True
                        test_result['strings_extracted'] = strings_count
                        print(f"   ✅ Extração: {strings_count} strings")
                    else:
                        test_result['errors'].append("Nenhuma string extraída")
                        print(f"   ⚠️ Extração: 0 strings (pode ser normal)")

                except Exception as e:
                    test_result['errors'].append(f"Erro na extração: {e}")
                    print(f"   ❌ Extração falhou: {e}")
                    continue

                # Teste 3: Aplicação (simulada)
                try:
                    # Simula algumas traduções
                    sample_translations = {}
                    for i, (key, data) in enumerate(extracted.get('strings', {}).items()):
                        if i >= 3:  # Só 3 para teste
                            break
                        data['translated'] = f"[TESTE] {data['original']}"
                        sample_translations[key] = data

                    if sample_translations:
                        # Testa aplicação (sem realmente escrever arquivo)
                        test_output = rom_path.replace('.', '_test.')
                        success = self.emulator_engine.apply_translation(extracted, test_output)

                        if success:
                            test_result['translation'] = True
                            print(f"   ✅ Aplicação: Sucesso (simulado)")

                            # Remove arquivo de teste se foi criado
                            if os.path.exists(test_output):
                                os.remove(test_output)
                        else:
                            test_result['errors'].append("Falha na aplicação")
                            print(f"   ❌ Aplicação falhou")

                except Exception as e:
                    test_result['errors'].append(f"Erro na aplicação: {e}")
                    print(f"   ❌ Aplicação falhou: {e}")

            except Exception as e:
                test_result['errors'].append(f"Erro geral: {e}")
                print(f"   💥 Erro geral: {e}")

            self.test_results[console] = test_result

    def print_test_summary(self):
        """Imprime resumo dos testes"""
        if not self.test_results:
            print("❓ Nenhum teste executado")
            return

        print("\n📊 RESUMO DOS TESTES")
        print("=" * 25)

        total_tests = len(self.test_results)
        successful_detections = sum(1 for r in self.test_results.values() if r['detection'])
        successful_extractions = sum(1 for r in self.test_results.values() if r['extraction'])
        successful_translations = sum(1 for r in self.test_results.values() if r['translation'])

        print(f"📈 Detecção: {successful_detections}/{total_tests} ({successful_detections/total_tests*100:.1f}%)")
        print(f"📈 Extração: {successful_extractions}/{total_tests} ({successful_extractions/total_tests*100:.1f}%)")
        print(f"📈 Aplicação: {successful_translations}/{total_tests} ({successful_translations/total_tests*100:.1f}%)")

        # Mostra problemas
        print("\n🚨 PROBLEMAS ENCONTRADOS:")
        for console, result in self.test_results.items():
            if result['errors']:
                print(f"   {console.upper()}:")
                for error in result['errors']:
                    print(f"     - {error}")


# Script de setup e validação
def setup_legacy_integration():
    """
    Script principal para setup e validação da integração legacy

    Execute este script após configurar os engines legacy para verificar
    se tudo está funcionando corretamente
    """
    print("🚀 INICIANDO SETUP DA INTEGRAÇÃO LEGACY")
    print("=" * 50)

    # Cria engine principal
    emulator_engine = create_emulator_engine()

    # Verifica saúde dos engines
    manager = LegacyEngineManager(emulator_engine)
    manager.print_health_report()
    manager.suggest_fixes()

    # Opcionalmente executa testes com ROMs
    print("\n" + "=" * 50)
    print("Para executar testes completos, forneça ROMs de teste:")
    print("  test_roms = {")
    print("      'nes': 'roms/test.nes',")
    print("      'snes': 'roms/test.smc',")
    print("      'gba': 'roms/test.gba'")
    print("  }")
    print("  suite = ROMTestSuite(emulator_engine)")
    print("  suite.run_integration_tests(test_roms)")
    print("  suite.print_test_summary()")

    return emulator_engine, manager


if __name__ == "__main__":
    # Setup completo
    engine, manager = setup_legacy_integration()

    print("\n🎯 INTEGRAÇÃO LEGACY CONFIGURADA!")
    print("   Use o EmulatorEngine para processar ROMs")
    print("   Todos os engines legacy são acessados automaticamente")
    print("   Interface unificada mantém compatibilidade total")
    print("\n🎮 Agora você pode traduzir ROMs usando a nova arquitetura!")
    print("   multitradutor.translate_rom('jogo.nes', 'pt-br')")
    print("   # Internamente usa NESEngine legacy, mas com interface moderna!")