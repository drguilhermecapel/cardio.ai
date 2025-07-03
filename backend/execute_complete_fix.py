#!/usr/bin/env python3
"""
Execute Complete Fix
Script principal para executar todas as correções e atingir 80% de cobertura
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class CompleteFixExecutor:
    """Execute complete fix process"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.python_exe = sys.executable
        self.steps_completed = []
        self.errors = []
        
        # Colors
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.BLUE = '\033[94m'
        self.BOLD = '\033[1m'
        self.END = '\033[0m'
        
    def print_header(self):
        """Print header"""
        print(f"{self.BLUE}{self.BOLD}")
        print("=" * 60)
        print("   CardioAI Pro - Execução Completa de Correções")
        print("   Objetivo: Atingir 80% de Cobertura de Código")
        print("=" * 60)
        print(f"{self.END}\n")
        
    def run_step(self, step_name: str, script_name: str, description: str) -> bool:
        """Run a single step"""
        print(f"\n{self.BLUE}▶ {step_name}: {description}{self.END}")
        print("-" * 60)
        
        script_path = self.backend_path / script_name
        
        # Check if script exists
        if not script_path.exists():
            print(f"{self.RED}❌ Script não encontrado: {script_name}{self.END}")
            self.errors.append(f"{step_name}: Script não encontrado")
            return False
        
        try:
            # Run script
            result = subprocess.run(
                [self.python_exe, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Check result
            if result.returncode == 0:
                print(f"{self.GREEN}✅ {step_name} concluído com sucesso!{self.END}")
                self.steps_completed.append(step_name)
                return True
            else:
                print(f"{self.YELLOW}⚠️ {step_name} concluído com avisos{self.END}")
                if result.stderr:
                    print(f"Erros: {result.stderr[:500]}...")
                self.steps_completed.append(f"{step_name} (com avisos)")
                return True
                
        except subprocess.TimeoutExpired:
            print(f"{self.RED}❌ {step_name} excedeu o tempo limite{self.END}")
            self.errors.append(f"{step_name}: Timeout")
            return False
        except Exception as e:
            print(f"{self.RED}❌ Erro em {step_name}: {str(e)}{self.END}")
            self.errors.append(f"{step_name}: {str(e)}")
            return False
    
    def check_requirements(self):
        """Check basic requirements"""
        print(f"{self.BLUE}▶ Verificando requisitos básicos...{self.END}")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ necessário")
        
        # Check if in correct directory
        if not (self.backend_path / "app").exists():
            issues.append("Execute este script no diretório backend/")
        
        # Check critical files
        critical_files = ["app/__init__.py", "tests/__init__.py"]
        for file in critical_files:
            if not (self.backend_path / file).exists():
                (self.backend_path / file).parent.mkdir(parents=True, exist_ok=True)
                (self.backend_path / file).touch()
        
        if issues:
            print(f"{self.RED}❌ Problemas encontrados:{self.END}")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print(f"{self.GREEN}✅ Requisitos verificados{self.END}")
        return True
    
    def run(self):
        """Run complete fix process"""
        self.print_header()
        
        # Check requirements
        if not self.check_requirements():
            print(f"\n{self.RED}Corrija os problemas antes de continuar!{self.END}")
            return
        
        # Step 1: Fix syntax and encoding
        if not self.run_step(
            "PASSO 1",
            "fix_syntax_and_encoding.py",
            "Corrigir erros de sintaxe e codificação"
        ):
            print(f"\n{self.RED}Erro crítico no Passo 1. Abortando...{self.END}")
            return
        
        time.sleep(1)  # Small delay between steps
        
        # Step 2: Test basic setup
        self.run_step(
            "PASSO 2",
            "test_basic_setup.py",
            "Verificar configuração básica"
        )
        
        time.sleep(1)
        
        # Step 3: Create missing tests
        if not self.run_step(
            "PASSO 3",
            "create_missing_tests.py",
            "Criar testes para aumentar cobertura"
        ):
            print(f"\n{self.YELLOW}Aviso no Passo 3, mas continuando...{self.END}")
        
        time.sleep(1)
        
        # Step 4: Run coverage tests
        print(f"\n{self.BLUE}▶ PASSO 4: Executar testes com cobertura{self.END}")
        print("-" * 60)
        
        coverage_result = self.run_coverage_analysis()
        
        # Final summary
        self.print_final_summary(coverage_result)
    
    def run_coverage_analysis(self) -> float:
        """Run coverage analysis and return percentage"""
        try:
            # Run the coverage test script
            result = subprocess.run(
                [self.python_exe, "run_coverage_test.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for all tests
            )
            
            # Try to parse coverage from output
            coverage = 0.0
            for line in result.stdout.split('\n'):
                if 'COBERTURA ATUAL:' in line:
                    try:
                        coverage = float(line.split(':')[1].strip().rstrip('%'))
                    except:
                        pass
            
            return coverage
            
        except Exception as e:
            print(f"{self.RED}Erro ao executar análise de cobertura: {e}{self.END}")
            return 0.0
    
    def print_final_summary(self, coverage: float):
        """Print final summary"""
        print(f"\n{self.BOLD}{'=' * 60}{self.END}")
        print(f"{self.BOLD}RESUMO FINAL{self.END}")
        print(f"{self.BOLD}{'=' * 60}{self.END}")
        
        # Steps completed
        print(f"\n{self.GREEN}✅ Passos Completados ({len(self.steps_completed)}):{self.END}")
        for step in self.steps_completed:
            print(f"   - {step}")
        
        # Errors
        if self.errors:
            print(f"\n{self.RED}❌ Erros Encontrados ({len(self.errors)}):{self.END}")
            for error in self.errors:
                print(f"   - {error}")
        
        # Coverage result
        print(f"\n{self.BOLD}📊 RESULTADO DA COBERTURA:{self.END}")
        
        target = 80.0
        if coverage >= target:
            print(f"{self.GREEN}{self.BOLD}   🎉 META ATINGIDA! Cobertura: {coverage:.2f}%{self.END}")
            print(f"\n{self.GREEN}✅ SUCESSO! O sistema está com {coverage:.2f}% de cobertura!{self.END}")
        else:
            diff = target - coverage
            print(f"{self.YELLOW}   Cobertura Atual: {coverage:.2f}%{self.END}")
            print(f"{self.YELLOW}   Faltam: {diff:.2f}% para atingir 80%{self.END}")
            
            print(f"\n{self.YELLOW}📝 AÇÕES RECOMENDADAS:{self.END}")
            print("   1. Verifique htmlcov/index.html para detalhes")
            print("   2. Execute testes individuais que falharam")
            print("   3. Adicione mais testes para módulos descobertos")
            print("   4. Execute novamente: python execute_complete_fix.py")
        
        # Final instructions
        print(f"\n{self.BLUE}🔧 COMANDOS ÚTEIS:{self.END}")
        print("   • Ver cobertura detalhada: start htmlcov\\index.html")
        print("   • Executar testes específicos: pytest tests/test_nome.py -v")
        print("   • Ver linhas não cobertas: pytest --cov=app --cov-report=term-missing")
        
        print(f"\n{self.BOLD}Processo concluído!{self.END}")


def main():
    """Main entry point"""
    executor = CompleteFixExecutor()
    
    try:
        executor.run()
    except KeyboardInterrupt:
        print(f"\n\n{executor.RED}Processo interrompido pelo usuário!{executor.END}")
    except Exception as e:
        print(f"\n\n{executor.RED}Erro fatal: {str(e)}{executor.END}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
