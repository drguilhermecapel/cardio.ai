#!/usr/bin/env python3
"""
Verify System Status
Verifica o status completo do sistema CardioAI Pro
"""

import os
import sys
import json
import importlib
from pathlib import Path
from datetime import datetime


class SystemStatusVerifier:
    """Verify complete system status"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.status = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "checks": {},
            "coverage": None,
            "ready": False
        }
        
    def run(self):
        """Run all verifications"""
        print("=" * 60)
        print("CardioAI Pro - Verificação de Status do Sistema")
        print("=" * 60)
        print()
        
        # Run checks
        self.check_python_version()
        self.check_directory_structure()
        self.check_core_files()
        self.check_dependencies()
        self.check_modules()
        self.check_test_files()
        self.check_coverage_report()
        
        # Final assessment
        self.assess_system_readiness()
        
        # Save report
        self.save_report()
        
        # Print summary
        self.print_summary()
        
    def check_python_version(self):
        """Check Python version"""
        print("1. Verificando versão do Python...")
        
        version = sys.version_info
        is_valid = version.major == 3 and version.minor >= 8
        
        self.status["checks"]["python_version"] = {
            "status": "OK" if is_valid else "ERROR",
            "value": f"{version.major}.{version.minor}.{version.micro}",
            "required": "3.8+"
        }
        
        if is_valid:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        else:
            print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requer 3.8+)")
    
    def check_directory_structure(self):
        """Check directory structure"""
        print("\n2. Verificando estrutura de diretórios...")
        
        required_dirs = [
            "app",
            "app/api",
            "app/core", 
            "app/db",
            "app/ml",
            "app/schemas",
            "app/services",
            "app/utils",
            "tests",
            "uploads",
            "logs"
        ]
        
        missing = []
        for dir_name in required_dirs:
            dir_path = self.backend_path / dir_name
            if not dir_path.exists():
                missing.append(dir_name)
        
        self.status["checks"]["directory_structure"] = {
            "status": "OK" if not missing else "WARNING",
            "total": len(required_dirs),
            "found": len(required_dirs) - len(missing),
            "missing": missing
        }
        
        if not missing:
            print(f"   ✅ Todos os {len(required_dirs)} diretórios presentes")
        else:
            print(f"   ⚠️ Faltando {len(missing)} diretórios: {', '.join(missing[:3])}...")
    
    def check_core_files(self):
        """Check core files"""
        print("\n3. Verificando arquivos principais...")
        
        core_files = [
            "app/services/ecg_service.py",
            "app/utils/memory_monitor.py",
            "app/schemas/ecg_analysis.py",
            "app/services/interpretability_service.py",
            "app/core/config.py",
            "app/core/exceptions.py",
            "app/db/base.py"
        ]
        
        missing = []
        syntax_errors = []
        
        for file_path in core_files:
            full_path = self.backend_path / file_path
            if not full_path.exists():
                missing.append(file_path)
            else:
                # Try to compile to check syntax
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), file_path, 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}: {e.msg}")
                except:
                    pass
        
        self.status["checks"]["core_files"] = {
            "status": "OK" if not missing and not syntax_errors else "ERROR",
            "total": len(core_files),
            "found": len(core_files) - len(missing),
            "missing": missing,
            "syntax_errors": syntax_errors
        }
        
        if not missing and not syntax_errors:
            print(f"   ✅ Todos os {len(core_files)} arquivos principais OK")
        else:
            if missing:
                print(f"   ❌ Faltando {len(missing)} arquivos")
            if syntax_errors:
                print(f"   ❌ {len(syntax_errors)} erros de sintaxe")
    
    def check_dependencies(self):
        """Check installed dependencies"""
        print("\n4. Verificando dependências...")
        
        required = [
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "numpy",
            "scipy",
            "pandas",
            "pytest",
            "pytest_cov"
        ]
        
        missing = []
        for package in required:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)
        
        self.status["checks"]["dependencies"] = {
            "status": "OK" if not missing else "ERROR",
            "total": len(required),
            "installed": len(required) - len(missing),
            "missing": missing
        }
        
        if not missing:
            print(f"   ✅ Todas as {len(required)} dependências principais instaladas")
        else:
            print(f"   ❌ Faltando {len(missing)} dependências: {', '.join(missing)}")
    
    def check_modules(self):
        """Check if modules can be imported"""
        print("\n5. Verificando imports dos módulos...")
        
        # Add to path
        sys.path.insert(0, str(self.backend_path))
        
        test_imports = [
            "app.core.config",
            "app.services.ecg_service",
            "app.utils.memory_monitor",
            "app.schemas.ecg_analysis"
        ]
        
        import_errors = []
        for module in test_imports:
            try:
                importlib.import_module(module)
            except Exception as e:
                import_errors.append(f"{module}: {str(e)[:50]}...")
        
        self.status["checks"]["module_imports"] = {
            "status": "OK" if not import_errors else "ERROR",
            "total": len(test_imports),
            "successful": len(test_imports) - len(import_errors),
            "errors": import_errors
        }
        
        if not import_errors:
            print(f"   ✅ Todos os {len(test_imports)} módulos importados com sucesso")
        else:
            print(f"   ❌ {len(import_errors)} erros de importação")
    
    def check_test_files(self):
        """Check test files"""
        print("\n6. Verificando arquivos de teste...")
        
        test_pattern = "test_*.py"
        test_files = list((self.backend_path / "tests").glob(test_pattern))
        
        # Check for comprehensive tests
        comprehensive_tests = [
            "test_ecg_service_comprehensive.py",
            "test_memory_monitor_comprehensive.py",
            "test_interpretability_comprehensive.py"
        ]
        
        found_comprehensive = []
        for test_name in comprehensive_tests:
            if (self.backend_path / "tests" / test_name).exists():
                found_comprehensive.append(test_name)
        
        self.status["checks"]["test_files"] = {
            "status": "OK" if len(test_files) > 20 else "WARNING",
            "total": len(test_files),
            "comprehensive_tests": len(found_comprehensive)
        }
        
        print(f"   ✅ {len(test_files)} arquivos de teste encontrados")
        print(f"   ✅ {len(found_comprehensive)} testes abrangentes criados")
    
    def check_coverage_report(self):
        """Check coverage report"""
        print("\n7. Verificando relatório de cobertura...")
        
        # Check for coverage.json
        coverage_json = self.backend_path / "coverage.json"
        htmlcov_dir = self.backend_path / "htmlcov"
        
        coverage_data = None
        if coverage_json.exists():
            try:
                with open(coverage_json, 'r') as f:
                    coverage_data = json.load(f)
                    coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
                    self.status["coverage"] = coverage_percent
            except:
                pass
        
        # Check final report
        final_report = self.backend_path / "coverage_final_report.json"
        if final_report.exists():
            try:
                with open(final_report, 'r') as f:
                    report_data = json.load(f)
                    if "coverage_percentage" in report_data:
                        self.status["coverage"] = report_data["coverage_percentage"]
            except:
                pass
        
        self.status["checks"]["coverage"] = {
            "status": "OK" if self.status["coverage"] and self.status["coverage"] >= 80 else "INFO",
            "percentage": self.status["coverage"] or "N/A",
            "html_report": htmlcov_dir.exists(),
            "json_report": coverage_json.exists()
        }
        
        if self.status["coverage"]:
            if self.status["coverage"] >= 80:
                print(f"   ✅ Cobertura: {self.status['coverage']:.2f}% (META ATINGIDA!)")
            else:
                print(f"   ⚠️ Cobertura: {self.status['coverage']:.2f}% (meta: 80%)")
        else:
            print("   ℹ️ Relatório de cobertura não encontrado")
    
    def assess_system_readiness(self):
        """Assess overall system readiness"""
        critical_checks = ["core_files", "dependencies", "module_imports"]
        
        all_critical_ok = all(
            self.status["checks"].get(check, {}).get("status") == "OK"
            for check in critical_checks
        )
        
        coverage_ok = self.status["coverage"] and self.status["coverage"] >= 80
        
        self.status["ready"] = all_critical_ok
        self.status["coverage_achieved"] = coverage_ok
    
    def save_report(self):
        """Save status report"""
        report_file = self.backend_path / "system_status_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2)
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 60)
        print("RESUMO DO STATUS")
        print("=" * 60)
        
        # Count statuses
        ok_count = sum(1 for check in self.status["checks"].values() if check["status"] == "OK")
        error_count = sum(1 for check in self.status["checks"].values() if check["status"] == "ERROR")
        warning_count = sum(1 for check in self.status["checks"].values() if check["status"] == "WARNING")
        
        print(f"\n✅ OK: {ok_count}")
        print(f"⚠️  Avisos: {warning_count}")
        print(f"❌ Erros: {error_count}")
        
        if self.status["coverage"]:
            print(f"\n📊 Cobertura: {self.status['coverage']:.2f}%")
        
        print(f"\n🚦 Status Geral: ", end="")
        if self.status["ready"] and self.status.get("coverage_achieved"):
            print("✅ SISTEMA PRONTO! Meta de 80% atingida!")
        elif self.status["ready"]:
            print("⚠️ Sistema funcional, mas cobertura abaixo de 80%")
        else:
            print("❌ Sistema precisa de correções")
        
        print("\n📄 Relatório salvo em: system_status_report.json")
        
        if not self.status["ready"] or not self.status.get("coverage_achieved"):
            print("\n🔧 Ações Recomendadas:")
            if error_count > 0:
                print("   1. Execute: python fix_syntax_and_encoding.py")
            if self.status["checks"]["dependencies"]["status"] != "OK":
                print("   2. Execute: pip install -r requirements.txt")
            if not self.status.get("coverage_achieved"):
                print("   3. Execute: python execute_complete_fix.py")


def main():
    """Main entry point"""
    verifier = SystemStatusVerifier()
    verifier.run()


if __name__ == "__main__":
    main()
