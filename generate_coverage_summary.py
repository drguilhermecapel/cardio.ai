#!/usr/bin/env python3
"""
Script para gerar resumo consolidado de cobertura
Executado pelo pre-commit antes do push
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import subprocess
import sys


class CoverageSummaryGenerator:
    """Gerador de resumo de cobertura para pre-commit"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.backend_coverage = self.project_root / "backend" / "coverage.xml"
        self.frontend_coverage = (
            self.project_root / "frontend" / "coverage" / "coverage-final.json"
        )
        self.summary_file = self.project_root / "COVERAGE_SUMMARY.md"

    def get_backend_coverage(self) -> Optional[Dict[str, float]]:
        """Obtém cobertura do backend do arquivo XML"""
        if not self.backend_coverage.exists():
            return None

        try:
            tree = ET.parse(self.backend_coverage)
            root = tree.getroot()

            line_rate = float(root.get("line-rate", 0)) * 100
            branch_rate = float(root.get("branch-rate", 0)) * 100

            # Verificar componentes críticos
            critical_components = {}
            for package in root.findall(".//package"):
                for class_elem in package.findall(".//class"):
                    filename = class_elem.get("filename", "")

                    # Componentes críticos médicos
                    if any(
                        critical in filename
                        for critical in [
                            "services/ecg/",
                            "services/diagnosis/",
                            "utils/medical/",
                            "core/medical_safety",
                        ]
                    ):
                        file_line_rate = float(class_elem.get("line-rate", 0)) * 100
                        file_branch_rate = float(class_elem.get("branch-rate", 0)) * 100
                        critical_components[filename] = (
                            file_line_rate + file_branch_rate
                        ) / 2

            return {
                "total": (line_rate + branch_rate) / 2,
                "lines": line_rate,
                "branches": branch_rate,
                "critical_components": critical_components,
            }

        except Exception as e:
            print(f"❌ Erro ao ler cobertura do backend: {e}")
            return None

    def get_frontend_coverage(self) -> Optional[Dict[str, float]]:
        """Obtém cobertura do frontend do arquivo JSON"""
        if not self.frontend_coverage.exists():
            return None

        try:
            with open(self.frontend_coverage, "r") as f:
                data = json.load(f)

            total = data.get("total", {})

            # Calcular percentuais
            lines_pct = total.get("lines", {}).get("pct", 0)
            branches_pct = total.get("branches", {}).get("pct", 0)
            functions_pct = total.get("functions", {}).get("pct", 0)
            statements_pct = total.get("statements", {}).get("pct", 0)

            # Verificar componentes críticos
            critical_components = {}
            for file_path, file_data in data.items():
                if file_path == "total":
                    continue

                # Componentes médicos críticos
                if any(
                    critical in file_path
                    for critical in [
                        "components/medical/",
                        "services/ecg/",
                        "services/diagnosis/",
                        "utils/medical/",
                    ]
                ):
                    file_lines = file_data.get("lines", {}).get("pct", 0)
                    file_branches = file_data.get("branches", {}).get("pct", 0)
                    critical_components[file_path] = (file_lines + file_branches) / 2

            return {
                "total": (lines_pct + branches_pct + functions_pct + statements_pct)
                / 4,
                "lines": lines_pct,
                "branches": branches_pct,
                "functions": functions_pct,
                "statements": statements_pct,
                "critical_components": critical_components,
            }

        except Exception as e:
            print(f"❌ Erro ao ler cobertura do frontend: {e}")
            return None

    def check_coverage_requirements(
        self, backend: Dict, frontend: Dict
    ) -> Dict[str, bool]:
        """Verifica se os requisitos de cobertura foram atendidos"""
        requirements = {
            "backend_global": backend["total"] >= 80.0,
            "frontend_global": frontend["total"] >= 80.0,
            "backend_critical": all(
                cov >= 100.0 for cov in backend["critical_components"].values()
            ),
            "frontend_critical": all(
                cov >= 100.0 for cov in frontend["critical_components"].values()
            ),
            "ml_service": True,  # Verificar especificamente o ML service
        }

        # Verificar ML Service especificamente
        ml_service_coverage = next(
            (
                cov
                for path, cov in backend["critical_components"].items()
                if "ml_model_service" in path
            ),
            None,
        )
        if ml_service_coverage is not None:
            requirements["ml_service"] = ml_service_coverage >= 80.0

        return requirements

    def generate_summary(self) -> str:
        """Gera resumo em Markdown"""
        backend = self.get_backend_coverage()
        frontend = self.get_frontend_coverage()

        if not backend or not frontend:
            return "❌ Não foi possível gerar resumo de cobertura\n"

        requirements = self.check_coverage_requirements(backend, frontend)
        all_passed = all(requirements.values())

        # Emoji de status
        status_emoji = "✅" if all_passed else "❌"

        summary = f"""# {status_emoji} Resumo de Cobertura - CardioAI Pro

**Data**: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
**Status**: {'APROVADO' if all_passed else 'REPROVADO'}

## 📊 Cobertura Global

| Stack | Total | Linhas | Branches | Status |
|-------|-------|---------|----------|--------|
| Backend | {backend['total']:.1f}% | {backend['lines']:.1f}% | {backend['branches']:.1f}% | {'✅' if requirements['backend_global'] else '❌'} |
| Frontend | {frontend['total']:.1f}% | {frontend['lines']:.1f}% | {frontend['branches']:.1f}% | {'✅' if requirements['frontend_global'] else '❌'} |

**Meta**: ≥ 80% para cobertura global

## 🏥 Componentes Médicos Críticos

### Backend
| Componente | Cobertura | Status |
|------------|-----------|--------|
"""

        # Adicionar componentes críticos do backend
        for path, coverage in sorted(backend["critical_components"].items()):
            component_name = Path(path).stem
            status = "✅" if coverage >= 100.0 else "❌"
            summary += f"| {component_name} | {coverage:.1f}% | {status} |\n"

        summary += f"""
### Frontend
| Componente | Cobertura | Status |
|------------|-----------|--------|
"""

        # Adicionar componentes críticos do frontend
        for path, coverage in sorted(frontend["critical_components"].items()):
            component_name = Path(path).stem
            status = "✅" if coverage >= 100.0 else "❌"
            summary += f"| {component_name} | {coverage:.1f}% | {status} |\n"

        summary += """
**Meta**: 100% para componentes críticos médicos

## 📋 Requisitos de Compliance

- [{'x' if requirements['backend_global'] else ' '}] Backend com cobertura ≥ 80%
- [{'x' if requirements['frontend_global'] else ' '}] Frontend com cobertura ≥ 80%
- [{'x' if requirements['backend_critical'] else ' '}] Componentes críticos backend = 100%
- [{'x' if requirements['frontend_critical'] else ' '}] Componentes críticos frontend = 100%
- [{'x' if requirements['ml_service'] else ' '}] ML Model Service ≥ 80%

"""

        # Adicionar ações necessárias se falhou
        if not all_passed:
            summary += "## ⚠️ Ações Necessárias\n\n"

            if not requirements["backend_global"]:
                summary += f"- **Backend**: Aumentar cobertura global de {backend['total']:.1f}% para 80%\n"

            if not requirements["frontend_global"]:
                summary += f"- **Frontend**: Aumentar cobertura global de {frontend['total']:.1f}% para 80%\n"

            if not requirements["backend_critical"]:
                failed_critical = [
                    path
                    for path, cov in backend["critical_components"].items()
                    if cov < 100.0
                ]
                summary += f"- **Componentes Críticos Backend**: {', '.join(Path(p).stem for p in failed_critical)}\n"

            if not requirements["frontend_critical"]:
                failed_critical = [
                    path
                    for path, cov in frontend["critical_components"].items()
                    if cov < 100.0
                ]
                summary += f"- **Componentes Críticos Frontend**: {', '.join(Path(p).stem for p in failed_critical)}\n"

            if not requirements["ml_service"]:
                summary += "- **ML Model Service**: Aumentar cobertura para 80%\n"

            summary += "\n### 🚫 Commit bloqueado até atingir requisitos mínimos\n"
        else:
            summary += "## ✅ Todos os requisitos de cobertura atendidos!\n"

        summary += f"""
---

### 🔧 Comandos Úteis

```bash
# Backend - Verificar cobertura
cd backend && poetry run pytest --cov=app --cov-report=html

# Frontend - Verificar cobertura  
cd frontend && npm run test:coverage

# Executar apenas testes críticos
make test:critical
```

### 📚 Documentação

- [Guia de Cobertura](docs/TEST_COVERAGE_GUIDE.md)
- [Requisitos ANVISA/FDA](docs/REGULATORY_COMPLIANCE.md)
- [Como Escrever Testes](docs/TESTING_BEST_PRACTICES.md)
"""

        return summary

    def save_summary(self, summary: str):
        """Salva resumo em arquivo"""
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"✅ Resumo salvo em: {self.summary_file}")

    def update_readme_badge(self, backend_coverage: float, frontend_coverage: float):
        """Atualiza badges de cobertura no README"""
        readme_path = self.project_root / "README.md"

        if not readme_path.exists():
            return

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Atualizar badges (assumindo formato shields.io)
        import re

        # Backend badge
        content = re.sub(
            r"!\[Backend Coverage\]\(.*?\)",
            f'![Backend Coverage](https://img.shields.io/badge/Backend%20Coverage-{backend_coverage:.0f}%25-{"brightgreen" if backend_coverage >= 80 else "yellow" if backend_coverage >= 60 else "red"})',
            content,
        )

        # Frontend badge
        content = re.sub(
            r"!\[Frontend Coverage\]\(.*?\)",
            f'![Frontend Coverage](https://img.shields.io/badge/Frontend%20Coverage-{frontend_coverage:.0f}%25-{"brightgreen" if frontend_coverage >= 80 else "yellow" if frontend_coverage >= 60 else "red"})',
            content,
        )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Badges do README atualizados")

    def run(self) -> bool:
        """Executa geração de resumo e retorna status"""
        print("🔍 Gerando resumo de cobertura...\n")

        # Gerar resumo
        summary = self.generate_summary()

        # Salvar arquivo
        self.save_summary(summary)

        # Imprimir resumo no console
        print(summary)

        # Verificar se passou
        backend = self.get_backend_coverage()
        frontend = self.get_frontend_coverage()

        if backend and frontend:
            # Atualizar badges
            self.update_readme_badge(backend["total"], frontend["total"])

            # Verificar requisitos
            requirements = self.check_coverage_requirements(backend, frontend)
            all_passed = all(requirements.values())

            if not all_passed:
                print("\n❌ Requisitos de cobertura não atendidos!")
                print("💡 Execute os comandos sugeridos para melhorar a cobertura")
                return False
            else:
                print("\n✅ Todos os requisitos de cobertura atendidos!")
                return True

        return False


def main():
    """Função principal"""
    generator = CoverageSummaryGenerator()

    # Verificar se estamos em um hook de pre-commit
    if "--pre-commit" in sys.argv:
        # Modo silencioso para pre-commit
        success = generator.run()
        sys.exit(0 if success else 1)
    else:
        # Modo verboso para execução manual
        success = generator.run()

        if not success:
            response = input("\nDeseja continuar mesmo assim? (s/N): ")
            if response.lower() != "s":
                sys.exit(1)

        sys.exit(0)


if __name__ == "__main__":
    main()
