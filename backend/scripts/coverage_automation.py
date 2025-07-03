#!/usr/bin/env python3
"""
Script de automação para análise e melhoria de cobertura de testes
CardioAI Pro - Backend
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import argparse
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CoverageMetrics:
    """Métricas de cobertura por arquivo/módulo"""

    file_path: str
    line_coverage: float
    branch_coverage: float
    missing_lines: List[int]
    missing_branches: List[str]
    complexity: int
    is_critical: bool


class CoverageAnalyzer:
    """Analisador de cobertura com foco em componentes médicos"""

    CRITICAL_PATHS = [
        "app/services/ecg/",
        "app/services/diagnosis/",
        "app/services/ml_model_service.py",
        "app/utils/medical/",
        "app/core/medical_safety.py",
        "app/api/v1/endpoints/ecg_analysis.py",
    ]

    COVERAGE_TARGETS = {
        "global": 80.0,
        "critical": 100.0,
        "medical": 95.0,
        "services": 85.0,
        "utils": 80.0,
        "api": 85.0,
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_file = project_root / "coverage.xml"
        self.htmlcov_dir = project_root / "htmlcov"
        self.metrics: Dict[str, CoverageMetrics] = {}

    def run_coverage(self, specific_module: Optional[str] = None) -> bool:
        """Executa testes com cobertura"""
        print("🧪 Executando testes com cobertura...")

        cmd = [
            "poetry",
            "run",
            "pytest",
            "--cov=app",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=term-missing:skip-covered",
            "-v",
        ]

        if specific_module:
            cmd.extend(["--cov", specific_module])
            cmd.append(f"tests/test_{Path(specific_module).stem}*.py")

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"❌ Testes falharam:\n{result.stderr}")
                return False

            print("✅ Testes executados com sucesso!")
            return True

        except Exception as e:
            print(f"❌ Erro ao executar testes: {e}")
            return False

    def parse_coverage_xml(self) -> Dict[str, float]:
        """Parse do arquivo XML de cobertura"""
        if not self.coverage_file.exists():
            print(f"❌ Arquivo de cobertura não encontrado: {self.coverage_file}")
            return {}

        tree = ET.parse(self.coverage_file)
        root = tree.getroot()

        # Cobertura global
        global_line_rate = float(root.get("line-rate", 0)) * 100
        global_branch_rate = float(root.get("branch-rate", 0)) * 100

        print(f"\n📊 Cobertura Global:")
        print(f"  - Linhas: {global_line_rate:.1f}%")
        print(f"  - Branches: {global_branch_rate:.1f}%")

        # Cobertura por arquivo
        for package in root.findall(".//package"):
            for class_elem in package.findall(".//class"):
                filename = class_elem.get("filename")
                if not filename:
                    continue

                # Calcular métricas
                line_rate = float(class_elem.get("line-rate", 0)) * 100
                branch_rate = float(class_elem.get("branch-rate", 0)) * 100
                complexity = int(class_elem.get("complexity", 0))

                # Linhas não cobertas
                missing_lines = []
                for line in class_elem.findall(".//line"):
                    if line.get("hits") == "0":
                        missing_lines.append(int(line.get("number")))

                # Verificar se é componente crítico
                is_critical = any(
                    critical in filename for critical in self.CRITICAL_PATHS
                )

                self.metrics[filename] = CoverageMetrics(
                    file_path=filename,
                    line_coverage=line_rate,
                    branch_coverage=branch_rate,
                    missing_lines=missing_lines,
                    missing_branches=[],  # TODO: Parse branch info
                    complexity=complexity,
                    is_critical=is_critical,
                )

        return {
            "global": (global_line_rate + global_branch_rate) / 2,
            "lines": global_line_rate,
            "branches": global_branch_rate,
        }

    def analyze_critical_components(self) -> List[Tuple[str, CoverageMetrics]]:
        """Analisa componentes críticos com cobertura insuficiente"""
        critical_issues = []

        print("\n🏥 Análise de Componentes Críticos:")
        print("=" * 60)

        for path, metrics in self.metrics.items():
            if metrics.is_critical:
                coverage = (metrics.line_coverage + metrics.branch_coverage) / 2
                target = self.COVERAGE_TARGETS["critical"]

                status = "✅" if coverage >= target else "❌"
                print(f"{status} {path}")
                print(f"   Cobertura: {coverage:.1f}% (alvo: {target}%)")

                if coverage < target:
                    critical_issues.append((path, metrics))
                    print(f"   Linhas faltando: {len(metrics.missing_lines)}")
                    if metrics.missing_lines[:5]:  # Mostrar até 5 linhas
                        print(f"   Primeiras linhas: {metrics.missing_lines[:5]}")

        return critical_issues

    def generate_improvement_report(self) -> Dict[str, any]:
        """Gera relatório com sugestões de melhoria"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": len(self.metrics),
                "critical_files": sum(
                    1 for m in self.metrics.values() if m.is_critical
                ),
                "files_below_target": 0,
                "total_missing_lines": 0,
            },
            "priorities": [],
            "suggestions": [],
        }

        # Calcular estatísticas
        for path, metrics in self.metrics.items():
            coverage = (metrics.line_coverage + metrics.branch_coverage) / 2

            # Determinar target baseado no tipo
            if metrics.is_critical:
                target = self.COVERAGE_TARGETS["critical"]
            elif "services" in path:
                target = self.COVERAGE_TARGETS["services"]
            elif "utils" in path:
                target = self.COVERAGE_TARGETS["utils"]
            elif "api" in path:
                target = self.COVERAGE_TARGETS["api"]
            else:
                target = self.COVERAGE_TARGETS["global"]

            if coverage < target:
                report["summary"]["files_below_target"] += 1
                report["summary"]["total_missing_lines"] += len(metrics.missing_lines)

                priority = (
                    "CRITICAL"
                    if metrics.is_critical
                    else "HIGH" if coverage < 60 else "MEDIUM"
                )

                report["priorities"].append(
                    {
                        "file": path,
                        "current_coverage": coverage,
                        "target_coverage": target,
                        "gap": target - coverage,
                        "missing_lines": len(metrics.missing_lines),
                        "priority": priority,
                        "complexity": metrics.complexity,
                    }
                )

        # Ordenar por prioridade e gap
        report["priorities"].sort(
            key=lambda x: (x["priority"] == "CRITICAL", x["gap"]), reverse=True
        )

        # Gerar sugestões
        report["suggestions"] = self._generate_suggestions(report["priorities"])

        return report

    def _generate_suggestions(self, priorities: List[Dict]) -> List[str]:
        """Gera sugestões específicas de melhoria"""
        suggestions = []

        # Top 5 arquivos prioritários
        top_files = priorities[:5]

        for file_info in top_files:
            path = file_info["file"]
            gap = file_info["gap"]
            missing = file_info["missing_lines"]

            if "ml_model_service" in path:
                suggestions.append(
                    f"ML Model Service: Adicionar testes para casos extremos e exceções. "
                    f"Faltam {missing} linhas ({gap:.1f}% abaixo da meta)"
                )

            elif "ecg" in path and "analysis" in path:
                suggestions.append(
                    f"ECG Analysis: Testar todos os tipos de arritmias e condições de sinal. "
                    f"Componente crítico com {missing} linhas não testadas"
                )

            elif "diagnosis" in path:
                suggestions.append(
                    f"Diagnosis Engine: Cobrir todos os algoritmos de decisão clínica. "
                    f"Necessário 100% de cobertura (faltam {gap:.1f}%)"
                )

            elif "api" in path:
                suggestions.append(
                    f"API Endpoint: Testar todos os casos de erro e validação. "
                    f"Adicionar testes de integração para {Path(path).stem}"
                )

        # Sugestões gerais
        if any(p["priority"] == "CRITICAL" for p in priorities):
            suggestions.append(
                "⚠️ URGENTE: Componentes críticos médicos devem ter 100% de cobertura "
                "para compliance com ANVISA/FDA"
            )

        return suggestions

    def generate_test_stubs(self, target_file: str) -> str:
        """Gera stubs de teste para arquivo específico"""
        if target_file not in self.metrics:
            return ""

        metrics = self.metrics[target_file]
        module_name = Path(target_file).stem

        stub = f'''"""
Testes para {module_name} - Aumentar cobertura
Linhas faltando: {metrics.missing_lines[:10]}{'...' if len(metrics.missing_lines) > 10 else ''}
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.{target_file.replace('/', '.').replace('.py', '')} import *


class Test{module_name.title().replace('_', '')}Coverage:
    """Testes para aumentar cobertura para {metrics.line_coverage:.1f}% -> 100%"""
    
'''

        # Adicionar sugestões baseadas no tipo de arquivo
        if "service" in target_file:
            stub += '''    @pytest.fixture
    def service(self):
        """Fixture do serviço"""
        # TODO: Configurar mocks necessários
        return Service()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Teste tratamento de erros - linhas não cobertas"""
        # TODO: Implementar testes para exceções
        pass
    
    @pytest.mark.asyncio  
    async def test_edge_cases(self, service):
        """Teste casos extremos"""
        # TODO: Testar valores limites
        pass
'''

        elif "api" in target_file:
            stub += '''    @pytest.fixture
    def client(self):
        """Test client fixture"""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_error_responses(self, client):
        """Teste respostas de erro - aumentar cobertura"""
        # TODO: Testar 400, 401, 403, 404, 422, 500
        pass
    
    def test_validation_errors(self, client):
        """Teste validação de entrada"""
        # TODO: Testar dados inválidos
        pass
'''

        return stub

    def export_html_report(self, output_dir: Path):
        """Exporta relatório HTML aprimorado"""
        if not self.htmlcov_dir.exists():
            print("❌ Diretório htmlcov não encontrado")
            return

        # Copiar e aprimorar relatório HTML
        import shutil

        shutil.copytree(self.htmlcov_dir, output_dir, dirs_exist_ok=True)

        # Adicionar dashboard customizado
        dashboard_html = self._generate_dashboard_html()

        with open(output_dir / "dashboard.html", "w") as f:
            f.write(dashboard_html)

        print(f"✅ Relatório HTML exportado para: {output_dir}")

    def _generate_dashboard_html(self) -> str:
        """Gera dashboard HTML customizado"""
        coverage_data = self.parse_coverage_xml()
        critical_issues = self.analyze_critical_components()

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>CardioAI Pro - Coverage Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ 
            display: inline-block; 
            padding: 20px; 
            margin: 10px;
            border-radius: 8px;
            background: #f0f0f0;
        }}
        .critical {{ background: #ffcccc; }}
        .good {{ background: #ccffcc; }}
        .warning {{ background: #ffffcc; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>CardioAI Pro - Cobertura de Testes</h1>
    
    <div class="metrics">
        <div class="metric {'good' if coverage_data.get('global', 0) >= 80 else 'warning'}">
            <h3>Cobertura Global</h3>
            <p>{coverage_data.get('global', 0):.1f}%</p>
        </div>
        
        <div class="metric {'critical' if len(critical_issues) > 0 else 'good'}">
            <h3>Componentes Críticos</h3>
            <p>{len(critical_issues)} com problemas</p>
        </div>
    </div>
    
    <h2>Componentes Críticos Precisando Atenção</h2>
    <table>
        <tr>
            <th>Arquivo</th>
            <th>Cobertura Atual</th>
            <th>Meta</th>
            <th>Linhas Faltando</th>
        </tr>
        {''.join(f"""
        <tr>
            <td>{path}</td>
            <td>{(metrics.line_coverage + metrics.branch_coverage) / 2:.1f}%</td>
            <td>100%</td>
            <td>{len(metrics.missing_lines)}</td>
        </tr>
        """ for path, metrics in critical_issues)}
    </table>
    
    <p><a href="index.html">Ver relatório completo</a></p>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(
        description="Automação de análise e melhoria de cobertura"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analisar cobertura atual"
    )
    parser.add_argument(
        "--run", action="store_true", help="Executar testes com cobertura"
    )
    parser.add_argument("--module", help="Módulo específico para análise")
    parser.add_argument("--generate-stubs", help="Gerar stubs de teste para arquivo")
    parser.add_argument("--export-report", help="Exportar relatório para diretório")
    parser.add_argument(
        "--improvement-plan", action="store_true", help="Gerar plano de melhoria"
    )

    args = parser.parse_args()

    # Encontrar diretório do projeto
    project_root = Path.cwd()
    if not (project_root / "pyproject.toml").exists():
        print("❌ Execute este script no diretório raiz do backend")
        sys.exit(1)

    analyzer = CoverageAnalyzer(project_root)

    # Executar testes se solicitado
    if args.run:
        if not analyzer.run_coverage(args.module):
            sys.exit(1)

    # Analisar cobertura
    if args.analyze or args.run:
        coverage_data = analyzer.parse_coverage_xml()
        critical_issues = analyzer.analyze_critical_components()

        # Verificar se atende aos requisitos
        if coverage_data.get("global", 0) < 80:
            print(f"\n❌ Cobertura global abaixo de 80%!")

        if critical_issues:
            print(
                f"\n❌ {len(critical_issues)} componentes críticos com cobertura insuficiente!"
            )

    # Gerar plano de melhoria
    if args.improvement_plan:
        report = analyzer.generate_improvement_report()

        print("\n📋 Plano de Melhoria de Cobertura")
        print("=" * 60)

        print(f"\nResumo:")
        print(f"  - Arquivos abaixo da meta: {report['summary']['files_below_target']}")
        print(
            f"  - Total de linhas não cobertas: {report['summary']['total_missing_lines']}"
        )

        print(f"\nPrioridades:")
        for i, priority in enumerate(report["priorities"][:10], 1):
            print(f"\n{i}. {priority['file']}")
            print(f"   Prioridade: {priority['priority']}")
            print(
                f"   Cobertura: {priority['current_coverage']:.1f}% → {priority['target_coverage']:.1f}%"
            )
            print(f"   Linhas faltando: {priority['missing_lines']}")

        print(f"\nSugestões:")
        for suggestion in report["suggestions"]:
            print(f"  • {suggestion}")

        # Salvar relatório
        report_path = (
            project_root
            / f'coverage_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Relatório salvo em: {report_path}")

    # Gerar stubs de teste
    if args.generate_stubs:
        stub = analyzer.generate_test_stubs(args.generate_stubs)
        if stub:
            stub_path = (
                project_root
                / f"tests/test_{Path(args.generate_stubs).stem}_coverage.py"
            )

            print(f"\n📝 Stub de teste gerado:")
            print("-" * 60)
            print(stub)
            print("-" * 60)

            if input(f"\nSalvar em {stub_path}? (s/n): ").lower() == "s":
                with open(stub_path, "w") as f:
                    f.write(stub)
                print(f"✅ Arquivo salvo!")
        else:
            print(f"❌ Arquivo não encontrado na análise de cobertura")

    # Exportar relatório HTML
    if args.export_report:
        output_dir = Path(args.export_report)
        output_dir.mkdir(exist_ok=True)
        analyzer.export_html_report(output_dir)


if __name__ == "__main__":
    main()
