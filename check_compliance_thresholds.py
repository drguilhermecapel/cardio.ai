#!/usr/bin/env python3
"""
Script para verificar se a cobertura atende aos requisitos de compliance ANVISA/FDA
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from datetime import datetime


class ComplianceThresholdChecker:
    """Verificador de limiares de cobertura para compliance médico"""

    # Componentes críticos que requerem 100% de cobertura
    CRITICAL_COMPONENTS = {
        "backend": [
            "app/services/ecg/analysis.py",
            "app/services/ecg/signal_quality.py",
            "app/services/diagnosis/engine.py",
            "app/utils/medical/validation.py",
            "app/core/medical_safety.py",
        ],
        "frontend": [
            "src/components/medical/ECGVisualization.tsx",
            "src/components/medical/DiagnosisDisplay.tsx",
            "src/services/ecg/analysis.ts",
            "src/services/diagnosis/engine.ts",
            "src/utils/medical/validation.ts",
        ],
    }

    # Requisitos mínimos de cobertura
    THRESHOLDS = {
        "global": {"min_coverage": 80.0, "target_coverage": 85.0},
        "critical": {"min_coverage": 100.0, "target_coverage": 100.0},
        "medical_components": {"min_coverage": 95.0, "target_coverage": 100.0},
    }

    def __init__(self, report_path: str = None):
        self.report_path = Path(report_path) if report_path else None
        self.results = {
            "backend": {},
            "frontend": {},
            "compliance": {"passed": False, "issues": [], "warnings": []},
        }

    def parse_coverage_xml(self, xml_path: Path) -> Dict[str, float]:
        """Parse arquivo XML de cobertura (formato Cobertura)"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        coverage_data = {}

        # Cobertura global
        global_coverage = {
            "line_rate": float(root.get("line-rate", 0)) * 100,
            "branch_rate": float(root.get("branch-rate", 0)) * 100,
        }
        coverage_data["global"] = (
            global_coverage["line_rate"] + global_coverage["branch_rate"]
        ) / 2

        # Cobertura por arquivo
        for package in root.findall(".//package"):
            for class_elem in package.findall(".//class"):
                filename = class_elem.get("filename")
                line_rate = float(class_elem.get("line-rate", 0)) * 100
                branch_rate = float(class_elem.get("branch-rate", 0)) * 100
                coverage_data[filename] = (line_rate + branch_rate) / 2

        return coverage_data

    def parse_lcov_file(self, lcov_path: Path) -> Dict[str, float]:
        """Parse arquivo LCOV de cobertura"""
        coverage_data = {}
        current_file = None
        file_stats = {}

        with open(lcov_path, "r") as f:
            for line in f:
                line = line.strip()

                if line.startswith("SF:"):
                    current_file = line[3:]
                    file_stats[current_file] = {
                        "lines_found": 0,
                        "lines_hit": 0,
                        "branches_found": 0,
                        "branches_hit": 0,
                    }

                elif line.startswith("LF:") and current_file:
                    file_stats[current_file]["lines_found"] = int(line[3:])

                elif line.startswith("LH:") and current_file:
                    file_stats[current_file]["lines_hit"] = int(line[3:])

                elif line.startswith("BRF:") and current_file:
                    file_stats[current_file]["branches_found"] = int(line[4:])

                elif line.startswith("BRH:") and current_file:
                    file_stats[current_file]["branches_hit"] = int(line[4:])

        # Calcular percentuais
        total_lines = 0
        total_hits = 0

        for file, stats in file_stats.items():
            if stats["lines_found"] > 0:
                line_coverage = (stats["lines_hit"] / stats["lines_found"]) * 100
                branch_coverage = 0

                if stats["branches_found"] > 0:
                    branch_coverage = (
                        stats["branches_hit"] / stats["branches_found"]
                    ) * 100

                coverage_data[file] = (line_coverage + branch_coverage) / 2
                total_lines += stats["lines_found"]
                total_hits += stats["lines_hit"]

        if total_lines > 0:
            coverage_data["global"] = (total_hits / total_lines) * 100

        return coverage_data

    def check_critical_components(
        self, coverage_data: Dict[str, float], component_type: str
    ) -> List[str]:
        """Verifica cobertura dos componentes críticos"""
        issues = []

        for component in self.CRITICAL_COMPONENTS[component_type]:
            # Procurar o componente nos dados de cobertura
            component_coverage = None

            for file, coverage in coverage_data.items():
                if component in file or file.endswith(component.split("/")[-1]):
                    component_coverage = coverage
                    break

            if component_coverage is None:
                issues.append(f"Componente crítico não encontrado: {component}")
            elif component_coverage < self.THRESHOLDS["critical"]["min_coverage"]:
                issues.append(
                    f"Componente crítico com cobertura insuficiente: "
                    f"{component} ({component_coverage:.1f}% < 100%)"
                )

        return issues

    def check_medical_components(self, coverage_data: Dict[str, float]) -> List[str]:
        """Verifica cobertura de componentes médicos em geral"""
        issues = []
        warnings = []

        medical_keywords = ["medical", "ecg", "diagnosis", "clinical", "patient"]

        for file, coverage in coverage_data.items():
            # Verificar se é um componente médico
            is_medical = any(keyword in file.lower() for keyword in medical_keywords)

            if is_medical:
                if coverage < self.THRESHOLDS["medical_components"]["min_coverage"]:
                    issues.append(
                        f"Componente médico com cobertura baixa: "
                        f"{file} ({coverage:.1f}% < 95%)"
                    )
                elif (
                    coverage < self.THRESHOLDS["medical_components"]["target_coverage"]
                ):
                    warnings.append(
                        f"Componente médico abaixo do ideal: "
                        f"{file} ({coverage:.1f}% < 100%)"
                    )

        return issues, warnings

    def validate_coverage(
        self, backend_coverage: Path, frontend_coverage: Path
    ) -> bool:
        """Valida cobertura completa do sistema"""
        print("🔍 Verificando cobertura para compliance ANVISA/FDA...\n")

        # Backend
        print("📊 Backend:")
        if backend_coverage.suffix == ".xml":
            backend_data = self.parse_coverage_xml(backend_coverage)
        else:
            backend_data = self.parse_lcov_file(backend_coverage)

        self.results["backend"] = backend_data

        # Verificar cobertura global do backend
        if backend_data.get("global", 0) < self.THRESHOLDS["global"]["min_coverage"]:
            self.results["compliance"]["issues"].append(
                f"Backend: Cobertura global insuficiente "
                f"({backend_data['global']:.1f}% < 80%)"
            )
        else:
            print(f"✅ Cobertura global: {backend_data['global']:.1f}%")

        # Verificar componentes críticos do backend
        backend_critical_issues = self.check_critical_components(
            backend_data, "backend"
        )
        self.results["compliance"]["issues"].extend(backend_critical_issues)

        if not backend_critical_issues:
            print("✅ Todos os componentes críticos com 100% de cobertura")

        # Frontend
        print("\n📊 Frontend:")
        if frontend_coverage.suffix == ".xml":
            frontend_data = self.parse_coverage_xml(frontend_coverage)
        else:
            frontend_data = self.parse_lcov_file(frontend_coverage)

        self.results["frontend"] = frontend_data

        # Verificar cobertura global do frontend
        if frontend_data.get("global", 0) < self.THRESHOLDS["global"]["min_coverage"]:
            self.results["compliance"]["issues"].append(
                f"Frontend: Cobertura global insuficiente "
                f"({frontend_data['global']:.1f}% < 80%)"
            )
        else:
            print(f"✅ Cobertura global: {frontend_data['global']:.1f}%")

        # Verificar componentes críticos do frontend
        frontend_critical_issues = self.check_critical_components(
            frontend_data, "frontend"
        )
        self.results["compliance"]["issues"].extend(frontend_critical_issues)

        if not frontend_critical_issues:
            print("✅ Todos os componentes críticos com 100% de cobertura")

        # Verificar componentes médicos
        backend_medical_issues, backend_medical_warnings = (
            self.check_medical_components(backend_data)
        )
        frontend_medical_issues, frontend_medical_warnings = (
            self.check_medical_components(frontend_data)
        )

        self.results["compliance"]["issues"].extend(backend_medical_issues)
        self.results["compliance"]["issues"].extend(frontend_medical_issues)
        self.results["compliance"]["warnings"].extend(backend_medical_warnings)
        self.results["compliance"]["warnings"].extend(frontend_medical_warnings)

        # Resultado final
        self.results["compliance"]["passed"] = (
            len(self.results["compliance"]["issues"]) == 0
        )

        return self.results["compliance"]["passed"]

    def generate_report(self, output_path: Path):
        """Gera relatório detalhado de compliance"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "compliance_status": (
                "PASSED" if self.results["compliance"]["passed"] else "FAILED"
            ),
            "summary": {
                "backend_global_coverage": self.results["backend"].get("global", 0),
                "frontend_global_coverage": self.results["frontend"].get("global", 0),
                "total_issues": len(self.results["compliance"]["issues"]),
                "total_warnings": len(self.results["compliance"]["warnings"]),
            },
            "details": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Relatório salvo em: {output_path}")

    def print_summary(self):
        """Imprime resumo dos resultados"""
        print("\n" + "=" * 60)
        print("📋 RESUMO DE COMPLIANCE")
        print("=" * 60)

        if self.results["compliance"]["passed"]:
            print("✅ STATUS: APROVADO")
        else:
            print("❌ STATUS: REPROVADO")

        if self.results["compliance"]["issues"]:
            print(
                f"\n❌ Problemas encontrados ({len(self.results['compliance']['issues'])}):"
            )
            for issue in self.results["compliance"]["issues"]:
                print(f"  - {issue}")

        if self.results["compliance"]["warnings"]:
            print(f"\n⚠️  Avisos ({len(self.results['compliance']['warnings'])}):")
            for warning in self.results["compliance"]["warnings"]:
                print(f"  - {warning}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Verifica limiares de cobertura para compliance ANVISA/FDA"
    )
    parser.add_argument(
        "--backend-coverage",
        required=True,
        help="Caminho para o arquivo de cobertura do backend (XML ou LCOV)",
    )
    parser.add_argument(
        "--frontend-coverage",
        required=True,
        help="Caminho para o arquivo de cobertura do frontend (XML ou LCOV)",
    )
    parser.add_argument(
        "--output",
        default="compliance-report.json",
        help="Caminho para salvar o relatório de compliance",
    )
    parser.add_argument(
        "--min-global",
        type=float,
        default=80.0,
        help="Cobertura global mínima requerida (%)",
    )
    parser.add_argument(
        "--min-critical",
        type=float,
        default=100.0,
        help="Cobertura mínima para componentes críticos (%)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Falha se houver qualquer aviso"
    )

    args = parser.parse_args()

    # Verificar arquivos
    backend_path = Path(args.backend_coverage)
    frontend_path = Path(args.frontend_coverage)

    if not backend_path.exists():
        print(f"❌ Arquivo de cobertura do backend não encontrado: {backend_path}")
        sys.exit(1)

    if not frontend_path.exists():
        print(f"❌ Arquivo de cobertura do frontend não encontrado: {frontend_path}")
        sys.exit(1)

    # Executar verificação
    checker = ComplianceThresholdChecker()

    # Atualizar limiares se especificados
    if args.min_global:
        checker.THRESHOLDS["global"]["min_coverage"] = args.min_global
    if args.min_critical:
        checker.THRESHOLDS["critical"]["min_coverage"] = args.min_critical

    # Validar cobertura
    passed = checker.validate_coverage(backend_path, frontend_path)

    # Gerar relatório
    output_path = Path(args.output)
    checker.generate_report(output_path)

    # Imprimir resumo
    checker.print_summary()

    # Código de saída
    if not passed:
        sys.exit(1)
    elif args.strict and checker.results["compliance"]["warnings"]:
        print("\n❌ Modo strict: Falha devido a avisos")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
