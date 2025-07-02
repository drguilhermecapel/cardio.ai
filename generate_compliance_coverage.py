#!/usr/bin/env python3
"""
Script para gerar relatório consolidado de cobertura para ANVISA/FDA
"""

import json
import os
from pathlib import Path
from datetime import datetime


class ComplianceCoverageReporter:
    def __init__(self):
        self.project_root = Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = self.project_root / f"compliance_reports_{self.timestamp}"
        self.report_dir.mkdir(exist_ok=True)

    def collect_frontend_coverage(self):
        """Coleta cobertura do frontend"""
        print("📊 Coletando cobertura do Frontend...")

        coverage_file = (
            self.project_root / "frontend" / "coverage" / "coverage-final.json"
        )
        if coverage_file.exists():
            with open(coverage_file) as f:
                return json.load(f)
        return None

    def collect_backend_coverage(self):
        """Coleta cobertura do backend"""
        print("📊 Coletando cobertura do Backend...")

        coverage_file = self.project_root / "backend" / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                return json.load(f)
        return None

    def generate_anvisa_report(self):
        """Gera relatório formato ANVISA"""
        print("🇧🇷 Gerando relatório ANVISA...")

        report_path = self.report_dir / "ANVISA_Cobertura_Testes.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Relatório de Cobertura de Testes - ANVISA RDC 40/2015\n\n")
            f.write(f"**Data**: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"**Sistema**: CardioAI Pro v1.0\n")
            f.write(
                f"**Classificação**: Classe II - Software como Dispositivo Médico\n\n"
            )

            # Adicionar análise de cobertura aqui

        print(f"✅ Relatório ANVISA gerado: {report_path}")

    def generate_fda_report(self):
        """Gera relatório formato FDA"""
        print("🇺🇸 Gerando relatório FDA...")

        report_path = self.report_dir / "FDA_V&V_Coverage_Report.md"

        with open(report_path, "w") as f:
            f.write("# Software V&V Coverage Report - FDA 21 CFR Part 820.30\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Device**: CardioAI Pro v1.0\n")
            f.write(f"**Classification**: Class II Medical Device Software\n\n")

            # Adicionar análise de cobertura aqui

        print(f"✅ Relatório FDA gerado: {report_path}")

    def generate_reports(self):
        """Gera todos os relatórios"""
        print("\n🏥 Gerando Relatórios de Compliance Regulatório...\n")

        self.generate_anvisa_report()
        self.generate_fda_report()

        print(f"\n✅ Relatórios gerados em: {self.report_dir}")


if __name__ == "__main__":
    reporter = ComplianceCoverageReporter()
    reporter.generate_reports()
