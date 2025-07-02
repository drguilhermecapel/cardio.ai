#!/usr/bin/env python3
"""
ECG Analyzer - Script Principal
Uso: python analyze_ecg.py <arquivo_ecg.pdf>
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ecg_analyzer import ECGAnalyzerStandalone
import argparse

def main():
    parser = argparse.ArgumentParser(description='Analisar ECG')
    parser.add_argument('input', help='Arquivo de ECG (PDF/JPG/JPEG)')
    parser.add_argument('--show-signals', action='store_true',
                       help='Mostrar sinais extraídos')
    parser.add_argument('--save-report', action='store_true',
                       help='Salvar relatório')

    args = parser.parse_args()

    # Carregar configuração
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Inicializar analisador
    model_path = os.path.join('..', config['model']['path'])
    analyzer = ECGAnalyzerStandalone(model_path)

    # Analisar
    result = analyzer.analyze(args.input, show_signals=args.show_signals)

    # Salvar relatório se solicitado
    if args.save_report and result['success']:
        report_file = args.input.replace('.pdf', '_report.json').replace('.jpg', '_report.json')
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n📄 Relatório salvo em: {report_file}")

if __name__ == "__main__":
    main()
