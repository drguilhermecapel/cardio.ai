#!/usr/bin/env python3
"""
Script principal para executar todos os próximos passos do CardioAI Pro
Executa sequencialmente: Download datasets → Setup GPU → Treinamento IA
"""

import logging
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """Configura logging para o script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("complete_setup.log")],
    )


def run_script(script_path: str, description: str) -> bool:
    """Executa um script Python e retorna se foi bem-sucedido"""
    logger = logging.getLogger(__name__)

    logger.info(f"🚀 Executando: {description}")
    logger.info(f"   Script: {script_path}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=Path(script_path).parent,
        )

        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCESSO")
            if result.stdout:
                logger.info("Output:")
                for line in result.stdout.split("\n")[-10:]:  # Últimas 10 linhas
                    if line.strip():
                        logger.info(f"   {line}")
            return True
        else:
            logger.error(f"❌ {description} - FALHA")
            logger.error(f"Código de saída: {result.returncode}")
            if result.stderr:
                logger.error("Erro:")
                for line in result.stderr.split("\n")[-5:]:  # Últimas 5 linhas de erro
                    if line.strip():
                        logger.error(f"   {line}")
            return False

    except Exception as e:
        logger.error(f"❌ Erro ao executar {description}: {e}")
        return False


def main():
    """Função principal para executar todos os próximos passos"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🎯 CARDIOAI PRO - EXECUÇÃO COMPLETA DOS PRÓXIMOS PASSOS")
    logger.info("=" * 70)
    logger.info("Implementando:")
    logger.info("1. Download dos Datasets (MIT-BIH, PTB-XL, PhysioNet Challenges)")
    logger.info("2. Setup de Hardware (GPU NVIDIA/MPS/CPU)")
    logger.info("3. Implementação de modelo híbrido de IA")
    logger.info("4. Treinamento da IA baseado nos datasets")
    logger.info("=" * 70)

    scripts_dir = Path(__file__).parent
    success_count = 0
    total_steps = 3

    logger.info("\n" + "=" * 60)
    logger.info("STEP 1/3: DOWNLOAD E SETUP DOS DATASETS")
    logger.info("=" * 60)

    setup_datasets_script = scripts_dir / "setup_datasets.py"
    if setup_datasets_script.exists():
        if run_script(str(setup_datasets_script), "Download e setup dos datasets"):
            success_count += 1
            logger.info("✅ Datasets configurados com sucesso!")
        else:
            logger.error("❌ Falha na configuração dos datasets")
    else:
        logger.error(f"❌ Script não encontrado: {setup_datasets_script}")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2/3: SETUP DE GPU E CONFIGURAÇÃO")
    logger.info("=" * 60)

    setup_gpu_script = scripts_dir / "setup_gpu_training.py"
    if setup_gpu_script.exists():
        if run_script(
            str(setup_gpu_script), "Setup de GPU e configuração de treinamento"
        ):
            success_count += 1
            logger.info("✅ GPU e treinamento configurados com sucesso!")
        else:
            logger.error("❌ Falha na configuração de GPU/treinamento")
    else:
        logger.error(f"❌ Script não encontrado: {setup_gpu_script}")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3/3: TREINAMENTO DO MODELO HÍBRIDO")
    logger.info("=" * 60)

    train_model_script = scripts_dir / "train_model.py"
    if train_model_script.exists():
        if run_script(str(train_model_script), "Treinamento do modelo híbrido de IA"):
            success_count += 1
            logger.info("✅ Modelo treinado com sucesso!")
        else:
            logger.error("❌ Falha no treinamento do modelo")
    else:
        logger.error(f"❌ Script não encontrado: {train_model_script}")

    logger.info("\n" + "=" * 70)
    logger.info("RELATÓRIO FINAL - EXECUÇÃO COMPLETA")
    logger.info("=" * 70)

    logger.info(f"📊 Passos concluídos: {success_count}/{total_steps}")

    if success_count == total_steps:
        logger.info("🎉 TODOS OS PRÓXIMOS PASSOS CONCLUÍDOS COM SUCESSO!")
        logger.info("")
        logger.info("✅ Resultados alcançados:")
        logger.info("   • Datasets baixados e configurados")
        logger.info("   • GPU/hardware configurado automaticamente")
        logger.info("   • Modelo híbrido CNN-BiLSTM-Transformer implementado")
        logger.info("   • IA treinada com base nos datasets para melhor acurácia")
        logger.info("")
        logger.info("🚀 CardioAI Pro está pronto para análise avançada de ECGs!")
        logger.info("   • Acurácia alvo: 99.6% (conforme especificações)")
        logger.info("   • Arquitetura híbrida: CNN + BiLSTM + Transformer")
        logger.info("   • Features multimodais: 1D + 2D + Wavelets")
        logger.info("   • Datasets: MIT-BIH + PTB-XL + PhysioNet Challenges")

    elif success_count > 0:
        logger.warning(
            f"⚠️  EXECUÇÃO PARCIAL: {success_count}/{total_steps} passos concluídos"
        )
        logger.info("Verifique os logs acima para detalhes dos erros")

    else:
        logger.error("❌ FALHA COMPLETA: Nenhum passo foi concluído com sucesso")
        logger.info("Verifique os logs de erro e dependências do sistema")

    logger.info(f"\n📝 Log completo salvo em: complete_setup.log")


if __name__ == "__main__":
    main()
