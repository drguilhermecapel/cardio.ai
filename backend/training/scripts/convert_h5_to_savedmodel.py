# backend/training/scripts/convert_h5_to_savedmodel.py

import tensorflow as tf
import argparse
import os

def convert_model(h5_path: str, output_dir: str):
    """
    Carrega um modelo Keras de um arquivo .h5 e o salva no formato SavedModel.

    Args:
        h5_path (str): O caminho para o arquivo do modelo .h5.
        output_dir (str): O diretório onde o SavedModel será salvo.
    """
    if not os.path.exists(h5_path):
        print(f"ERRO: Arquivo de entrada .h5 não encontrado em: {h5_path}")
        return

    print(f"Carregando modelo do arquivo: {h5_path}...")
    try:
        # Carrega o modelo Keras completo
        model = tf.keras.models.load_model(h5_path)
        print("Modelo carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar o modelo .h5: {e}")
        return

    print(f"Salvando modelo no formato SavedModel em: {output_dir}...")
    try:
        # Salva o modelo no formato SavedModel
        tf.saved_model.save(model, output_dir)
        print("Conversão para SavedModel concluída com sucesso!")
        print(f"O modelo agora pode ser carregado a partir do diretório: {output_dir}")
    except Exception as e:
        print(f"ERRO ao salvar o modelo no formato SavedModel: {e}")
        return

if __name__ == '__main__':
    # Cria um parser para argumentos de linha de comando para tornar o script mais flexível
    parser = argparse.ArgumentParser(
        description="Converte um modelo Keras .h5 para o formato SavedModel do TensorFlow."
    )
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="Caminho para o arquivo de modelo .h5 de entrada."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Caminho do diretório para salvar o SavedModel."
    )

    args = parser.parse_args()

    # Chama a função de conversão com os argumentos fornecidos
    convert_model(args.input_h5, args.output_dir)

