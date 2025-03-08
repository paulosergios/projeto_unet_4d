import os
# Desabilita o formato nativo do MKL e limita o uso de threads
os.environ["TF_ENABLE_MKL_NATIVE_FORMAT"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from script.predict import predict_4d  # Certifique-se de que você tem a função predict_4d implementada
from script.predict import load_nifti_image
from visualization import visualize_results
from keras.models import load_model
import tensorflow as tf

# Limita o paralelismo via API do TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def main():
    # Caminhos para as imagens e o modelo
    image_path = "C:/Users/paulo/Desktop/Dataset/database/testing/patient112/patient112_4d.nii.gz"
    output_path = "C:/Users/paulo/Desktop/predict_4d.nii.gz"
    model_weights_path = "C:/Users/paulo/Desktop/projeto_unet_para_4d/models/best_unet_model.keras"
    true_mask_path = "C:/Users/paulo/Desktop/Dataset/database/testing/patient112/patient112_frame01_gt.nii.gz"

    # Carregando o modelo
    try:
        model = load_model(model_weights_path)
    except Exception as e:
        print("Erro ao carregar o modelo:", e)
        return

    # Realizando a predição 4D
    try:
        predict_4d(model, image_path, output_path, true_mask_path)
    except Exception as e:
        print("Erro na predição 4D:", e)
        return
    
    # Carregando a imagem original
    try:
        original_image = load_nifti_image(image_path)
    except Exception as e:
        print("Erro ao carregar a imagem original:", e)
        return

    # Carregando a máscara verdadeira, se existir
    true_mask = None
    if os.path.exists(true_mask_path):
        try:
            true_mask = load_nifti_image(true_mask_path)
        except Exception as e:
            print("Erro ao carregar a máscara verdadeira:", e)
            return
    
    # Visualizando os resultados
    if os.path.exists(output_path):
        try:
            visualize_results(original_image, output_path, true_mask)
        except Exception as e:
            print("Erro na visualização:", e)
            return

if __name__ == "__main__":
    main()
