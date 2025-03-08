import os
import numpy as np
import nibabel as nib

def load_nifti_image(filepath):
    nifti_image = nib.load(filepath)
    image_data = nifti_image.get_fdata()
    return np.array(image_data)

def load_images_and_masks(data_dir, is_testing=False):
    images = []
    masks = []

    start_patient = 101 if is_testing else 1
    end_patient = 151 if is_testing else 101

    for patient_id in range(start_patient, end_patient):
        patient_dir = os.path.join(data_dir, f"patient{patient_id:03d}")
        if not os.path.exists(patient_dir):
            print(f"Erro: A pasta do paciente {patient_id} não existe.")
            continue

        for filename in os.listdir(patient_dir):
            if filename.endswith('_frame01.nii.gz'):
                image_path = os.path.join(patient_dir, filename)
                mask_path = image_path.replace('_frame01.nii.gz', '_frame01_gt.nii.gz')

                if not os.path.exists(mask_path):
                    print(f"Erro: Máscara não encontrada para a imagem {filename}.")
                    continue

                print(f"Carregando imagem de: {image_path}")
                image = load_nifti_image(image_path)
                if image is None or image.size == 0:
                    print(f"Erro: A imagem carregada está vazia: {image_path}")
                    continue
                
                print(f"Carregando máscara de: {mask_path}")
                mask = load_nifti_image(mask_path)
                if mask is None or mask.size == 0:
                    print(f"Erro: A máscara carregada está vazia: {mask_path}")
                    continue

                if len(images) == 0:
                    image_shape = image.shape
                elif image.shape != image_shape:
                    print(f"A imagem {filename} tem um formato diferente. Esperado: {image_shape}, mas encontrado: {image.shape}")
                    continue

                images.append(image)
                masks.append(mask)

    return np.array(images), np.array(masks)

def load_data(data_dir, is_testing=False):
    """Carrega os dados (imagens e máscaras) do diretório especificado para treinamento ou teste."""
    print(f"Buscando arquivos em: {data_dir}")
    images, masks = load_images_and_masks(data_dir, is_testing)
    
    print(f"Dimensões das imagens carregadas: {images.shape}")
    print(f"Dimensões das máscaras carregadas: {masks.shape}")
    
    return images, masks

if __name__ == "__main__":
    train_data_dir = "C:/Users/paulo/Desktop/Dataset/database/training"
    train_images, train_masks = load_data(train_data_dir, is_testing=False)

    test_data_dir = "C:/Users/paulo/Desktop/Dataset/database/testing"
    test_images, test_masks = load_data(test_data_dir, is_testing=True)
