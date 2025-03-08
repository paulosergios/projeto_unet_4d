import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from skimage import morphology
import os

def load_nifti_image(filepath):
    """Carrega uma imagem NIfTI e retorna seus dados."""
    print(f"Carregando imagem NIfTI de {filepath}...")
    nifti_image = nib.load(filepath)
    data = np.array(nifti_image.get_fdata())
    print(f"Imagem carregada com shape: {data.shape}")
    return data

def save_nifti_image(filepath, image_data):
    """Salva uma imagem NIfTI no caminho especificado."""
    print(f"Tentando salvar a imagem NIfTI em {filepath}...")
    nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))
    nib.save(nifti_img, filepath)
    print(f"Imagem salva com sucesso em {filepath}!")

def smooth_mask(mask, sigma=1):
    """Aplica suavização na máscara."""
    print("Aplicando suavização na máscara...")
    return gaussian_filter(mask, sigma=sigma)

def postprocess_mask(mask, percentile=90):
    """Processa a máscara: suaviza, binariza e remove objetos pequenos."""
    print("Processando a máscara predita...")
    smoothed_mask = smooth_mask(mask)
    # O threshold pode ser adaptado conforme sua necessidade
    threshold_value = 0.1
    print(f"Valor de limiar para binarização: {threshold_value}")
    binary_mask = (smoothed_mask > threshold_value).astype(np.uint8)
    print("Removendo objetos pequenos da máscara binarizada...")
    processed_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=500).astype(np.uint8)
    return processed_mask

def predict_4d(model, image_path, output_path, true_mask_path):
    """Realiza a predição da máscara 4D, iterando sobre cada volume 3D."""
    try:
        print("Iniciando predição 4D...")
        
        # Carrega a imagem 4D
        nifti_image = load_nifti_image(image_path)
        if len(nifti_image.shape) != 4:
            raise ValueError("A imagem deve ser um volume 4D (ex: (X, Y, Z, T)).")
        
        n_frames = nifti_image.shape[3]
        processed_masks = []
        
        # Se existir, carrega a ground truth (que pode ser 3D ou 4D)
        true_mask_data = load_nifti_image(true_mask_path)
        
        for t in range(n_frames):
            print(f"\nProcessando frame {t+1}/{n_frames}...")
            # Extrai o volume 3D para o frame atual
            volume_3d = nifti_image[..., t]
            
            # Redimensiona o volume para o shape esperado pelo modelo.
            # Por exemplo, se o modelo espera (216, 256, 32), calcule os scale factors:
            scale_factors = (
                216 / volume_3d.shape[0],
                256 / volume_3d.shape[1],
                32  / volume_3d.shape[2]
            )
            resized_volume = zoom(volume_3d, scale_factors, order=1).astype('float32') / 255.0
            print(f"Frame redimensionado para: {resized_volume.shape}")
            
            # Adiciona as dimensões de canal e batch para o modelo
            resized_volume = np.expand_dims(resized_volume, axis=-1)  # canal
            resized_volume = np.expand_dims(resized_volume, axis=0)    # batch
            print(f"Shape para predição: {resized_volume.shape}")
            
            # Realiza a predição para o frame atual
            prediction = model.predict(resized_volume)
            pred_mask = prediction[0, :, :, :, 0]  # assume que o modelo retorna shape (1, X, Y, Z, 1)
            print(f"Shape da máscara predita: {pred_mask.shape}")
            
            # Ajuste da máscara para as dimensões físicas originais
            # Utilize a ground truth para definir o tamanho físico desejado.
            if t == 0:
                gt_img = nib.load(true_mask_path)
                gt_header = gt_img.header
                gt_spacing = gt_header.get_zooms()[:3]
                gt_shape = gt_img.shape[:3]
                target_physical_size = [gt_spacing[i] * gt_shape[i] for i in range(3)]
                print(f"Tamanho físico alvo: {target_physical_size}")
            
            # Assumindo que o espaçamento atual é de 1.0 mm para cada eixo
            current_physical_size = [pred_mask.shape[i] * 1.0 for i in range(3)]
            resize_factors_phys = [
                target_physical_size[i] / current_physical_size[i] for i in range(3)
            ]
            print(f"Fatores de redimensionamento físico: {resize_factors_phys}")
            
            adjusted_pred_mask = zoom(pred_mask, resize_factors_phys, order=0)
            print(f"Shape da máscara ajustada: {adjusted_pred_mask.shape}")
            
            # Pós-processamento da máscara
            pred_mask_processed = postprocess_mask(adjusted_pred_mask, percentile=90)
            
            # Se a ground truth for 4D, extraia o frame correspondente; senão, use-a diretamente
            if len(true_mask_data.shape) == 4:
                true_mask_frame = true_mask_data[..., t]
            else:
                true_mask_frame = true_mask_data
            
            # Garante que as shapes batam para a multiplicação (refinamento da máscara)
            if true_mask_frame.shape != pred_mask_processed.shape:
                true_mask_frame = zoom(true_mask_frame, (
                    pred_mask_processed.shape[0] / true_mask_frame.shape[0],
                    pred_mask_processed.shape[1] / true_mask_frame.shape[1],
                    pred_mask_processed.shape[2] / true_mask_frame.shape[2]
                ), order=0)
            
            # Refinamento: manter somente a ROI presente na ground truth
            refined_mask = pred_mask_processed * (true_mask_frame > 0).astype(np.uint8)
            processed_masks.append(refined_mask)
        
        # Empilha as máscaras de cada frame para formar o volume 4D final
        refined_mask_4d = np.stack(processed_masks, axis=-1)
        print(f"\nShape final da máscara 4D: {refined_mask_4d.shape}")
        
        # Salva a máscara processada
        save_nifti_image(output_path, refined_mask_4d)
        print("Predição 4D concluída com sucesso!")
    
    except OSError as e:
        print(f"Erro de sistema operacional: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")