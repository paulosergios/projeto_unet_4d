import numpy as np
import nibabel as nib
from skimage.transform import resize
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def dice_coefficient(y_true, y_pred):
    """
    Calcula o Dice Coefficient entre duas máscaras binárias.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    sum_masks = np.sum(y_true) + np.sum(y_pred)
    if sum_masks == 0:
        return 1.0  # Se ambas as máscaras estiverem vazias, consideramos perfeito.
    return 2.0 * intersection / sum_masks

# Caminhos dos arquivos NIfTI
pred_path = "C:/Users/paulo/Desktop/predict_4d.nii.gz"
gt_path   = "C:/Users/paulo/Desktop/Dataset/database/testing/patient112/patient112_frame01_gt.nii.gz"

# Carregar as imagens NIfTI
pred_img = nib.load(pred_path)
gt_img   = nib.load(gt_path)

# Obter os dados
pred_data = pred_img.get_fdata()  # Esperado (X, Y, Z, T)
gt_data   = gt_img.get_fdata()    # Esperado (X, Y, Z)

print("Shape da máscara predita (4D):", pred_data.shape)
print("Shape da máscara ground truth (3D):", gt_data.shape)

# Se os shapes forem diferentes, redimensionar a ground truth para o shape da predição (exceto a dimensão temporal)
if gt_data.shape != pred_data.shape[:3]:
    print("Redimensionando a ground truth para o shape da predição...")
    gt_data = resize(gt_data, pred_data.shape[:3], order=0, preserve_range=True, anti_aliasing=False)
    print("Novo shape da ground truth:", gt_data.shape)

# Inicializar listas para armazenar métricas de cada frame
dice_scores = []
iou_scores = []
accuracy_scores = []
sensitivity_scores = []
specificity_scores = []

# Iterar sobre cada frame da máscara predita (4D -> 3D)
num_frames = pred_data.shape[3]
for t in range(num_frames):
    print(f"\nAnalisando frame {t + 1}/{num_frames}...")
    
    pred_bin = (pred_data[:, :, :, t] > 0.5).astype(np.uint8)
    gt_bin   = (gt_data > 0.5).astype(np.uint8)
    
    # Cálculo das métricas
    dice = dice_coefficient(gt_bin, pred_bin)
    iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten())
    acc = accuracy_score(gt_bin.flatten(), pred_bin.flatten())
    cm = confusion_matrix(gt_bin.flatten(), pred_bin.flatten())
    TN, FP, FN, TP = cm.ravel()
    total_pixels = cm.sum()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    dice_scores.append(dice)
    iou_scores.append(iou)
    accuracy_scores.append(acc)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

# Exibir gráfico das métricas ao longo dos frames
time = np.arange(1, num_frames + 1)
plt.figure(figsize=(10, 5))
sns.lineplot(x=time, y=dice_scores, label="Dice Coefficient", marker="o")
sns.lineplot(x=time, y=iou_scores, label="IoU", marker="s")
sns.lineplot(x=time, y=accuracy_scores, label="Accuracy", marker="^")
sns.lineplot(x=time, y=sensitivity_scores, label="Sensitivity", marker="D")
sns.lineplot(x=time, y=specificity_scores, label="Specificity", marker="X")
plt.xlabel("Frame")
plt.ylabel("Score")
plt.title("Métricas ao longo dos frames")
plt.legend()
plt.grid()
plt.show()

print("\nMédia das métricas para todos os frames:")
print(f"Média Dice Coefficient: {np.mean(dice_scores):.4f}")
print(f"Média IoU: {np.mean(iou_scores):.4f}")
print(f"Média Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Média Sensitivity: {np.mean(sensitivity_scores):.4f}")
print(f"Média Specificity: {np.mean(specificity_scores):.4f}")
