
#config.py — Configuración centralizada del proyecto.
#Rutas, hiperparámetros y decisiones de diseño

from __future__ import annotations
import os
from pathlib import Path

# Rutas del proyecto

# Directorio raíz: ia-final/
BASE_DIR = Path(__file__).resolve().parent.parent # .resolve() normaliza la ruta, .parent sube un nivel por vez.
# Carpeta de datos procesados (split.json, ventanas, stats).
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# Carpeta de checkpoints del modelo entrenado.
MODELS_DIR = BASE_DIR / "models"
# Ubicación de las anotaciones procesadas del split.
SPLIT_FILE = PROCESSED_DIR / "split.json"

# Dataset CHB-MIT: se setea por argumento en cada script; este es el default local. En Google Colab se pasa con --data-dir la ruta de Google Drive.
DATASET_DIR = Path(os.environ.get("CHBMIT_DIR", r"C:/Users/Usuario/tesis/physionet.org/files/chbmit/1.0.0",))

# Stats del estandarizador por canal
SCALER_STATS_FILE = PROCESSED_DIR / "scaler_stats.npz"
# Reporte de preprocesado
PROCESSING_REPORT_FILE = PROCESSED_DIR / "processing_report.json"

# Señal y segmentación
# Frecuencia de muestreo en Hz
FS = 256
# Duración de cada ventana de análisis (segundos). 5.12 * 256 = 1310.72.
WIN_SECONDS = 5.12
# Muestras por ventana: redondeamos HACIA ABAJO -> 1310 (par).
# Ventaja del par: solapamiento 50% EXACTO (650/1310) y pooling sin restos.
WIN_SAMPLES = int(FS * WIN_SECONDS)

# Muestras por ventana finales (usadas por todo el pipeline).
WIN_SAMPLES_USED = WIN_SAMPLES
# Solapamiento entre ventanas contiguas (50%).
OVERLAP = 0.5
# Paso (stride) entre inicios de ventana: 50% de la ventana = 655 muestras.
STRIDE_SAMPLES = int(WIN_SAMPLES_USED * (1.0 - OVERLAP))

# Filtrado pasabanda de frecuencias baja (0.5 Hz, deriva lenta) y alta (50 Hz, ruido de red).
LOW_FREQ = 0.5
HIGH_FREQ = 50.0

# Orden del filtro Butterworth (5 es un buen compromiso entre pendiente y estabilidad numérica).
FILTER_ORDER = 5 #q tan "bruscamente" se corta la señal

# Estandarización por canal (estadísticas calculadas SOLO sobre TRAIN)
#  z-score= (x - mu) / sigma.
#  Estándar en pipelines de EEG (MNE/Braindecode). Conserva la forma de la distribución, solo reescala Es más
#  robusta a outliers (artefactos del EEG) que la normalización
SCALER = "zscore" #como opcional puedo poner minmax


# Canales compatibles con el dataset TUEV por si se implementa transfer learning
CHANNELS_TUEV = ["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2"]

N_CHANNELS = len(CHANNELS_TUEV)

# Semilla global para reproducibilidad (datos, batches e inicialización).
SEED = 42

# Split por paciente (inter-paciente, 70/30 proporcional)
TEST_RATIO = 0.30
TRAIN_RATIO = 1.0 - TEST_RATIO
N_VAL_PATIENTS = 2 # Cantidad de pacientes que se reservan de train para VALIDACIÓN.

# Peso relativo de los SEGUNDOS DE CRISIS frente a los ARCHIVOS en el reparto
# codicioso del split. Los segundos de crisis determinan cuántas ventanas
# POSITIVAS caen en test (de eso depende la sensibilidad medida); los archivos
# (~horas) determinan el volumen total (de eso depende el FPR).
# Valor 2.0 = la fracción de crisis pesa el doble que la de archivos: deja
# ambas fracciones de test cerca de 0.30 (0.306 archivos / 0.186 crisis).
# Valor 1.0 = test con muchas horas pero pocas crisis (0.314/0.129).
SPLIT_W_ZSEC = 2.0

# Hiperparámetros de entrenamiento
# Optimizador AdamW: ajuste adaptativo por peso + decaimiento de pesos (regularización contra overfitting). Learning rate de arranque.
LEARNING_RATE = 3e-4
# Batch: cantidad de ventanas que ve la red antes de cada paso de gradiente.
BATCH_SIZE = 64
# Máximo de épocas (una época = 1 pasada por todas las ventanas muestreadas).
EPOCHS = 40
# Early stopping: cortar si la pérdida de validación no mejora en 8 épocas.
PATIENCE = 8
# Para entrenamiento con 1 sola GPU / CPU (evita warning de torch).
DEVICE = "cpu"  # será "cuda" en Colab (se setea al entrenar).

# Estrategia de desbalance (train)
# Ratios de negative:positive en cada batch de train.
# Ej: 3 => 3 ventanas no-crisis por cada ventana de crisis.
NEG_POS_RATIO = 3
# Dropout para regularización del modelo.
DROPOUT = 0.4 #en cada lote apaga al 40% de las neuronas al azar, así no se sobreajusta

# Umbral de decisión de la clasificación binaria:
#la red no devuelve un sí o un no, sino un valor entre 0 y 1, q es la probabilidad de q haya crisis. Si la probabilidad es mayor al
#treshold, se considera q la red predijo crisis. Es ajustable para priorizar sensibilidad o especificidad
THRESHOLD = 0.6


# Arquitectura CNN-1D. se usa por model.py
# Filtros (canales) por cada bloque Conv1D.
CONV_CHANNELS = [64, 128, 256, 256, 256, 256]
# Kernel de cada bloque Conv1D.
CONV_KERNELS = [7, 5, 5, 3, 3, 3]
# Neurons de las capas densas finales.
FC_UNITS = 256