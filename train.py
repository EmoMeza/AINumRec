# ── PLAN (MLX — Apple Silicon) ───────────────────────────────────────────────
#
# 1. IMPORTS
#    - numpy para manejo de datos
#    - mlx.core y mlx.nn para el modelo y tensores
#    - mlx.optimizers para el optimizador (SGD, Adam, etc.)
#
# 2. CARGAR DATASET
#    - Leer los 4 archivos .idx ubyte del dataset MNIST
#    - Separar en train (60k) y test (10k)
#
# 3. PREPROCESAR
#    - Normalizar píxeles de uint8 [0-255] a float32 [0.0-1.0]
#    - Añadir dimensión de canal: (N, 28, 28) → (N, 28, 28, 1)  ← MLX usa NHWC
#    - Convertir labels a int32
#
# 4. DEFINIR EL MODELO CNN
#    - Conv2D → ReLU → MaxPool
#    - Conv2D → ReLU → MaxPool
#    - Flatten
#    - Linear → ReLU
#    - Linear → salida (10 clases)
#
# 5. ENTRENAMIENTO
#    - Definir función de loss (cross entropy)
#    - Loop de epochs con batches
#    - Backprop con mx.grad y actualizar pesos
#
# 6. EVALUACIÓN
#    - Calcular accuracy sobre el set de test
#
# 7. GUARDAR MODELO
#    - Guardar pesos en saved_model/
#
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
