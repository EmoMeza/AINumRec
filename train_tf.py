# ── PLAN (TensorFlow/Keras) ───────────────────────────────────────────────────
#
# 1. IMPORTS
#    - numpy para manejo de datos
#    - tensorflow y keras para el modelo
#
# 2. CARGAR DATASET
#    - Leer los 4 archivos .idx ubyte del dataset MNIST
#    - Separar en train (60k) y test (10k)
#
# 3. PREPROCESAR
#    - Normalizar píxeles de uint8 [0-255] a float32 [0.0-1.0]
#    - Añadir dimensión de canal: (N, 28, 28) → (N, 28, 28, 1)  ← TF usa NHWC
#    - Convertir labels a int32
#
# 4. DEFINIR EL MODELO CNN
#    - Conv2D → ReLU → MaxPool
#    - Conv2D → ReLU → MaxPool
#    - Flatten
#    - Dense → ReLU
#    - Dense → salida (10 clases, softmax)
#
# 5. ENTRENAMIENTO
#    - Compilar modelo (optimizer, loss, metrics)
#    - model.fit() con batches y epochs
#
# 6. EVALUACIÓN
#    - model.evaluate() sobre el set de test
#
# 7. GUARDAR MODELO
#    - Guardar en saved_model/
#
# ─────────────────────────────────────────────────────────────────────────────
