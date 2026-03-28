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

import struct
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# ── 2. CARGAR DATASET ─────────────────────────────────────────────────────────

def load_images(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        rows, cols = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load the dataset
train_images = load_images('dataset/train-images.idx3-ubyte')
train_labels = load_labels('dataset/train-labels.idx1-ubyte')
test_images  = load_images('dataset/t10k-images.idx3-ubyte')
test_labels  = load_labels('dataset/t10k-labels.idx1-ubyte')

print(f"Train: {train_images.shape} | Test: {test_images.shape}")

# ── 3. PREPROCESAR ────────────────────────────────────────────────────────────

train_images = train_images.astype(np.float32) / 255.0
train_images = np.expand_dims(train_images, axis=-1)

train_labels = train_labels.astype(np.int32)

test_images = test_images.astype(np.float32) / 255.0
test_images = np.expand_dims(test_images, axis=-1)

test_labels = test_labels.astype(np.int32)

# ── 4. DEFINIR EL MODELO CNN ─────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(1600, 128)
        self.fc2   = nn.Linear(128, 10)

    def __call__(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        return x

# ── 5. ENTRENAMIENTO ─────────────────────────────────────────────────────────

model      = CNN()
optimizer  = optim.Adam(learning_rate=0.001)
EPOCHS     = 10
BATCH_SIZE = 32

def loss_fn(model, X, y):
    logits = model(X)
    return mx.mean(nn.losses.cross_entropy(logits, y))

loss_and_grad = nn.value_and_grad(model, loss_fn)

for epoch in range(EPOCHS):
    for i in range(0, len(train_images), BATCH_SIZE):
        X_batch = mx.array(train_images[i:i+BATCH_SIZE])
        y_batch = mx.array(train_labels[i:i+BATCH_SIZE])
        loss, grads = loss_and_grad(model, X_batch, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        batch = i // BATCH_SIZE
        if batch % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch} | Loss: {loss.item():.4f}")

# ── 6. EVALUACIÓN ─────────────────────────────────────────────────────────────

correct = 0
for i in range(0, len(test_images), BATCH_SIZE):
    X_batch    = mx.array(test_images[i:i+BATCH_SIZE])
    y_batch    = mx.array(test_labels[i:i+BATCH_SIZE])
    logits     = model(X_batch)
    predictions = mx.argmax(logits, axis=1)
    correct    += mx.sum(predictions == y_batch).item()

accuracy = correct / len(test_images)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ── 7. GUARDAR MODELO ─────────────────────────────────────────────────────────

# TODO(human): Guarda los pesos del modelo en saved_model/
# Pista: necesitas crear la carpeta si no existe, y usar model.save_weights()
# El archivo debería llamarse algo como 'saved_model/cnn_weights.npz'

import os 

if not os.path.exists("saved_model"):
    os.makedirs("saved_model")

save_path = "saved_model/cnn_weights.npz"

model.save_weights(save_path)

print(f"Model weights saved to {save_path}")
