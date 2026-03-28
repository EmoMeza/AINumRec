# Apuntes — AINumRec CNN con MLX

---

## 1. Bits, Bytes y tipos de datos

### La diferencia fundamental

- **1 bit** = 0 o 1
- **1 byte** = 8 bits = puede representar valores del 0 al 255
- **uint8** = unsigned integer de 8 bits = exactamente 1 byte por valor → rango 0-255

Un píxel blanco en memoria: `11111111` → 255
Un píxel negro en memoria: `00000000` → 0
Un píxel gris en memoria: `10000000` → 128

Las imágenes de MNIST son **escala de grises** — no solo blanco o negro, sino 256 tonos. Cada píxel ocupa exactamente 1 byte.

### ¿Por qué uint8 y no otro tipo?

Porque MNIST fue creado así deliberadamente. Los datos **ya estaban guardados como bytes** en el archivo — `np.frombuffer` no convierte nada, solo le dice a numpy "toma estos bytes y tratalos como uint8". Si hubieras puesto `dtype=np.float32`, numpy tomaría los mismos bytes y los interpretaría como floats de 4 bytes — obtendrías basura.

---

## 2. El formato IDX de MNIST

### ¿Quién lo define?

**Yann LeCun** — el investigador que creó MNIST en los 90s. No es un estándar ISO ni un formato oficial. Es una especificación ad-hoc publicada en su sitio web. Funciona solo porque está documentado y todo el mundo usa MNIST.

> **Lección importante:** En ciencia de datos, los formatos binarios sin documentación son ilegibles. Si LeCun no hubiera publicado la especificación, tendrías que hacer ingeniería inversa. Por eso los datasets modernos usan HDF5 o Parquet — formatos auto-descriptivos donde el esquema está dentro del archivo.

### Estructura del header

El nombre `idx3` y `idx1` indica la cantidad de dimensiones de los datos:

**Imágenes (`idx3` = 3 dimensiones: cantidad × filas × columnas):**
```
Byte 0-3   │ Magic number (0x00000803) → "soy archivo de imágenes MNIST"
Byte 4-7   │ Cantidad de imágenes
Byte 8-11  │ Número de filas (28)
Byte 12-15 │ Número de columnas (28)
Byte 16+   │ Píxeles (N × 28 × 28 bytes)
```

**Labels (`idx1` = 1 dimensión: solo cantidad):**
```
Byte 0-3   │ Magic number (0x00000801) → "soy archivo de labels"
Byte 4-7   │ Cantidad de labels
Byte 8+    │ Labels (N bytes, valores 0-9)
```

El magic number existe para verificar que el archivo está correcto. Si lees ese valor y no coincide → archivo corrupto o formato equivocado.

---

## 3. Lectura de archivos binarios

### `struct.unpack` — leer bytes con formato

```python
magic, num = struct.unpack('>II', f.read(8))
rows, cols = struct.unpack('>II', f.read(8))
```

**Desglose de `'>II'`:**
- `>` → big-endian: el byte más significativo va primero (así lo guarda MNIST)
- `I` → unsigned int de 32 bits (4 bytes)
- `II` → dos de esos seguidos → lee 8 bytes y los divide en dos enteros

**Big-endian vs Little-endian:**
El número 10000 en hexadecimal es `0x00002710`. En big-endian se guarda como `00 00 27 10`. En little-endian (como usa x86) sería `10 27 00 00`. Si mezclas el orden lees un número incorrecto sin error visible.

### El cursor del archivo

Cuando abres un archivo con `open()`, Python mantiene una posición actual (cursor). Cada `f.read(n)` lee `n` bytes **desde donde está el cursor** y lo avanza automáticamente:

```
Posición: │ 0   │ 4   │ 8   │ 12  │ 16  │ píxeles...
          │magic│ num │rows │cols │

f.read(8) → lee bytes 0-7,  cursor queda en 8
f.read(8) → lee bytes 8-15, cursor queda en 16
f.read()  → lee TODO desde 16 hasta el final
```

Puedes verificarlo con `f.tell()` que devuelve la posición actual. También puedes mover el cursor con `f.seek(16)` para saltar directamente sin leer.

### Dos formas de leer el mismo archivo

```python
# Opción 1: con cursor (lee y valida el header)
with open(path, 'rb') as f:
    magic, num = struct.unpack('>II', f.read(8))
    rows, cols = struct.unpack('>II', f.read(8))
    data = np.frombuffer(f.read(), dtype=np.uint8)

# Opción 2: con offset (salta el header directo)
with open(path, 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
```

La opción 1 es más explícita y te da acceso a los valores del header (num, rows, cols). La opción 2 es más corta pero no valida nada.

### `np.frombuffer` y reshape

```python
data = np.frombuffer(f.read(), dtype=np.uint8)
# → array plano: [255, 0, 128, 45, ...]  ← 7,840,000 números seguidos

images = data.reshape(-1, 28, 28)
# → (10000, 28, 28)
```

**¿Cómo sabe reshape cuántas imágenes hay?** El `-1` le dice "calcula tú": `7,840,000 ÷ (28×28) = 10,000`. Si el archivo tiene un byte extra, da error — lo que también sirve como validación.

**¿Por qué no hace reshape en labels?** Porque cada label es un solo número — ya es un array 1D. No hay dimensiones espaciales que reorganizar. `labels[42]` te da directamente el label de la imagen 42.

### Imagen y label están sincronizados por posición

No hay IDs ni punteros. Son dos arrays paralelos:
```
images[0]  ←→  labels[0]    (la imagen 0 es el dígito que dice labels[0])
images[42] ←→  labels[42]
```
Si mezclas el orden de uno sin mezclar el otro, todo el dataset queda incorrecto silenciosamente.

---

## 4. Preprocesamiento

### Paso 1 — Normalizar: `uint8 [0-255]` → `float32 [0.0-1.0]`

```python
images = images.astype(np.float32) / 255.0
```

**¿Por qué float32 y no float64?**
- float32 = 4 bytes/número → 60k imágenes ≈ 188 MB
- float64 = 8 bytes/número → 60k imágenes ≈ 376 MB
- Las redes no necesitan tanta precisión. float32 es el estándar en deep learning.

**¿Por qué dividir por 255?**
Las funciones de activación fueron diseñadas para inputs pequeños. Con valores 0-255, los gradientes durante el entrenamiento se vuelven muy grandes → los pesos se actualizan demasiado → entrenamiento inestable. Con 0.0-1.0 los gradientes son manejables.

### Paso 2 — Standardization (alternativa más robusta)

```python
media = train_images.mean()   # ≈ 0.1307 para MNIST
std   = train_images.std()    # ≈ 0.3081 para MNIST
x_std = (images - media) / std
# resultado: valores centrados en 0, mayoría entre -2 y +2
```

**¿Por qué media=0 importa?** Si todos tus inputs son positivos (0 a 1), los gradientes siempre van en la misma dirección. Con media=0 hay positivos y negativos → los gradientes se equilibran → aprendizaje más directo en ambas direcciones.

Los valores `0.1307` y `0.3081` de MNIST son tan conocidos que se hardcodean directamente en la mayoría de implementaciones.

**¿Cuándo usar cada uno?**
| | Normalización `/255` | Standardization |
|---|---|---|
| MNIST (simple, uniforme) | ✓ suficiente | también funciona |
| Fotos naturales complejas | menos ideal | mejor opción |

### Paso 3 — Añadir dimensión de canal

```python
images = np.expand_dims(images, axis=-1)
# (60000, 28, 28) → (60000, 28, 28, 1)
```

El `1` es el canal de color. Gris = 1 canal, RGB = 3 canales. MLX y TensorFlow usan formato **NHWC**: (batch, height, width, channels).

### Paso 4 — Convertir labels

```python
labels = labels.astype(np.int32)
```

Solo cambia el tipo, sin dividir. Los labels son enteros (0-9), no floats.

---

## 5. La arquitectura CNN

### ¿Qué es una CNN?

Una red neuronal convolucional procesa datos respetando su estructura espacial. A diferencia de una red fully-connected (que aplanaría los 784 píxeles y perdería toda información de vecindad), una CNN sabe que el píxel `[2][3]` es vecino del `[2][4]`.

### Conv2D — el corazón de la CNN

```python
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
```

Un filtro (kernel) de 3×3 se desliza sobre la imagen. En cada posición multiplica los 9 píxeles por los 9 pesos del filtro y suma → 1 número. Ese número indica "qué tanto esta zona contiene el patrón que busca el filtro".

```
Zona de la imagen:    Filtro (pesos aprendidos):    Resultado:
  0.1  0.8  0.2         0.5   0.0  -0.5
  0.0  1.0  0.9    ×    0.5   0.0  -0.5    →   suma = 1 número
  0.0  0.8  0.3         0.5   0.0  -0.5
```

El filtro se desliza por TODA la imagen → produce un **feature map** (mapa de dónde encontró ese patrón).

**Parámetros:**
- **`in_channels`** — cuántos planos tiene el input. No es sobre bordes — es sobre capas de información. Gris=1, RGB=3, output de conv anterior = lo que sea.
- **`out_channels`** — cuántos filtros distintos aplicas. 32 filtros = 32 feature maps = 32 "preguntas" distintas sobre la imagen. **Los filtros no se diseñan a mano — se aprenden durante el entrenamiento.**
- **`kernel_size`** — tamaño del filtro. 3×3 es el estándar para casi todo.

**Restricción obligatoria:** `in_channels` de cada Conv debe coincidir exactamente con `out_channels` de la capa anterior. Si no → crash en runtime. No es convención, es matemática.

**Pérdida de bordes:** Un kernel 3×3 no puede centrarse en el borde porque no tiene vecinos de un lado. Por eso el output pierde 1 píxel de cada borde → `28×28 → 26×26`. Fórmula: `tamaño - (kernel_size - 1)`. Si quieres mantener el tamaño usa `padding=1`.

### ReLU

```
f(x) = max(0, x)
```

**No se aplica sobre los píxeles originales** — se aplica sobre los outputs de Conv. Los filtros tienen pesos que pueden ser negativos. El output de una convolución sobre píxeles [0,1] puede perfectamente ser negativo:

```
píxeles: [0.3, 0.8, 0.5]    pesos: [0.5, -1.2, 0.8]
output:  0.3×0.5 + 0.8×(-1.2) + 0.5×0.8 = -0.41
```

ReLU convierte ese `-0.41` en `0`, diciéndole a la red: **"esta feature no está presente aquí"**.

Sin ReLU, no importa cuántas capas pongas — la red entera se comporta como una sola multiplicación lineal y no puede aprender patrones complejos.

**¿Por qué no ReLU en la última capa?** La última capa produce logits (números crudos, no probabilidades). Si aplicaras ReLU, los negativos se harían 0 y perderías información sobre clases poco probables. La función de loss necesita esos valores tal cual.

### MaxPool

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

- **`kernel_size=2`** — mira bloques de 2×2 (4 píxeles)
- **`stride=2`** — se mueve de 2 en 2 (sin solaparse)

Toma el **máximo** de cada bloque. Reduce el tamaño a la mitad.

```
Feature map:                MaxPool(2×2, stride=2):
  0.1  0.8  │  0.2  0.5         0.8   0.5
  0.5  0.3  │  0.1  0.4    →    0.7   0.3
  ──────────┼──────────
  0.7  0.2  │  0.3  0.1
  0.1  0.5  │  0.0  0.3
```

**Opera independientemente sobre cada canal** — no mezcla los 32 o 64 canales entre sí. Cada feature map se reduce por separado.

**¿Por qué el máximo y no el promedio?** El máximo preserva "esta feature estuvo presente aquí". El promedio la diluiría. (También existe AvgPool pero MaxPool es más común para detección de features.)

**Tres razones para reducir tamaño:**
1. Velocidad — menos datos en capas siguientes
2. Memoria — importante en GPU/Metal
3. Tolerancia a desplazamiento — si el dígito está corrido 1px, el máximo del bloque sigue siendo el mismo

### Reshape (Flatten)

```python
x = x.reshape(x.shape[0], -1)
# (batch, 5, 5, 64) → (batch, 1600)
```

Mismo dato, distinta organización. No se pierde información. Es el puente entre las capas Conv (que trabajan en 2D espacial) y las capas Linear (que trabajan con vectores 1D).

- **`x.shape[0]`** → mantén el batch
- **`-1`** → calcula tú el resto: `5×5×64 = 1600`

Usar `-1` evita hardcodear el número — si cambias la arquitectura, se adapta solo.

### Linear (Fully Connected)

```python
nn.Linear(1600, 128)   # fc1: comprime features
nn.Linear(128, 10)     # fc2: 10 salidas (una por dígito)
```

Cada neurona está conectada a **todos** los valores del vector anterior. Aprende cómo combinar globalmente las features detectadas por las Conv para tomar la decisión final.

El `10` en fc2 no es un hiperparámetro — es fijo porque hay exactamente 10 clases (0-9). Si quisieras clasificar 1001 números, pondrías `Linear(128, 1001)` — funcionaría, pero necesitarías millones de ejemplos de entrenamiento.

---

## 6. Evolución de shapes — cálculo completo

```
entrada:             (batch, 28, 28, 1)
conv1 (kernel=3):    (batch, 26, 26, 32)   ← 28-(3-1)=26
relu:                (batch, 26, 26, 32)   ← misma forma, valores cambiados
pool (2×2, s=2):     (batch, 13, 13, 32)   ← 26/2=13
conv2 (kernel=3):    (batch, 11, 11, 64)   ← 13-(3-1)=11
relu:                (batch, 11, 11, 64)
pool (2×2, s=2):     (batch,  5,  5, 64)   ← 11/2=5 (redondeo abajo)
reshape:             (batch,   1600)        ← 5×5×64=1600
fc1:                 (batch,    128)
relu:                (batch,    128)
fc2:                 (batch,     10)        ← salida final
```

**Regla Conv:** `salida = entrada - (kernel_size - 1)`
**Regla Pool:** `salida = entrada / stride` (redondeado abajo)

> **Tip práctico:** Nadie hace este cálculo a mano para modelos complejos. Agrega prints temporales en `__call__`, corre una pasada con un batch falso, lee los shapes, y ya tienes el número para `fc1`. Luego borras los prints.

---

## 7. Batches y entrenamiento

No se entrena con 1 imagen a la vez ni con todo el dataset de una — se entrena en **lotes (batches)**:

```
Dataset: 60,000 imágenes
Batch size: 32

Paso 1:    imágenes 0-31    → 32 predicciones → error promedio → ajusta pesos
Paso 2:    imágenes 32-63   → ...
...
Paso 1875: imágenes 59968+  → 1 epoch completo (recorrió todo el dataset)
```

**¿Por qué no 1 imagen?** El gradiente de 1 imagen es muy ruidoso — cada imagen es un caso especial. El promedio de 32 imágenes da una dirección de actualización más estable.

**¿Por qué no todo el dataset?** No cabe en memoria para datasets grandes. Y actualizar los pesos solo una vez por epoch aprende muy lento.

**Batch size como hiperparámetro:**
- Batches pequeños (8-16) → generalizan mejor, más ruidosos, más lentos
- Batches grandes (128-256) → más rápidos, pueden converger peor
- En Apple Silicon con memoria unificada puedes usar batches más grandes

---

## 8. Canales: cuántos usar y cuándo

**El patrón estándar:** duplicar canales en cada capa Conv.
```
Conv1: 32 canales   (features simples: bordes, texturas básicas)
Conv2: 64 canales   (features complejas: combinaciones de las anteriores)
```

No es obligatorio duplicar — es una convención. Las capas más profundas necesitan más capacidad para representar combinaciones más complejas.

**¿Qué pasa si eliges mal?**
```
Muy pocos → underfitting: accuracy bajo en train Y test
Demasiados → overfitting: accuracy alto en train, bajo en test
```

**¿Por qué potencias de 2 (32, 64, 128...)?** No es matemáticamente obligatorio. Las GPUs y Apple Silicon procesan en bloques optimizados para potencias de 2. Usar 30 o 33 funciona, pero es marginalmente menos eficiente en hardware. Es más cultura que regla.

---

## 9. Conv1D, Conv2D, Conv3D

El mismo concepto en distintas dimensiones:

| | Datos | Ejemplo |
|---|---|---|
| Conv1D | Secuencias 1D | Audio, texto, series de tiempo |
| Conv2D | Imágenes 2D | Fotos, grises, MNIST |
| Conv3D | Volúmenes 3D | Video (2D espacio + 1D tiempo), escáneres médicos |

No existe Conv4D+ porque el costo computacional crece exponencialmente. Para datos de alta dimensionalidad se usan Transformers o arquitecturas híbridas.

---

## 10. Arquitecturas multi-input (Late Fusion)

Si quieres combinar una imagen con metadata (promedio, contraste, gamma...):

```python
def __call__(self, imagen, metadata):
    x = self.conv1(imagen)
    # ... capas conv ...
    x = x.reshape(x.shape[0], -1)          # (batch, 1600)

    combined = mx.concatenate([x, metadata], axis=-1)  # (batch, 1603)
    x = nn.relu(self.fc1(combined))
    return self.fc2(x)
```

Cada tipo de dato tiene su propio "camino" y se fusionan antes de las capas fc. Esto se llama **Late Fusion** y es la forma estándar de combinar inputs heterogéneos.

---

## 11. Patrón estándar CNN — resumen

```
[Conv → ReLU → Pool] × N    ← extracción de features (repite N veces)
        ↓
     Reshape                 ← una sola vez, puente 3D→1D
        ↓
[Linear → ReLU] × M         ← clasificación (repite M veces)
        ↓
  Linear                     ← sin ReLU, tamaño = número de clases
```

Cuándo usar cada capa:
- **Conv** → siempre seguida de ReLU
- **ReLU** → siempre después de Conv o Linear (excepto la última)
- **Pool** → después de Conv+ReLU para reducir tamaño
- **Reshape** → una sola vez, justo antes de la primera Linear
- **Linear** → al final, para combinar y decidir

---

## 12. MLX vs TensorFlow

| | MLX | TensorFlow |
|---|---|---|
| Diseñado para | Apple Silicon (M-series) | NVIDIA GPU (CUDA) |
| Memoria | Unificada CPU+GPU sin copias | Separada, requiere transferencias |
| Ejecución | Lazy (optimiza el grafo antes) | Eager por defecto |
| API | Similar a PyTorch | Keras más alto nivel |
| Documentación | Menos recursos online | Muchísimos tutoriales |
| Velocidad en M5 | Nativa, aprovecha Neural Engine | Parcial via tensorflow-metal |

En este proyecto usamos ambos para comparar rendimiento en el mismo hardware.

---

## 13. Contexto histórico

La arquitectura de este proyecto es una versión simplificada de **LeNet-5**, la CNN original de Yann LeCun diseñada en **1998** precisamente para MNIST. Es el punto de partida de todo el deep learning moderno para imágenes.

Con 32 y 64 canales obtendrás >99% de accuracy en MNIST. Con solo 8 y 16 canales llegarías a >98% — el problema es suficientemente simple que no necesitas mucha capacidad.

La misma idea de `fc2: Linear(128, N_clases)` aplica en todos los modelos:
- Este proyecto: `Linear(128, 10)` — 10 dígitos
- GPT: `Linear(4096, 50257)` — 50,257 tokens del vocabulario
- ResNet ImageNet: `Linear(2048, 1000)` — 1,000 categorías de objetos

Es exactamente el mismo concepto, escalado.
