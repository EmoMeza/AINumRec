# AINumRec — Reconocimiento de Dígitos con CNN

Sistema de reconocimiento de dígitos escritos a mano usando redes neuronales convolucionales (CNN), entrenado con el dataset MNIST y con soporte para predicciones en tiempo real mediante cámara.

> Proyecto de aprendizaje enfocado en visión por computadora y Machine Learning aplicado.

---

## Objetivo

Construir un modelo capaz de identificar dígitos del **0 al 9** dibujados a mano, integrándolo con la cámara para hacer predicciones en tiempo real sobre papel.

---

## Tecnologías

- **Python** — lenguaje principal
- **MLX** — entrenamiento optimizado para Apple Silicon (M-series)
- **TensorFlow / Keras** — entrenamiento alternativo multiplataforma
- **OpenCV** — captura y procesamiento de imágenes desde la cámara
- **MNIST Dataset** — dataset estándar de 70,000 imágenes de dígitos escritos a mano

---

## Entrenamiento

El proyecto incluye dos implementaciones del mismo modelo CNN para comparar rendimiento:

| Archivo | Framework | Ideal para |
|---|---|---|
| `train.py` | MLX | Apple Silicon (M1/M2/M3/M4/M5) |
| `train_tf.py` | TensorFlow / Keras | Multiplataforma / NVIDIA GPU |

---

## Estructura del proyecto

```
AINumRec/
│
├── README.md             # Este archivo
├── train.py              # Entrenamiento con MLX (Apple Silicon)
├── train_tf.py           # Entrenamiento con TensorFlow/Keras
├── camera_predict.py     # Predicción en tiempo real con cámara
├── visualize images.py   # Visualización de muestras del dataset
├── utils.py              # Preprocesamiento y funciones auxiliares
├── dataset/              # Archivos MNIST (.idx ubyte) — no incluidos en git
└── saved_model/          # Modelo entrenado guardado
```

---

## Objetivos de aprendizaje

- Entender cómo funcionan las CNN aplicadas a imágenes
- Preprocesar datos y aplicar técnicas de data augmentation
- Evaluar y ajustar modelos de Machine Learning
- Comparar frameworks de entrenamiento (MLX vs TensorFlow)
- Integrar un modelo entrenado con entrada de cámara en tiempo real

---

## Roadmap

- [ ] Preparar y preprocesar el dataset MNIST
- [ ] Entrenar un modelo CNN con MLX
- [ ] Entrenar el mismo modelo con TensorFlow y comparar resultados
- [ ] Evaluar la precisión y ajustar hiperparámetros
- [ ] Implementar predicción en tiempo real desde la cámara

---

## Referencias

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [MLX Docs](https://ml-explore.github.io/mlx/build/html/index.html)
- [TensorFlow — Guía de CNN](https://www.tensorflow.org/tutorials/images/cnn)
- [OpenCV Python Docs](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
