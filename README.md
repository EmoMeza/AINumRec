# AINumRec — Reconocimiento de Dígitos con CNN

Sistema de reconocimiento de dígitos escritos a mano usando redes neuronales convolucionales (CNN), entrenado con el dataset MNIST y con soporte para predicciones en tiempo real mediante cámara.

> Proyecto de aprendizaje enfocado en visión por computadora y Machine Learning aplicado.

---

## Objetivo

Construir un modelo capaz de identificar dígitos del **0 al 9** dibujados a mano, integrándolo con la cámara para hacer predicciones en tiempo real sobre papel.

---

## Tecnologías

- **Python** — lenguaje principal
- **TensorFlow / Keras** — entrenamiento del modelo CNN
- **OpenCV** — captura y procesamiento de imágenes desde la cámara
- **MNIST Dataset** — dataset estándar de 70,000 imágenes de dígitos escritos a mano

---

## Estructura del proyecto

```
AINumRec/
│
├── README.md             # Este archivo
├── train_model.py        # Entrenamiento del modelo CNN
├── camera_predict.py     # Predicción en tiempo real con cámara
├── utils.py              # Preprocesamiento y funciones auxiliares
└── saved_model/          # Modelo entrenado guardado
```

---

## Objetivos de aprendizaje

- Entender cómo funcionan las CNN aplicadas a imágenes
- Preprocesar datos y aplicar técnicas de data augmentation
- Evaluar y ajustar modelos de Machine Learning
- Integrar un modelo entrenado con entrada de cámara en tiempo real

---

## Roadmap

- [ ] Preparar y preprocesar el dataset MNIST
- [ ] Entrenar un modelo CNN básico para clasificación de dígitos
- [ ] Evaluar la precisión y ajustar hiperparámetros
- [ ] Implementar predicción en tiempo real desde la cámara

---

## Referencias

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow — Guía de CNN](https://www.tensorflow.org/tutorials/images/cnn)
- [OpenCV Python Docs](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
