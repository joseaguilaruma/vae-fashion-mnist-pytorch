# 🧠 Variational Autoencoder (VAE) para Fashion-MNIST con PyTorch

Implementación desde cero de un **Variational Autoencoder (VAE)** utilizando PyTorch para la generación de datos y análisis de espacios latentes sobre el dataset Fashion-MNIST.

Este proyecto forma parte de la asignatura *Programación para la Inteligencia Artificial* del Grado en Ingeniería Informática de la Universidad de Málaga (UMA).

## 🎯 Objetivo del Proyecto
El objetivo principal es construir y entrenar una red autocodificadora variacional que, a diferencia de un Autoencoder tradicional, asocia distribuciones gaussianas al espacio latente utilizando el "truco de reparametrización". Se implementa una función de pérdida combinada:
*   **Pérdida de reconstrucción (MSE)**
*   **Divergencia de Kullback-Leibler (KL)** para regularizar el espacio latente.

## 🛠️ Tecnologías y Librerías Utilizadas
*   **Lenguaje:** Python 3
*   **Deep Learning:** PyTorch, Torchvision
*   **Análisis de Datos:** NumPy, Scikit-Learn (t-SNE)
*   **Visualización:** Matplotlib, Tqdm

## ⚙️ Arquitectura del Modelo
El VAE está diseñado con la siguiente arquitectura:
*   **Encoder:** Capas lineales densas (784 -> 512 -> 128 -> 64) que se dividen en dos cabezas independientes para predecir la media ($\mu$) y el logaritmo de la varianza ($\log\sigma^2$).
*   **Espacio Latente:** Configurado en **8 dimensiones**.
*   **Decoder:** Capas lineales densas (8 -> 64 -> 128 -> 512 -> 784) con una función de activación Sigmoid en la salida para la reconstrucción de las imágenes.

## 🚀 Cómo ejecutarlo
El proyecto consta de un único cuaderno Jupyter que contiene tanto las explicaciones teóricas como el código de entrenamiento.

Puedes ejecutarlo fácilmente en Google Colab o en local:
1. Clona el repositorio: `git clone https://github.com/joseaguilaruma/vae-fashion-mnist-pytorch.git`
2. Instala las dependencias necesarias: `pip install torch torchvision matplotlib scikit-learn numpy tqdm`
3. Abre y ejecuta todas las celdas de `vae_fashion_mnist.ipynb`. El dataset Fashion-MNIST se descargará automáticamente.

## 📈 Resultados del Entrenamiento
El modelo fue entrenado durante 200 épocas utilizando GPU (CUDA). A lo largo del cuaderno se extraen y visualizan:
1.  **Evolución del Error:** Gráficas de la función de pérdida comparando el conjunto de entrenamiento vs validación.
2.  **Visualización del Espacio Latente:** Representaciones bidimensionales (Dim 0 vs 1, Dim 3 vs 5, Dim 7 vs 4) de las componentes del vector $\mu$ del conjunto de test, mostrando cómo el modelo agrupa las distintas prendas de ropa en el espacio latente de forma no supervisada.

---
*Desarrollado por [Jose Francisco Aguilar Granados]((https://www.linkedin.com/in/jose-aguilar-b3113a406/))*
