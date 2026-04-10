# 🧠 MNIST ANN with Batch Normalization

This project demonstrates how **Batch Normalization** improves the performance of an Artificial Neural Network (ANN) on the **MNIST handwritten digits dataset**.

---

## 📌 Project Overview

In this project, we build a simple ANN model using TensorFlow/Keras and enhance it by adding **Batch Normalization layers**.

Batch Normalization helps:
- Stabilize learning  
- Speed up training  
- Improve model accuracy  

---

## 🧾 What is Batch Normalization?

Batch Normalization is a technique used to **normalize the inputs of each layer** in a neural network.

### 🔑 Key Idea:
It keeps the mean close to 0 and variance close to 1 for each mini-batch.

### 📐 Formula:

- Mean:  
  μ = (1/m) * Σx  

- Variance:  
  σ² = (1/m) * Σ(x - μ)²  

- Normalization:  
  x̂ = (x - μ) / √(σ² + ε)  

- Scale and Shift:  
  y = γx̂ + β  

Where:
- γ (gamma) = scale (learnable)  
- β (beta) = shift (learnable)  
- ε = small constant  

---

## 🏗️ Model Architecture

Input Layer (784)  
↓  
Dense Layer (32, ReLU)  
↓  
Batch Normalization  
↓  
Dense Layer (32, ReLU)  
↓  
Batch Normalization  
↓  
Output Layer (10, Softmax)  

---

## ⚙️ Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

model2 = Sequential([
    Dense(32, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
---

## 🚀 Training
```python
model2.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)
```
---

## 📊 Evaluation
```python
test_loss, test_acc = model2.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
```
---

## 📌 Conclusion
Batch Normalization significantly improves model performance by stabilizing learning and accelerating convergence.

It is an essential technique for building efficient deep learning models.

