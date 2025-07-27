
# MNIST Digit Classification using Fully Connected Neural Network (ANN)

This project implements a fully connected Artificial Neural Network (ANN) from scratch for the task of digit classification on the MNIST dataset. The model is built using NumPy and trained using forward and backward propagation without relying on high-level deep learning libraries.

---

## 🧠 Architecture

- **Input Layer**: 784 neurons (flattened 28x28 grayscale image)
- **Hidden Layers**:
  - Layer 1: 256 neurons
  - Layer 2: 128 neurons
  - Layer 3: 64 neurons
  - Activation: `Tanh` (default), optionally `ReLU` or `Sigmoid`
- **Output Layer**: 10 neurons (for digit classes 0-9), with `Softmax` activation

---

## 🔧 Implementation Details

### ➤ Forward Propagation
Each layer computes:
- **Linear step**: \( Z^{[i]} = W^{[i]}A^{[i-1]} + b^{[i]} \)
- **Activation**: \( A^{[i]} = g(Z^{[i]}) \), where `g` is Tanh, ReLU, or Sigmoid

**Output Layer** uses Softmax:
\[
A^{[L]}_j = rac{\exp(Z^{[L]}_j)}{\sum_k \exp(Z^{[L]}_k)}
\]

### ➤ Dropout Regularization
To mitigate overfitting:
\[
A^{[i]}_{	ext{dropout}} = rac{A^{[i]} \cdot 	ext{Mask}}{1 - 	ext{dropout rate}}
\]

---

## 🔁 Backpropagation & Optimization

### ➤ Gradient Calculation
- Output layer:
  - \( dZ^{[L]} = A^{[L]} - Y \)
  - \( dW^{[L]} = rac{1}{m} dZ^{[L]} A^{[L-1]T} + \lambda W^{[L]} \)
  - \( db^{[L]} = rac{1}{m} \sum dZ^{[L]} \)

- Hidden layers (for layer *i*):
  - \( dZ^{[i]} = (W^{[i+1]T} \cdot dZ^{[i+1]}) \circ g'(Z^{[i]}) \)

### ➤ Weight Updates
- Dual optimization combining **Adam** and **RMSprop**:
  - Adam: estimates momentum and adaptive learning rate
  - RMSprop: gradient normalization
  - Combined update is average of both optimizations

---

## 🧪 Data Handling

- **Dataset**: MNIST (via `torchvision.datasets`)
- **Normalization**: Pixel values scaled to [0, 1]
- **Label Encoding**: One-hot vectors
- **Split**:
  - 80% training
  - 20% validation
  - 100% test data (stratified sampling)

---

## 📉 Training Strategy

- **Early Stopping**: Stops if validation loss doesn't improve for 10 epochs
- **Epochs**: 100 (max), may end earlier due to early stopping
- **Loss Plotting**: Training vs Validation loss over time

---

## 📊 Results

- **Test Accuracy**: **93.46%**
- **Visualization**:
  - Predicted vs actual digits in a grid format
  - Loss curves for training and validation

---

## 🛠️ Technologies Used

- **Language**: Python (no deep learning frameworks used)
- **Libraries**: `numpy`, `matplotlib`, `torchvision` (only for dataset)

---

## 📂 File Structure

```
.
├── CODE.ipynb                # Main implementation in Jupyter Notebook
├── CS419_ASSIGNMENT_REPORT.pdf  # Report with equations and explanation
├── README.md
```

---

## ✅ Features

- Built from scratch using NumPy
- Dropout for regularization
- Dual optimizer: Adam + RMSprop
- Early stopping for generalization
- Tanh/ReLU/Sigmoid activation toggle

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/mnist-ann-from-scratch.git
cd mnist-ann-from-scratch
```

2. Install dependencies:
```bash
pip install numpy matplotlib torchvision
```

3. Open the notebook:
```bash
jupyter notebook CODE.ipynb
```

---

## 👨‍💻 Author

**Santosh Guntuku**  
Roll No: `23B2158`  
Course: CS419 - Machine Learning  
Department of Mechanical Engineering, IIT Bombay

---
