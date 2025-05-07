
# 🧠 Deep Learning Journey: TensorFlow & PyTorch

Welcome to my deep learning roadmap! This repository contains my hands-on practice and project implementations using **TensorFlow** and **PyTorch**.

---

## 📘 Table of Contents

- [📘 Table of Contents](#-table-of-contents)
- [📚 Foundational Concepts](#-foundational-concepts)
  - [TensorFlow](#tensorflow)
  - [PyTorch](#pytorch)
- [🚀 Project Roadmap](#-project-roadmap)
  - [TensorFlow Projects](#tensorflow-projects)
  - [PyTorch Projects](#pytorch-projects)
- [📂 Repository Structure](#-repository-structure)
- [📚 Resources](#-resources)

---

## 📚 Foundational Concepts

### TensorFlow

> TensorFlow 2.x uses eager execution by default, making it easier to experiment and debug.

#### ✅ Constants
```python
import tensorflow as tf

const_tensor = tf.constant([1.0, 2.0, 3.0])
print(const_tensor)
````

#### 📝 Variables

```python
var_tensor = tf.Variable([1.0, 2.0, 3.0])
var_tensor.assign([4.0, 5.0, 6.0])
print(var_tensor)
```

#### 🛑 Placeholders (Legacy)

```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
doubled = placeholder * 2

with tf.Session() as sess:
    print(sess.run(doubled, feed_dict={placeholder: [1.0, 2.0, 3.0]}))
```

---

### PyTorch

> PyTorch uses a dynamic computation graph — making it flexible and intuitive.

#### ✅ Tensors

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
print(tensor)
```

#### 🔁 Tensors with Gradients

```python
tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(tensor)
```

#### 🔧 Dynamic Inputs

```python
def double_input(x):
    return x * 2

input_tensor = torch.tensor([1.0, 2.0, 3.0])
print(double_input(input_tensor))
```

---

## 🚀 Project Roadmap

### TensorFlow Projects

| #  | Project Name                               | Description                          |
| -- | ------------------------------------------ | ------------------------------------ |
| 1  | Linear Regression                          | Fit a line to data using MSE loss    |
| 2  | Logistic Regression                        | Binary classification model          |
| 3  | Multilayer Perceptron (MLP)                | Basic feedforward neural net         |
| 4  | Training with GradientTape                 | Custom training loop                 |
| 5  | Custom Loss & Metrics                      | Build and use your own functions     |
| 6  | Batch Normalization                        | Normalize activations between layers |
| 7  | Learning Rate Scheduling                   | Use schedulers to vary learning rate |
| 8  | Early Stopping & Checkpointing             | Prevent overfitting and save models  |
| 9  | Binary Classification on Tabular Data      | Using structured datasets            |
| 10 | Multi-Class Classification (Fashion MNIST) | Image classification                 |

---

### PyTorch Projects

| #  | Project Name                          | Description                               |
| -- | ------------------------------------- | ----------------------------------------- |
| 1  | Linear Regression                     | Fit a line to data using MSE loss         |
| 2  | Logistic Regression                   | Binary classification model               |
| 3  | MLP from Scratch                      | Build a neural network manually           |
| 4  | Custom Training Loop                  | Full control over forward/backward passes |
| 5  | Custom Loss Functions                 | Create your own loss metrics              |
| 6  | Batch Normalization                   | Normalize layer outputs                   |
| 7  | Learning Rate Scheduling              | Dynamically change learning rate          |
| 8  | Early Stopping & Checkpointing        | Monitor and save best models              |
| 9  | Binary Classification on Tabular Data | Use structured tabular inputs             |
| 10 | Multi-Class Classification (CIFAR-10) | Image classification                      |

---

## 📂 Repository Structure

```
.
├── TensorFlow/
│   ├── 01_linear_regression_tf.ipynb
│   ├── 02_logistic_regression_tf.ipynb
│   └── ...
├── PyTorch/
│   ├── 01_linear_regression_pt.ipynb
│   ├── 02_logistic_regression_pt.ipynb
│   └── ...
├── README.md
```

---

## 📚 Resources

* [TensorFlow Official Docs](https://www.tensorflow.org/)
* [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
* [Deep Learning with PyTorch: 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [TensorFlow Beginner Guide](https://www.tensorflow.org/tutorials/quickstart/beginner)

