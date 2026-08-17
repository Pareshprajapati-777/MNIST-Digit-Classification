# 🔢 MNIST Digit Classification

> 🧠 **Handwritten Digit Recognition using TensorFlow CNN + Streamlit**

A clean and interactive **Machine Learning / Deep Learning web application** that recognizes handwritten digits from uploaded images. The project uses the **MNIST dataset**, a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras, and a Streamlit interface for real-time prediction.

---

## ✨ Features

- 🔢 Recognizes handwritten digits from **0–9**
- 🧠 CNN-based image classification using **TensorFlow/Keras**
- 📂 Upload images in **PNG, JPG, or JPEG** format
- 🎯 Displays the predicted digit
- 📊 Shows prediction confidence
- 📈 Displays model test accuracy
- ⚡ Streamlit caching for efficient dataset/model loading
- 🎨 Clean and responsive Streamlit UI
- 🖼️ Converts uploaded images to **28 × 28 grayscale** format

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming language |
| 🧠 TensorFlow / Keras | CNN model development |
| 🎨 Streamlit | Interactive web application |
| 🔢 NumPy | Numerical and image-array processing |
| 📚 MNIST | Handwritten digit dataset |

---

## 🧠 Model Architecture

The project uses a lightweight CNN architecture:

```text
Input: 28 × 28 × 1
        ↓
Conv2D (32 filters, 3×3) + ReLU
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3) + ReLU
        ↓
MaxPooling2D (2×2)
        ↓
Flatten
        ↓
Dense (64) + ReLU
        ↓
Dense (32) + ReLU
        ↓
Dense (10) + Softmax
        ↓
Prediction: 0–9
```

The model is trained with **Adam optimizer** and **Sparse Categorical Crossentropy** for 5 epochs with a batch size of 32. The implementation also uses a validation split of 20%. 

---

## 📊 Dataset

The application uses the built-in **MNIST handwritten digit dataset**:

- 🖼️ Total images: **70,000**
- 🎯 Classes: **10 (0–9)**
- 📐 Image size: **28 × 28 pixels**
- ⚫ Image type: **Grayscale**
- 🏋️ Training images: **60,000**
- 🧪 Test images: **10,000**

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Pareshprajapati-777/MNIST-Digit-Classification.git
cd MNIST-Digit-Classification
```

### 2️⃣ Install dependencies

```bash
pip install tensorflow streamlit numpy
```

### 3️⃣ Run the application

```bash
streamlit run Mnist_Number.py
```

The Streamlit application will open in the browser. 🌐

---

## 🖥️ How It Works

1. 📥 The MNIST dataset is loaded automatically.
2. 🧹 Images are normalized to the **0–1** range.
3. 🧠 A CNN model is trained on the handwritten digit dataset.
4. 📤 Upload a digit image through the Streamlit interface.
5. 🖼️ The image is resized to **28 × 28** and converted to grayscale.
6. 🔄 The image is normalized and inverted for the model input.
7. 🤖 The CNN predicts the digit.
8. 📊 The application displays the predicted digit and confidence score.

---

## 📁 Project Structure

```text
MNIST-Digit-Classification/
│
├── 🐍 Mnist_Number.py    # Main Streamlit application
├── 📖 README.md          # Project documentation
└── 📄 .gitignore         # Git ignored files (if added)
```

---

## 🎯 Example Output

After uploading a handwritten digit image, the application displays:

```text
🔢 Predicted Digit : 7
📊 Confidence      : 98.XX%
```

> 💡 Prediction confidence depends on the uploaded image and how closely it resembles the MNIST-style handwritten digits used during training.

---

## 🔮 Future Improvements

- 🚀 Save and load a trained model instead of training on every fresh session
- 🎨 Add a drawing canvas for handwritten digit input
- 📱 Improve mobile responsiveness
- 📊 Add a confusion matrix and classification report
- ⚡ Optimize inference and application startup time
- 🧠 Experiment with data augmentation and improved CNN architectures

---

## 👨‍💻 Author

**Paresh Prajapati**

🔗 GitHub: [Pareshprajapati-777](https://github.com/Pareshprajapati-777)

---

## ⭐ Support

If this project is useful, consider giving the repository a **⭐ Star** on GitHub. It helps support the project and future improvements.

---

## 📜 License

This project is intended for **educational and portfolio purposes**. Add an explicit open-source license to the repository if redistribution terms are required.

---

### ❤️ Built with Python, TensorFlow, Keras, NumPy & Streamlit