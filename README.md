# 🧠 VisionAI — Image Recognition

**Upload it. Classify it. See what your AI actually sees.**

VisionAI is a deep-learning-powered image recognition application built with **Streamlit and MobileNetV2 (TensorFlow/Keras)** that identifies the objects in any photo in real time. It returns the top-5 predicted classes out of ImageNet's 1,000 categories, each with a confidence score, animated bar, and plain explanation of what the label means.

### 🚀 Live Demo

**Try VisionAI:** https://vision-ai-image-recognition.streamlit.app

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-MobileNetV2-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Overview

Understanding *what* a neural network sees in an image usually means writing code, loading a model, and wiring up preprocessing by hand.

**VisionAI** removes all of that — just upload a photo (or use the built-in sample) and a pretrained MobileNetV2 model, trained on over a million ImageNet images, classifies it live in your browser. No training, no setup, no GPU required.

* ✅ Top-5 predicted object classes
* 🎯 Confidence score for every prediction
* 💡 Plain-English notes on ambiguous or generic-category labels
* 📊 Animated confidence bars plus a data table
* ⚡ Fast repeat predictions via response caching
* 🖥️ Two-screen experience: landing page + working classifier

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🖼️ **Image Upload** | Upload a JPG or PNG photo, or try the built-in sample image. |
| 🎯 **Confidence Score** | Generates a confidence percentage for each of the top-5 predictions. |
| 📊 **Visual Results** | Animated confidence bars plus an expandable data table of all results. |
| ⚠️ **Low-Confidence Notes** | Explains why photos of people/faces often score low (ImageNet has no "face" class). |
| ℹ️ **Label Clarification** | Flags generic body-style vehicle labels (e.g. "jeep") that are often mistaken for brand names. |
| 🔁 **Zero Training Needed** | Uses transfer learning — a model someone else already trained — instead of building one from scratch. |
| ⚡ **Cached Predictions** | Classifying the same image twice returns instantly via `st.cache_data`. |
| 🎨 **Modern UI** | Custom dark, gradient-accented interface with a branded navbar and hero landing page. |

---

## 🧠 How VisionAI Works

```text
                    ┌─────────────────────┐
                    │   Image Input        │
                    │ Upload / Sample Photo │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Resize to 224×224    │
                    │ & Preprocess Pixels  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ MobileNetV2          │
                    │ (Pretrained, ImageNet)│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Decode Predictions    │
                    │ Top-5 Labels + Scores │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           Top Guess     Confidence Bars   Data Table
```

Each image is resized to 224×224, preprocessed to match MobileNetV2's expected input, and run through the model's compiled graph. The decoded output is a structured list of predictions:

```json
[
  {"label": "mandrill", "confidence": 0.94},
  {"label": "baboon", "confidence": 0.03}
]
```

---

## 🛠️ Tech Stack

### Frontend & Application
* **Python 3.11**
* **Streamlit** — Web application framework

### AI & Deep Learning
* **TensorFlow / Keras** — Model runtime
* **MobileNetV2** — Pretrained image classification model (ImageNet-1000)

### Data Processing
* **NumPy** — Image array preprocessing
* **Pandas** — Results table and data export
* **Pillow (PIL)** — Image loading and resizing

### Deployment
* **Streamlit Cloud** — Application hosting

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Paramjeet-Lamba/VisionAi-Image-Recognition-.git
cd VisionAi-Image-Recognition-
```

### 2. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

(This installs Streamlit, TensorFlow, Pillow, NumPy, and Pandas.)

### 3. Run VisionAI

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

You'll land on the **home page** first. Click **🚀 Launch Classifier** to reach the working tool, and **← Home** to go back.

---

## 📁 Project Structure

```text
VisionAi-Image-Recognition/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
│
└── sample_image.jpg
```

---

## 🧪 Usage

### Step 1 — Launch the Classifier

Open VisionAI and click **🚀 Launch Classifier** from the home page.

### Step 2 — Choose an Image

* Drag-and-drop or browse for a JPG/PNG photo, **or**
* Click **"🐒 Or try the sample image instead"**.

### Step 3 — Classify

Click **✨ Classify this image** to run it through the model.

### Step 4 — Review Results

For every image, VisionAI provides:

* Top predicted label with confidence %
* Prediction time in milliseconds
* Top-5 confidence bars
* Expandable data table of all results

---

## ⚠️ Important Disclaimer

VisionAI uses MobileNetV2, a model pretrained on the **1,000 object categories in ImageNet** — mostly everyday objects, animals, and vehicles. It has **no dedicated "person" or "face" category**, so a low-confidence or unexpected result on a portrait photo is expected model behavior, not a bug. A high confidence score means the pattern strongly matched something the model learned during training — it is **not** a guarantee of correctness.

---

## 🔮 Future Improvements

Potential improvements for future versions include:

* 🌐 Support for custom, fine-tuned models
* 🔎 Face detection as a separate, dedicated mode
* 📸 Multi-image batch classification
* 📊 Historical prediction analytics
* 🧩 Browser extension
* 📱 Mobile-friendly layout refinements
* 🔗 Grad-CAM visual explanations (show *where* the model looked)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you have an idea that can improve VisionAI, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">

### 🧠 VisionAI

**Instant Image Recognition, Powered by Deep Learning**

*See what your AI actually sees.*

Built with ❤️ using **Python, Streamlit & TensorFlow**

</p>
