# Diabetic Retinopathy Detection System (Explainable AI Based)

An end-to-end **Explainable AI (XAI)** based Diabetic Retinopathy (DR) detection and grading system built using **PyTorch, EfficientNet-B3, Grad-CAM++, LIME, Streamlit, and SQLAlchemy**.  
This system allows clinicians to upload retinal fundus images, get DR stage predictions, visualize model explanations, and generate structured clinical reports.

---

## 🚀 Features

- 🖼️ Upload retinal fundus images
- 🧠 DR severity classification (5 classes: 0–4)
- 🔥 Grad-CAM++ heatmap visualization
- 🧩 LIME-based local explanations
- 📊 Confidence scores and probability distribution
- 📄 Automated PDF report generation
- 🗄️ Database storage for patient history
- 📈 Dashboard for analytics and tracking
- 🎨 Streamlit-based interactive UI

---

## 🗄️ Dataset Used
```bash
https://www.kaggle.com/datasets/mariaherrerot/aptos2019/data
```

---

## 🧠 Architecture (AI + XAI Pipeline)

The system follows a **Deep Learning + Explainability pipeline**:

### 1️⃣ Image Preprocessing
- CLAHE for contrast enhancement  
- Resize to 300×300  
- Normalize using ImageNet statistics  

### 2️⃣ Data Augmentation
- Rotation, flipping, translation  
- Color jitter and noise addition  
- Applied only on training set  

### 3️⃣ Model Architecture
- EfficientNet-B3 (pretrained on ImageNet)  
- Attention mechanism for feature focus  
- Custom classification head (BatchNorm + Dropout + FC layers)  

### 4️⃣ Training
- Label Smoothing Loss  
- AdamW optimizer  
- Cosine Annealing scheduler  
- Gradient clipping + Early stopping  

### 5️⃣ Prediction
- Softmax probability output  
- 5-class DR severity prediction  

### 6️⃣ Explainability
- Grad-CAM++ → heatmap visualization  
- LIME → superpixel-based explanation  
- Region-based retinal analysis  

### 7️⃣ Database Storage
- Patient details  
- Prediction results  
- Probability distribution  
- Grad-CAM++ & LIME outputs  

### 8️⃣ Report Generation
- PDF report with:
  - Prediction  
  - Confidence  
  - Risk level  
  - Heatmap visualization  
  - Clinical suggestions  

---

## 🏗️ Project Structure
```
DR_Project/
├── app/
│ └── app.py # Streamlit application
├── data/
│ └── raw/
|   ├── test_images/
│   ├── train_images/
│   ├── val_images/
│   ├── test.csv
|   ├── train.csv
│   └── valid.csv
├── src/
│ ├── model.py # EfficientNet-B3 model
│ ├── train.py # Training pipeline
│ ├── evaluate.py # Evaluation script
│ ├── gradcam.py # Grad-CAM++ implementation
│ ├── lime_explainer.py # LIME explanations
│ └── utils.py
├── database/
│ ├── db.py # Database connection
│ ├── models.py # SQLAlchemy models
│ └── crud.py # DB operations
├── models/
│ └── best_model.pth # Trained model
├── reports/
├── uploads/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/dr-detection-system.git
cd dr-detection-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Create Virtual Environment

```bash
pip install -r requirements.txt
```

### ▶️ Running the Project
#### Step 1: Train the Model
```bash
python src/train.py
```
#### This will:
#### Train EfficientNet-B3 model
#### Save best model as: models/best_model.pth

#### Step 2: Evaluate the Model
```bash
python src/evaluate.py
```

#### This will:
#### Load trained model
#### Evaluate on test dataset
#### Display accuracy, confusion matrix, and metrics

#### Step 3: Run Streamlit App
```bash
streamlit run app/app.py
```

#### Then open:
```bash
http://localhost:8501
```

---

## 🧪 Model Performance

- ✅ Accuracy: **83.67% (Final Deployed Model)**
- 📈 AUC-ROC: **0.95**

### 🔍 Strong performance on:
- No DR  
- Moderate DR  

### ⚠️ Challenges:
- Severe DR  
- Proliferative DR classification  

---

## 🧩 Technologies Used

- Python  
- PyTorch  
- timm (EfficientNet)  
- Streamlit  
- OpenCV  
- NumPy, Pandas  
- Grad-CAM++  
- LIME  
- SQLAlchemy (SQLite DB)  
- ReportLab (PDF generation)  

---

## 📌 Future Improvements

- 🔬 Multi-dataset validation (APTOS, EyePACS, Messidor)  
- 🧠 Lesion segmentation (microaneurysms, exudates)  
- 🏥 Integration with EHR/PACS systems  
- 📊 Real-time clinical deployment  
- ⚡ GPU optimization for faster inference  

---

## 📄 License

This project is developed for academic and research purposes.

---

## 🙌 Acknowledgement

Developed as part of an academic research project on **Explainable AI for Diabetic Retinopathy Detection**.

### 👩‍💻 Created by:
- Simran Arya  
- Anshika Singh  
- Payoshi Gupta  
- Aayushi Vinod  
- Suyogya Tiwari  
- Aviral Srivastava  