#  Predicting Maintenance Steps Using Machine Learning  
*AER850: Introduction to Machine Learning – Project 1*  
Author: **Babak Tafreshi**  
Toronto Metropolitan University  

---

##  Project Overview  
This project applies **supervised machine learning** to predict maintenance steps in the disassembly of an **aircraft inverter** using 3D spatial coordinates (**X, Y, Z**).  
The model aims to support **AR-assisted aerospace maintenance**, allowing automated step detection and guidance during technical procedures.  

The dataset contains **13 unique disassembly steps**, each corresponding to a distinct set of spatial coordinates.  
Four ML algorithms were trained and compared:  
- **K-Nearest Neighbors (KNN)**  
- **Support Vector Machine (SVM)**  
- **Random Forest (RF)**  
- **Gradient Boosting (GB)**  

The **Gradient Boosting Classifier** achieved the best performance with **99.53% accuracy**, **0.996 weighted precision**, and **0.995 F1-score**.

---

##  Project Workflow  

| Step | Description | Tools Used |
|------|--------------|-------------|
| **Step 1** | Data Processing – Load and clean CSV data | Pandas |
| **Step 2** | Data Visualization – Explore distributions and clusters | Matplotlib, NumPy |
| **Step 3** | Correlation Analysis – Compute Pearson correlations | Seaborn, Pandas |
| **Step 4** | Model Development – Train and tune classifiers | Scikit-learn |
| **Step 5** | Model Evaluation – Compare metrics and confusion matrices | Scikit-learn (Metrics) |
| **Step 6** | Stacked Model Analysis – Combine SVM + RF via meta-learning | Scikit-learn (StackingClassifier) |
| **Step 7** | Model Deployment – Save and reload best model using Joblib | Joblib |

---

## Results Summary  

| Model | Accuracy | Precision | F1-score |
|--------|:---------:|:-----------:|:----------:|
| KNN | 0.9814 | 0.9834 | 0.9813 |
| SVM (RBF) | 0.9814 | 0.9827 | 0.9813 |
| Random Forest | 0.9860 | 0.9874 | 0.9860 |
| **Gradient Boosting** | **0.9953** | **0.9960** | **0.9953** |
| Stacked (SVM + RF) | 0.9814 | 0.9827 | 0.9813 |

The **Gradient Boosting model** was packaged for reuse and correctly predicted maintenance steps for new coordinate samples:

[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]


---

## 🧠 Key Insights  
- The **X-coordinate** showed the strongest correlation with the maintenance step (r ≈ 0.82).  
- 3D visualization revealed distinct clusters for each step, confirming data separability.  
- Stacking models showed minimal gain since Gradient Boosting already achieved near-perfect accuracy.  

---

## 🧩 Technologies Used  
- **Python 3.10+**  
- **Libraries:**  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

---

## 💻 Usage Instructions  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/AER850-ML-Project.git
cd AER850-ML-Project


INSTALL DEPENDENCIES:

pip install -r requirements.txt



from joblib import load
import numpy as np

model = load("final_gradient_boosting_model.joblib")
coords = np.array([
    [9.375,3.0625,1.51],
    [6.995,5.125,0.3875],
    [0,3.0625,1.93],
    [9.4,3,1.8],
    [9.4,3,1.3]
])
print(model.predict(coords))


python project1_main.py



Repository structure:
├── data/
│   └── Project1_Data.csv
├── src/
│   ├── project1_main.py
├── outputs/
│   └── figures/
├── README.md
└── requirements.txt


 License

This project is provided for academic and educational purposes under the MIT License


Acknowledgment

Developed as part of AER850: Introduction to Machine Learning at Toronto Metropolitan University (TMU),
focusing on applications of Machine Learning in Aerospace Maintenance and AR-based Predictive Systems.
.

