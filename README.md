# Predicting Maintenance Steps Using Machine Learning
*AER850: Introduction to Machine Learning – Project 1*  
**Author:** Babak Tafreshi  
**Institution:** Toronto Metropolitan University  

---

## Project Overview
This project applies **supervised machine learning** to predict maintenance steps in the disassembly of an **aircraft inverter** using 3D spatial coordinates (**X, Y, Z**).  
The goal is to support **Augmented Reality (AR)-assisted aerospace maintenance**, enabling automated step detection and guidance during complex technical procedures.

The dataset consists of **13 unique disassembly steps**, each defined by specific spatial coordinates. Four machine learning algorithms were developed and compared:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **Gradient Boosting (GB)**

Among these, the **Gradient Boosting Classifier** achieved the highest performance with **99.53% accuracy**, **0.996 weighted precision**, and **0.995 F1-score**.

---

## Project Workflow

| Step | Description | Tools Used |
|------|--------------|-------------|
| **Step 1** | Data Processing – Load and clean CSV data | Pandas |
| **Step 2** | Data Visualization – Explore distributions and clusters | Matplotlib, NumPy |
| **Step 3** | Correlation Analysis – Compute Pearson correlations | Seaborn, Pandas |
| **Step 4** | Model Development – Train and tune classifiers | Scikit-learn |
| **Step 5** | Model Evaluation – Compare metrics and confusion matrices | Scikit-learn (Metrics) |
| **Step 6** | Stacked Model Analysis – Combine SVM + RF using meta-learning | Scikit-learn (StackingClassifier) |
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

The **Gradient Boosting model** was saved and deployed for inference, accurately predicting maintenance steps for new coordinate inputs:
[9.375, 3.0625, 1.51]
[6.995, 5.125, 0.3875]
[0, 3.0625, 1.93]
[9.4, 3, 1.8]
[9.4, 3, 1.3]

yaml
Copy code

---

## Key Insights
- The **X-coordinate** demonstrated the strongest correlation with the maintenance step (r ≈ 0.82).  
- 3D scatter plots confirmed distinct spatial clusters for each maintenance step.  
- Stacking models provided no significant improvement since Gradient Boosting achieved near-perfect classification performance.  

---

## Technologies Used
- **Programming Language:** Python 3.10+  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, joblib  

---

## Usage Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/AER850-ML-Project.git
cd AER850-ML-Project
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the Project
bash
Copy code
python src/project1_main.py
4. Load and Predict Using the Trained Model
python
Copy code
from joblib import load
import numpy as np

model = load("final_gradient_boosting_model.joblib")

coords = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

print(model.predict(coords))
Repository Structure
css
Copy code
AER850-ML-Project/
│
├── data/
│   └── Project1_Data.csv
│
├── src/
│   └── project1_main.py
│
├── outputs/
│   └── figures/
│
├── README.md
└── requirements.txt
License
This project is provided for academic and educational purposes under the MIT License.

Acknowledgment
Developed as part of AER850: Introduction to Machine Learning at Toronto Metropolitan University (TMU), focusing on the application of Machine Learning in Aerospace Maintenance and AR-based Predictive Systems.
