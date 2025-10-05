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
