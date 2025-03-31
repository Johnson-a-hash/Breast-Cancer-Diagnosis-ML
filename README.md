
# Wisconsin Breast Cancer ML Project

This repository contains a machine learning project aimed at diagnosing breast cancer using the **Wisconsin Breast Cancer Dataset (WBCD)**. This project compares the performance of nine traditional machine learning algorithms to determine the most effective model for classifying breast tumors as benign or malignant.

## 📁 Dataset

The data used for this project is the **Wisconsin Breast Cancer Dataset (WBCD)** obtained from the UCI Machine Learning Repository. It includes **699 samples** with **10 attributes** (9 predictors and 1 target variable).

### Attributes:
1. **Clump Thickness (CT)** (1 - 10)
2. **Uniformity of Cell Size (UCSi)** (1 - 10)
3. **Uniformity of Cell Shape (UCSh)** (1 - 10)
4. **Marginal Adhesion (MA)** (1 - 10)
5. **Single Epithelial Cell Size (SECS)** (1 - 10)
6. **Bare Nuclei (BN)** (1 - 10)
7. **Bland Chromatin (BC)** (1 - 10)
8. **Normal Nucleoli (NN)** (1 - 10)
9. **Mitoses** (1 - 10)
10. **Class (Target Variable):** 0 = Benign, 1 = Malignant

## 🎯 Objective
The primary goal of this project is to:
- Identify the most accurate machine learning model for breast cancer diagnosis.
- Compare models based on various performance metrics.

## 🔍 Methods Used
Nine different machine learning algorithms were evaluated:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**
- **Artificial Neural Network (ANN)**
- **Gradient Boosting**
- **Discriminant Analysis (LDA & QDA)**

## 📊 Performance Metrics
The models were evaluated using the following metrics:
- **Accuracy**
- **Sensitivity (Recall)**
- **Specificity**
- **F1 Score**
- **Balanced Accuracy**

## 📈 Results
The best performing models were:
- **K-Nearest Neighbor (KNN)** and **Artificial Neural Network (ANN)**, both achieving:
  - **Balanced Accuracy:** 97.73%
  - **F1 Score:** 96.30%

## 📌 Conclusion
The KNN and ANN models were found to be the most effective for predicting breast cancer using this dataset. Future research could apply these models to other datasets and explore the performance of additional machine learning algorithms.

## 🚀 Installation & Usage
1. Clone the repository:
```bash
    git clone https://github.com/your-username/Wisconsin-Breast-Cancer-ML.git
```
2. Install dependencies:
```bash
    pip install -r requirements.txt
```
3. Run the analysis:
```bash
    python main.py
```

## 📚 References
- UCI Machine Learning Repository: Wisconsin Breast Cancer Dataset
- Research Paper: Diagnosing Wisconsin Breast Cancer: A Machine Learning Approach by Alberta Araba Johnson

## 📄 License
This project is licensed under the MIT License.
