# Frog Species Classification

A machine learning project that classifies frog species using MFCC features extracted from frog calls. Five classification algorithms are implemented, compared and deployed through a Streamlit application.

## Project Links

- **GitHub:** https://github.com/Pruthvik-K/2026AJ05306-ML-Assignment-2
- **Live Application:** https://2026aj05306-ml-assignment-2-nva82wixowkc33lqujqgkz.streamlit.app/
- **Dataset:** [UCI – Anuran Calls (MFCCs)](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs)

## Dataset Description

The project uses the Anuran Calls (MFCCs) dataset from the UCI Machine Learning Repository.

- 7,195 observations
- 22 MFCC input features
- 10 frog species
- No missing values

`Species` is the target variable. `Family`, `Genus` and `RecordID` are excluded from model training.

A group-aware train-test split is performed using `RecordID` to ensure that observations from the same recording do not appear in both training and testing sets.

## Models Implemented

The following five classification algorithms are trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Feature scaling is applied where required. Hyperparameter tuning is performed using GridSearchCV with group-aware cross-validation, using Macro F1 as the scoring metric.

## Model Evaluation

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). Macro F1 is also included to better understand performance across the imbalanced species classes.

### Comparison of Tuned Models

All values below are calculated on the held-out test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8976 | 0.9748 | 0.9085 | 0.8976 | 0.9003 | 0.8552 | 0.6377 |
| Decision Tree | 0.8296 | 0.8393 | 0.8834 | 0.8296 | 0.8495 | 0.7619 | 0.5323 |
| KNN | 0.9078 | 0.9334 | 0.9283 | 0.9078 | 0.9129 | 0.8699 | 0.6732 |
| Naive Bayes | 0.8169 | 0.9633 | 0.8797 | 0.8169 | 0.8385 | 0.7490 | 0.5642 |
| SVM | 0.9097 | 0.9832 | 0.9069 | 0.9097 | 0.9050 | 0.8724 | 0.5888 |

*Precision, Recall and F1 are weighted averages. AUC is Macro One-vs-Rest.*

### Observations

| Model | Observation |
|---|---|
| Logistic Regression | Provides consistent performance with 89.76% accuracy and a strong AUC of 0.9748. |
| Decision Tree | Achieves lower performance compared to the other models, with an accuracy of 82.96%. |
| KNN | Achieves the highest Macro F1 (0.6732) and Weighted F1 (0.9129), indicating better balanced classification performance across species. |
| Naive Bayes | Achieves a high AUC but has comparatively lower accuracy and F1 score. |
| SVM | Achieves the highest accuracy (90.97%), AUC (0.9832) and MCC (0.8724) among the tuned models. |

**Overall Winner: KNN**, when Macro F1 is prioritized because of the class imbalance. SVM is the strongest model in terms of Accuracy, AUC and MCC.

## Streamlit Application

The application allows users to:

- Select one of the five trained models.
- Upload a labelled CSV dataset.
- Evaluate model performance.
- View evaluation metrics, confusion matrix and classification report.
- Download predictions as a CSV file.

The application loads previously trained models and does not perform training when a dataset is uploaded.

## Project Structure

```text
├── ASSIGNMENT_2.ipynb       # Model development and evaluation
├── app.py                   # Streamlit application
├── data/
│   └── Frogs_MFCCs.csv      # Original dataset
├── models/                  # Saved trained models
│   ├── baseline/            # Baseline models
│   └── ...                  # Tuned models and metadata
├── test_data.csv            # Held-out test dataset
├── frog_species_predictions.csv
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Pruthvik-K/2026AJ05306-ML-Assignment-2.git
cd 2026AJ05306-ML-Assignment-2

pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

To explore the complete training and evaluation workflow, open `ASSIGNMENT_2.ipynb` using Jupyter Notebook.

## Technologies Used

Python, NumPy, Pandas, Scikit-learn, Joblib, Streamlit, Jupyter Notebook and GitHub.