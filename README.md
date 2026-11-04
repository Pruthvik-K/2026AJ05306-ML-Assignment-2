# Frog Species Classification

A machine learning project that classifies frog species using MFCC features extracted from frog calls. The project implements five classification algorithms, compares their performance, and provides a Streamlit application to evaluate the trained models.

## Project Links

- **GitHub:** https://github.com/Pruthvik-K/2026AJ05306-ML-Assignment-2
- **Live Application:** https://2026aj05306-ml-assignment-2-nva82wixowkc33lqujqgkz.streamlit.app/
- **Dataset:** [UCI – Anuran Calls (MFCCs)](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs)

## Dataset

The project uses the Anuran Calls (MFCCs) dataset from the UCI Machine Learning Repository.

- 7,195 observations
- 22 MFCC input features
- 10 frog species
- No missing values

`Species` is used as the target variable. `Family`, `Genus` and `RecordID` are excluded from model training.

A group-aware train-test split is used based on `RecordID` to prevent observations from the same recording appearing in both training and testing sets.

## Models Implemented

Five classification algorithms are trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Feature scaling is applied where required. Hyperparameter tuning is performed using GridSearchCV with group-aware cross-validation and Macro F1 as the scoring metric.

## Evaluation

The models are evaluated using:

- Accuracy
- AUC (Macro One-vs-Rest)
- Precision
- Recall
- Weighted F1 Score
- Macro F1 Score
- Matthews Correlation Coefficient (MCC)

The tuned models achieved the following results on the held-out test dataset:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 89.76% | 63.77% |
| Decision Tree | 82.96% | 53.23% |
| KNN | 90.78% | 67.32% |
| Naive Bayes | 81.69% | 56.42% |
| SVM | 90.97% | 58.88% |

KNN achieved the highest Macro F1, while SVM achieved the highest Accuracy.

## Streamlit Application

The application allows users to:

- Select one of the five trained models.
- Upload a labelled CSV dataset.
- Evaluate model performance.
- View evaluation metrics, confusion matrix and classification report.
- Download predictions as a CSV file.

The application uses saved models and does not perform training when a dataset is uploaded.

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

Start the application:

```bash
streamlit run app.py
```

To explore the complete training and evaluation process, open `ASSIGNMENT_2.ipynb` using Jupyter Notebook.

## Technologies Used

Python, NumPy, Pandas, Scikit-learn, Joblib, Streamlit, Jupyter Notebook and GitHub.