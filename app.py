
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Frog Species Classification",
    page_icon="🐸",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# --------------------------------------------------
# 2. Load metadata and saved models
# --------------------------------------------------

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    metadata = load_metadata()
    feature_columns = metadata["feature_columns"]
    class_labels = metadata["class_labels"]
    target_column = metadata["target_column"]

except Exception as error:
    st.error(f"Could not load model metadata: {error}")
    st.stop()


# --------------------------------------------------
# 3. Application heading
# --------------------------------------------------

st.title("🐸 Frog Species Classification")

st.markdown(
    """
    This application uses machine learning models trained on
    Mel-Frequency Cepstral Coefficients (MFCCs) extracted from
    frog calls to classify frog species.

    Upload a labelled CSV dataset, select a trained classifier,
    and evaluate its performance.
    """
)

st.info(
    "The application loads previously trained models. "
    "It does not perform model training."
)


# --------------------------------------------------
# 4. Model selection
# --------------------------------------------------

st.subheader("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys())
)

model_path = MODEL_DIR / MODEL_FILES[selected_model_name]

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as error:
    st.error(f"Could not load the selected model: {error}")
    st.stop()

st.success(f"{selected_model_name} loaded successfully.")


# --------------------------------------------------
# 5. Upload test dataset
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing MFCC features and Species",
    type=["csv"]
)

if uploaded_file is None:

    st.warning("Please upload a CSV file to begin evaluation.")

    st.markdown(
        """
        **Expected CSV format:**
        - 22 MFCC feature columns
        - One target column named `Species`
        - No missing values
        """
    )

    st.stop()


# --------------------------------------------------
# 6. Validate uploaded dataset
# --------------------------------------------------

try:
    uploaded_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()


st.write("### Uploaded Dataset Preview")
st.dataframe(uploaded_data.head(), use_container_width=True)

missing_columns = [
    column for column in feature_columns + [target_column]
    if column not in uploaded_data.columns
]

if missing_columns:
    st.error(
        "The uploaded dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded dataset contains no observations.")
    st.stop()

# Select only the expected model inputs
X_test = uploaded_data[feature_columns].copy()
y_test = uploaded_data[target_column].copy()

# Ensure feature values are numeric
try:
    X_test = X_test.apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError):
    st.error("All MFCC feature columns must contain numeric values.")
    st.stop()

if X_test.isnull().values.any() or y_test.isnull().any():
    st.error("The uploaded dataset contains missing values.")
    st.stop()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error("The MFCC features contain infinite or invalid values.")
    st.stop()

unknown_labels = set(y_test.unique()) - set(class_labels)

if unknown_labels:
    st.error(
        "The target column contains unknown species labels: "
        + ", ".join(sorted(unknown_labels))
    )
    st.stop()


# --------------------------------------------------
# 7. Display dataset information
# --------------------------------------------------

st.write("### Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Total Observations", len(uploaded_data))
col2.metric("Input Features", len(feature_columns))
col3.metric("Species Classes", y_test.nunique())


# --------------------------------------------------
# 8. Evaluate selected model
# --------------------------------------------------

if st.button("Evaluate Model", type="primary"):

    try:
        # Generate predictions
        y_pred = model.predict(X_test)

        # Generate class probabilities for AUC
        y_prob = model.predict_proba(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # Macro One-vs-Rest multiclass AUC
        try:
            auc = roc_auc_score(
                y_test,
                y_prob,
                labels=class_labels,
                multi_class="ovr",
                average="macro"
            )
            auc_text = f"{auc:.4f}"

        except ValueError:
            auc = None
            auc_text = "Unavailable"

        # ------------------------------------------
        # 9. Display evaluation metrics
        # ------------------------------------------

        st.divider()
        st.subheader("Model Evaluation Results")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric("Accuracy", f"{accuracy:.4f}")
        metric_col2.metric("AUC (Macro OvR)", auc_text)
        metric_col3.metric("Precision (Weighted)", f"{precision:.4f}")

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric("Recall (Weighted)", f"{recall:.4f}")
        metric_col5.metric("F1 Score (Weighted)", f"{f1:.4f}")
        metric_col6.metric("MCC", f"{mcc:.4f}")

        st.metric("Macro F1 Score", f"{macro_f1:.4f}")

        if auc is None:
            st.warning(
                "AUC could not be calculated. "
                "Check whether every species is represented "
                "in the uploaded test dataset."
            )

        # ------------------------------------------
        # 10. Confusion matrix
        # ------------------------------------------

        st.divider()
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=class_labels
        )

        confusion_df = pd.DataFrame(
            cm,
            index=class_labels,
            columns=class_labels
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

        # ------------------------------------------
        # 11. Classification report
        # ------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ------------------------------------------
        # 12. Download predictions
        # ------------------------------------------

        results = uploaded_data.copy()
        results["Actual Species"] = y_test
        results["Predicted Species"] = y_pred

        st.download_button(
            label="Download Predictions",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="frog_species_predictions.csv",
            mime="text/csv"
        )

        st.success("Model evaluation completed successfully.")

    except Exception as error:
        st.error(f"An error occurred during evaluation: {error}")
