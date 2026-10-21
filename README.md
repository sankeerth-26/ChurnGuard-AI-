# ChurnGuard AI — Customer Churn Prediction Platform

A machine learning application that predicts customer churn risk and provides insights into the factors that may influence customer retention.

##  Live Demo

**Try the application:** [ChurnGuard AI — Live Demo](https://churnguard-ai-a.streamlit.app/)

**GitHub Repository:** [ChurnGuard AI](https://github.com/sankeerth-26/ChurnGuard-AI-)

## 📌 Overview

ChurnGuard AI uses a trained XGBoost model to analyze customer information and estimate churn risk. The application provides an interactive dashboard for exploring customer profiles, generating predictions, analyzing important churn drivers, and viewing business recommendations.

##  Features

- **Customer Profile Analysis:** Enter customer attributes through an interactive sidebar.
- **Churn Risk Prediction:** Estimate customer churn risk using the trained machine learning pipeline.
- **Batch CSV Prediction:** Upload customer records for bulk prediction.
- **Model Performance Metrics:** Display evaluation metrics, including F1 score, ROC-AUC, and accuracy.
- **Explainable AI:** Explore important churn factors using SHAP visualizations.
- **Business Recommendations:** Present potential retention actions based on identified risk factors.
- **Interactive Dashboard:** Explore customer insights through a custom dark-themed Streamlit interface.

##  Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook

## 📂 Project Structure

```text
ChurnGuard-AI-/
├── .streamlit/
│   └── config.toml
├── visuals/
│   ├── confusion_matrices.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_xgb.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── shap_dependence_wallet.png
│   ├── shap_summary_bar.png
│   ├── shap_summary_beeswarm.png
│   ├── target_distribution.png
│   └── README.md
├── app.py
├── churn.csv
├── churn_model_pipeline.pkl
├── churn_prediction.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.11
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sankeerth-26/ChurnGuard-AI-.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ChurnGuard-AI-
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Launch the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in your terminal, usually [http://localhost:8501](http://localhost:8501).

##  How to Use

1. Launch the Streamlit application.
2. Enter customer attributes in the sidebar.
3. Use the single-customer prediction interface to evaluate an individual customer.
4. Use the batch prediction interface to process a CSV file.
5. Review churn-related insights, model metrics, and business recommendations.
6. Explore SHAP visualizations to understand factors influencing predictions.

##  Model and Evaluation

The application uses a pre-trained XGBoost pipeline. The dashboard displays evaluation metrics reported by the existing project. These figures should not be interpreted as independently verified results for new datasets.

Predictions depend on the trained model, input data, and compatibility of the installed dependencies. The serialized model should be tested with the documented package versions before being used for important business decisions.

##  Future Improvements

- Improve model validation and probability calibration.
- Add customer segmentation and retention analytics.
- Improve prediction explanations and reporting.
- Develop additional interactive business intelligence dashboards.

## Acknowledgment

This project is a customized version of the open-source project originally published by **Victor Pius**:

[Original Repository — customer-churn-prediction](https://github.com/profpius/customer-churn-prediction)

The original MIT license and applicable copyright notices are retained in the `LICENSE` file. Please refer to that file for the applicable license terms.

##  Author

**Sankeerth S**

- **GitHub:** [sankeerth-26](https://github.com/sankeerth-26)
- **LinkedIn:** [Sankeerth S](https://www.linkedin.com/in/sankeerth-s-1bbba1294)

---

*ChurnGuard AI — Customer Churn Prediction Platform*
