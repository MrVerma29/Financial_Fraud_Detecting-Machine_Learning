# Financial Fraud Detection Project

## Project Overview
This project aims to detect fraudulent transactions in a simulated financial dataset using machine learning models. The goal is to predict whether a transaction is fraudulent based on several transaction-specific features. Fraud detection in financial systems is critical for safeguarding financial institutions and customers against financial losses due to fraudulent activities.

## Dataset Description
The dataset used in this project consists of transactional data with the following features:

- **step**: Maps a unit of time in the real world. Each step represents 1 hour. The dataset covers 744 steps (equivalent to 30 days).
- **type**: Type of the transaction, which can be one of the following:
  - `1` - CASH-IN
  - `2` - CASH-OUT
  - `3` - DEBIT
  - `4` - PAYMENT
  - `5` - TRANSFER
- **amount**: The amount of the transaction in local currency.
- **nameOrig**: ID of the customer initiating the transaction.
- **oldbalanceOrg**: Initial balance of the customer before the transaction.
- **newbalanceOrig**: New balance of the customer after the transaction.
- **nameDest**: ID of the recipient of the transaction.
- **oldbalanceDest**: Initial balance of the recipient before the transaction (not available for merchants).
- **newbalanceDest**: New balance of the recipient after the transaction (not available for merchants).
- **isFraud**: Binary label indicating if the transaction is fraudulent (1) or not (0).
- **isFlaggedFraud**: Binary label indicating if the transaction was flagged as potential fraud due to large transfer attempts (i.e., transactions above 200,000 in a single transfer).

## Project Workflow
1. **Data Preprocessing**:
   - Removed irrelevant columns (like `oldbalanceOrg`) that introduce multicollinearity.
   - Handled missing values and outliers, if any.
   - Encoded the categorical features, such as the `type` of transaction.
   - Scaled the numerical features to normalize the data for model training.

2. **Exploratory Data Analysis (EDA)**:
   - Conducted univariate, bivariate, and multivariate analysis to understand the distribution of the data and relationships between features.
   - Plotted bar charts to show the distribution of transaction types and analyzed their correlation with fraud.
   - Used heatmaps to observe correlations between features.

3. **Feature Selection**:
   - Selected key features based on domain knowledge and multicollinearity checks.

4. **Modeling**:
   - Applied machine learning algorithms such as **Random Forest** and **XGBoost** for fraud detection.
   - Performed cross-validation to ensure model robustness and prevent overfitting.
   - Evaluated the models using confusion matrix, classification report, and ROC curve.

5. **Model Evaluation**:
   - Confusion matrix was used to observe the true positive, false positive, true negative, and false negative counts.
   - Precision, recall, F1-score, and accuracy metrics were calculated to assess model performance.
   - ROC curves were plotted to visualize the model’s performance in distinguishing between fraudulent and non-fraudulent transactions.

## Key Findings
- **Random Forest** achieved a cross-validated accuracy of **92.67%**, while **XGBoost** achieved a slightly better accuracy of **93.53%**.
- Both models performed well in identifying fraudulent transactions with a high recall and precision, indicating that they correctly identified a large portion of fraud cases with minimal false positives.

## Tools and Libraries Used
This project was developed using the following libraries:

- **NumPy**: Used for numerical computations.
- **Pandas**: Used for data manipulation and analysis.
- **Matplotlib** and **Seaborn**: Used for visualizing the dataset and analysis.
- **Scikit-learn**:
  - Preprocessing: Used for scaling and encoding features.
  - Model training: Random Forest and XGBoost models were used.
  - Evaluation metrics: Used for confusion matrix, classification reports, and ROC curves.
- **XGBoost**: A high-performance library for gradient-boosted decision trees, used for fraud detection modeling.

## How to Use
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your_username/financial-fraud-detection.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script to see the model in action:
   ```bash
   python fraud_detection.py
   ```

## Key Features to Predict Fraud
The following key features were used to predict fraudulent transactions:
1. **Transaction Type (type)**: Certain transaction types like TRANSFER and CASH_OUT are more likely to be involved in fraud.
2. **Transaction Amount (amount)**: High transaction amounts may indicate a higher chance of fraud.
3. **Balance Variables**: Discrepancies between old and new balances (both for origin and destination accounts) were important indicators for fraudulent transactions.

## Future Work
- Implement other advanced machine learning techniques like neural networks.
- Apply more sophisticated techniques for feature engineering.
- Explore the use of unsupervised learning methods to detect potential fraud cases without labels.

## Conclusion
This project successfully built and evaluated machine learning models to detect fraudulent financial transactions with high accuracy and precision. The results show that both Random Forest and XGBoost are highly effective in identifying fraudulent behavior, and their use in financial systems could significantly reduce the occurrence of financial fraud.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
