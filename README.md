# Heart-Diseases-Risk-Prediction
Heart Diseases Risk Prediction App
by Poojitha Gollapudi 
Overview
This project analyses patient health records using machine learning to predict the risk of heart disease. It includes data preprocessing, exploratory analysis, model training, evaluation, and a Stream lit-based dashboard for real-time predictions.
Project Structure
- Uses an open-source healthcare dataset.
- Data preprocessing & feature engineering.
- EDA with visualization.
- ML models: Logistic Regression, Random Forest, XGBoost, SVM.
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC.
- Stream lit dashboard for live predictions.
Health_Risk_Prediction/
├── data/
│   └── heart.csv
├── notebooks/
│   └── heart_disease_prediction.ipynb
├── models/
│   ├── best_model.pkl
│   └── feature_columns.pkl
├── app/
│   └── app.py
├── README.docx

Key Features
Dataset Loading & Understanding – Importing an open-source healthcare dataset and examining structure, statistics, and variable descriptions.
Data Cleaning – Handling missing values, duplicates, incorrect formats, and inconsistent entries.
Data Preprocessing – Encoding categorical variables, scaling numerical features, and preparing data for modeling.
Feature Engineering – Selecting relevant medical attributes and ensuring proper feature alignment for model training and prediction.
Exploratory Data Analysis (EDA) – Creating correlation heatmaps, distribution plots, boxplots, and trend visualizations to identify key health indicators.
Model Building – Training multiple machine learning models including Logistic Regression, Random Forest, SVM, and XGBoost.
Model Evaluation – Comparing models using accuracy, precision, recall, F1-score, and ROC-AUC curves.
Model Selection & Saving – Automatically identifying the best-performing model and saving it as best_model.pkl along with feature_columns.pkl.
Prediction Pipeline – Creating a standardized input-to-output prediction flow ensuring consistent results.
Streamlit Dashboard – Developing an interactive UI for entering patient details and generating live disease risk predictions.
Deployment Ready – Final project structured for easy execution, reuse, and extension.

Installation & Dependencies
To run this project, install the required Python libraries using the following command:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit pickle5
Or install everything from a requirements.txt file (recommended):
pip install -r requirements.txt
Dependencies Used:
pandas – Data loading and manipulation
numpy – Numerical computations
matplotlib – Data visualization
seaborn – Statistical plotting and EDA visuals
scikit-learn – Machine learning models and evaluation metrics
xgboost – Advanced gradient boosting model
streamlit – Interactive prediction dashboard
pickle5 – Saving and loading trained models

Usage
1.Place the dataset inside the data/ folder
2./data/heart.csv
3.Open the Jupyter Notebook for analysis:
4.jupyter notebook
Run all cells to perform:
oData cleaning
oPreprocessing
oFeature engineering
oExploratory Data Analysis (EDA)
oModel training
oModel evaluation
5.Save the best-performing model by running these lines inside your notebook:
6.pickle.dump(best_model, open('models/best_model.pkl', 'wb'))
7.pickle.dump(feature_columns, open('models/feature_columns.pkl', 'wb'))
8.Launch the Streamlit dashboard to perform real-time predictions:
9.cd app
10.streamlit run app.py
11.Enter patient details in the Streamlit UI
to get an instant prediction of heart disease risk.
12. High-Risk Example (Should Predict “High Risk”)
13.Age: 68
14.Sex: Male
15.Chest Pain Type (cp): 0
16.Resting Blood Pressure (trestbps): 160
17.Cholesterol (chol): 310
18.Fasting Blood Sugar (fbs): 1
19.Resting ECG (restecg): 1
20.Max Heart Rate (thalach): 95
21.Exercise Induced Angina (exang): 1
22.Oldpeak: 3.1
23.Slope: 0
24.Number of Major Vessels (ca): 2
25.Thal: 3

26.Low-Risk Example (Should Predict “Low Risk”)
27.Age: 40
28.Sex: Female
29.Chest Pain Type (cp): 3
30.Resting Blood Pressure (trestbps): 115
31.Cholesterol (chol): 180
32.Fasting Blood Sugar (fbs): 0
33.Resting ECG (restecg): 1
34.Max Heart Rate (thalach): 185
35.Exercise Induced Angina (exang): 0
36.Oldpeak: 0.4
37.Slope: 2
38.Number of Major Vessels (ca): 0
39.Thal: 2
Results
The dataset was successfully cleaned, preprocessed, and prepared for machine learning.
Exploratory Data Analysis (EDA) identified key health indicators such as age, cholesterol, max heart rate, oldpeak, and chest pain type as strong predictors of heart disease.
Multiple machine learning models were trained and evaluated, including Logistic Regression, Random Forest, SVM, and XGBoost.
After comparing evaluation metrics, the XGBoost Classifier (or whichever model performed best in your notebook) achieved the highest performance based on:
oAccuracy
oPrecision
oRecall
oF1-Score
oROC-AUC score
The best-performing model was saved as best_model.pkl, and the feature order was saved as feature_columns.pkl for consistent predictions.
A fully functional Streamlit dashboard was built, allowing users to input patient medical attributes and receive real-time predictions of heart disease risk.
The final output includes both the trained prediction system and a user-friendly interface accessible through any browser.

Technologies Used
Python
Pandas & NumPy – Data cleaning, manipulation, and preprocessing
Scikit-learn & XGBoost – Machine learning model development and evaluation
Matplotlib & Seaborn – Exploratory data analysis and visualizations
Streamlit – Interactive prediction dashboard
Jupyter Notebook – End-to-end analysis and model development environment

Business Applications
Early detection of potential health risks
Clinical decision support for healthcare professionals
Personalized patient risk assessment
Improved preventive healthcare strategies
Prioritization of high-risk patients for treatment or screening
Data-driven insights for hospital resource planning
This project demonstrates end-to-end disease risk prediction—from raw patient data to a deployable prediction tool that helps support healthcare decision-making.

Output
The system provides:
Cleaned and preprocessed patient health dataset
Feature-engineered dataset aligned for modeling
Multiple trained machine learning models
Evaluation reports and performance comparisons
Automatically selected best-performing model (best_model.pkl)
Streamlit dashboard for real-time patient risk prediction
Visualizations of key health indicators and model behavior
A threshold of 0.35 was chosen because:
It gave the best trade-off between recall and precision when you checked metrics.
It improved recall significantly without making accuracy drop too much.
On the ROC and Precision-Recall curves, 0.35 was near the elbow point, indicating optimal sensitivity.
This ensures the model:
Catches maximum heart disease cases
Minimizes the risk of missing an actual patient
Balances medical safety with prediction accuracy



