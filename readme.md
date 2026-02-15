**Machine Learning Assignment – 2**

M.Tech (AIML / DSE) – Machine Learning

Work Integrated Learning Programmes Division, BITS Pilani

Name: Yazhini R

Roll no: 2024dc04255


**a. Problem Statement**

The objective of this assignment is to design, implement, and deploy multiple machine learning classification models on a real-world dataset. The task involves training different classifiers, evaluating them using standard performance metrics, and deploying the trained models through an interactive Streamlit web application.

The assignment demonstrates the end-to-end machine learning workflow, including data preprocessing, model training, evaluation, UI development, and cloud deployment.


**b. Dataset Description**

Dataset Name: Heart Disease Dataset (Cleveland)

Dataset Source: UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/heart+disease


**Dataset Overview**

The dataset contains clinical and demographic information of patients and is used to predict the presence or absence of heart disease.

Number of Instances: ~920 (after cleaning)

Number of Features: 13 (excluding target)

Problem Type: Binary Classification

**Target Variable**

0- No Heart Disease

1-Presence of Heart Disease


**Features**

Age

Sex

Chest pain type

Resting blood pressure

Serum cholesterol

Fasting blood sugar

Resting ECG results

Maximum heart rate achieved

Exercise induced angina

ST depression

Slope of peak exercise ST segment

Number of major vessels

Thalassemia


**c. Models Used \& Evaluation Metrics**


All models were trained on the same dataset using a consistent preprocessing pipeline.

The following evaluation metrics were calculated for each model:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)


| ML Model Name            | Accuracy | AUC     | Precision | Recall | F1 Score | MCC     |

| ------------------------ | -------- | ------  | --------- | ------ | -------- | ------- |

| Logistic Regression      | 0.8333   | 0.9497  | 0.8461    | 0.7857 | 0.8148   | 0.6651  |

| Decision Tree            | 0.7000   | 0.6964  | 0.6964    | 0.6428 | 0.6666   | 03955   |

| kNN                      | 0.8833   | 0.9492  | 0.9200    | 0.8214 | 0.8679   | 0.7679  |

| Naive Bayes              | 0.8833   | 0.9375  | 0.8888    | 0.8571 | 0.8727   | 0.7655  |

| Random Forest (Ensemble) | 0.8500   | 0.9408  | 0.8800    | 0.7857 | 0.8301   | 0.7002  |

| XGBoost (Ensemble)       | 0.8666   | 0.8917  | 0.8846    | 0.8214 | 0.8518   | 0.7326  |


**d.Model Performance Observations**


| ML Model Name            | Observation about Model Performance                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performed well as a baseline model with balanced precision and recall, indicating effective linear separation of features.  |
| Decision Tree            | Captured non-linear relationships but showed signs of overfitting compared to ensemble models.                              |
| kNN                      | Performance depended on distance-based learning and benefited from feature scaling, but was sensitive to data distribution. |
| Naive Bayes              | Demonstrated fast computation and stable performance but assumed feature independence, limiting accuracy.                   |
| Random Forest (Ensemble) | Improved overall performance by reducing overfitting and capturing complex feature interactions.                            |
| XGBoost (Ensemble)       | Achieved the best performance across most metrics due to gradient boosting and optimized learning strategy.                 |

**e. GitHub Repository Structure**

project-folder/
│── app.py
│── requirements.txt
│── README.md
│
└── model/
    │── data_loader.py
    │── evaluation.py
    │── logistic_regression.py
    │── decision_tree.py
    │── knn.py
    │── naive_bayes.pyf. 
    │── random_forest.py
    │── xgboost_model.py


**f.Streamlit Application Features** 

The Streamlit web application includes the following mandatory features:

CSV dataset upload option (test data only)

Model selection dropdown

Display of evaluation metrics

Confusion matrix / classification report visualization

The application has been deployed using Streamlit Community Cloud.


**g. Deployment**

Steps followed for deployment:

Pushed complete project code to GitHub

Logged into Streamlit Community Cloud

Selected repository and app.py

Deployed the application successfully







