## Customer Churn Prediction Project
A simple customer churn prediction project for Telecommunitation company. Goal is to build a ML model to predict the churn of the customers. Business usecase is to identify the target customers for retention programs and promotions. So, that more customers are retained on the platform and business impact is low. 


### Purpose

This project will help me in showcasing the ML Engineering Simulation involving Feature Engineering, Model Training, Deployment with Inference, Monitoring and Batch Processing.

#### Dataset Used
* Dataset : Popular Dataset on Kaggle for building simulations for Customer Churn in Telecom.
* Dataset URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* Dataset Features: 
    - Demographic Information: Age, Gender, have any dependents and partner etc.
    - Customer Account Information: Duration since active customer, contract, billinh, payment method, monthly and total charges etc.
    - Services Signed Up for: Phone, Multiple lines, Internet, Online security, Online backup, Device protection, Tech support, and Streaming TV and movies etc.  
* Dependent Variable - Churn (customers who left within the last month)


### Steps for building a simulation of ML pipelines

#### Data Engineering
* Purpose of Data Engineering is to collect data from different sources aggregate it and bring it to one source.
* As we are using a dataset from Kaggle which is already aggregated. In this case we just need to do simple data cleaning, outlier removal and handling empty values.
* But ideally in production Data Engineering takes a significant effort and time. As we need to pull the data from various sources aggregate it and bring it to similar scale.
* This is the very first and extremely important step for ML models. As this will help us in building ML models at scalable and robust. Help in address the Data Quality.

### Feature Engineering
* Convert data to features which can be used to train the ML model
* Perform Data Standardization
* Extract new features from data. Example like if we have raw timestamp of last login in the data, we can create a feature of days from last login.
* Remove any unwanted data sources, for extracting better signals for better ML model
* In production use cases, we store all the generated features in a feature store. From where we can pick these features, retrain the model.


#### Model Training
* Goal is not to just create a ML model. But build a training pipeline which can be used to train a model using extracted features. This pipeline can be used to retrain the model as per requirement.
* We split the data into Train, Validation and Test set. We use training dataset to train the model and learn the patterns from the data. Then validation sets are used for optimizing the hyperparameters. In practice we usually use Cross validation.
* Finally calculate and monitor the perfomance on Train and Test set to compare it. If there is a significant difference means model is either underfit or overfit (assuming the data distribution of train and test set is similar).
* Metrics for monitoring the performance vary based on the data, mode, business etc.

1. Model Performance Metrics
2. Business Metrics

#### Model Inference Generation
* Again this is a pipeline, which will involve data processing, feature engineering, model prediction, store the predictions for monitoring and return the prediction.
* It also includes Validation of Input Data.

#### Model Performance Monitoring and Retraining
1. Logging Experimentation
2. Logging Model Training [Performance Metrics, Latency, Cost]
3. Statistical and EDA on inference data
4. Monitoring results from Exploratory Data Analysis
5. Logging Inference Time Results including Latency and Cost.
6. Monitoring Model Performance including Data Drift and Model Drift
7. Storing back the inference results and use them for training ML models


#### Batch Processing Simulation
[TODO]



#### Testing 
* Unit Testing
* Integration Testing
* Performance Testing
* A/B Testing
* Stress Testing [Out of scope]
* User Acceptance Testing [Out of scope]

### Technology
1. Python
2. XGBoost
3. Github
4. Docker
5. REST API
6. FAST API
7. Streamlit App
8. MLFlow
9. AirFlow
9. SQLite
10. Jupyter Notebook for experimentation


### Resources
1. Made with ML
2. Designing ML Systems by Chip Huyen
3. ML Engineering for Production Course by Andrew NG

### Next Steps
[TODO]





