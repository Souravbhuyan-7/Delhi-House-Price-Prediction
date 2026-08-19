# Delhi House Price Prediction

## Project Overview

This machine-learning project predicts house prices in Delhi using property details such as area, BHK, bathrooms, furnishing, locality, parking, property status, transaction type, and property type.

The dataset was cleaned, analysed, preprocessed, and used to train multiple regression models. The best-performing model was saved and connected to an interactive Streamlit application.

## Dataset

The project uses the `MagicBricks.csv` Delhi housing dataset downloaded from Kaggle.

* Original records: 1,259
* Cleaned records: 1,154
* Target column: `Price`
* Input features: 9

## Project Workflow

1. Loaded and understood the dataset
2. Handled missing and invalid values
3. Removed duplicate records
4. Performed exploratory data analysis
5. Selected features and target
6. Split data into training and testing sets
7. Created a preprocessing pipeline
8. Trained and compared regression models
9. Saved the best model using Joblib
10. Created a Streamlit prediction application

## Models Used

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Random Forest was selected as the best model based on its evaluation results.

## Technologies

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Project Structure

```text
Delhi House Price Prediction/
├── model/
│   └── house_price_model.joblib
├── app.py
├── house price.ipynb
├── MagicBricks.csv
├── MagicBricks_cleaned.csv
├── requirements.txt
└── README.md
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the application in a browser:

```text
http://localhost:8501
```

> The predicted price is a machine-learning estimate and should not be considered an official property valuation.

