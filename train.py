from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Paths are based on the location of train.py.
PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_FOLDER / "MagicBricks_cleaned.csv"
MODEL_FOLDER = PROJECT_FOLDER / "model"
MODEL_PATH = MODEL_FOLDER / "house_price_model.joblib"


# Columns used by app.py to make a prediction.
FEATURE_COLUMNS = [
    "Area",
    "BHK",
    "Bathroom",
    "Furnishing",
    "Locality",
    "Parking",
    "Status",
    "Transaction",
    "Type",
]

TARGET_COLUMN = "Price"

NUMERICAL_FEATURES = [
    "Area",
    "BHK",
    "Bathroom",
    "Parking",
]

CATEGORICAL_FEATURES = [
    "Furnishing",
    "Locality",
    "Status",
    "Transaction",
    "Type",
]


def train_model():
    """Load the data, train the model, evaluate it, and save it."""

    # 1. Load the cleaned dataset.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.name}\n"
            "Keep train.py and MagicBricks_cleaned.csv in the same folder."
        )

    data = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully!")
    print("Dataset shape:", data.shape)

    # Check that the dataset contains every required column.
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    # 2. Separate the input features and target price.
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    # 3. Split the data: 80% for training and 20% for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("Train-test split completed successfully!")
    print("Training rows:", X_train.shape[0])
    print("Testing rows:", X_test.shape[0])

    # 4. Prepare numerical columns.
    numerical_pipeline = Pipeline(
        steps=[
            ("missing_values", SimpleImputer(strategy="median")),
            ("scaling", StandardScaler()),
        ]
    )

    # 5. Prepare categorical columns.
    categorical_pipeline = Pipeline(
        steps=[
            ("missing_values", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # 6. Combine numerical and categorical preprocessing.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # 7. Combine preprocessing and the best model from the notebook.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 8. Train the complete pipeline.
    model.fit(X_train, y_train)
    print("Random Forest model trained successfully!")

    # 9. Evaluate the trained model.
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae / 100000:.2f} lakh")
    print(f"RMSE: {rmse / 100000:.2f} lakh")
    print(f"R2 score: {r2:.3f}")

    # 10. Save the complete preprocessing and model pipeline.
    MODEL_FOLDER.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully!")
    print("Saved location:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
