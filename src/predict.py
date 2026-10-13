"""Load the saved pipeline and predict survival for example passengers.

Run:  python src/predict.py
"""
from pathlib import Path

import joblib
import pandas as pd

from train import add_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "titanic_pipeline.joblib"

EXAMPLES = pd.DataFrame(
    [
        {"Name": "Doe, Mrs. Jane", "Pclass": 1, "Sex": "female", "Age": 34, "SibSp": 1, "Parch": 0, "Fare": 80.0, "Embarked": "S"},
        {"Name": "Smith, Mr. John", "Pclass": 3, "Sex": "male", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 7.9, "Embarked": "S"},
    ]
)


def main() -> None:
    model = joblib.load(MODEL_PATH)
    # Titles that are rare in training are mapped to "Rare"; unseen titles are ignored by the encoder.
    data = add_features(EXAMPLES.assign(Survived=0))
    cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass", "Sex", "Embarked", "Title", "IsAlone"]
    proba = model.predict_proba(data[cols])[:, 1]
    for (_, row), p in zip(EXAMPLES.iterrows(), proba):
        print(f"{row['Name']:<20} survival probability: {p:.2f}")


if __name__ == "__main__":
    main()
