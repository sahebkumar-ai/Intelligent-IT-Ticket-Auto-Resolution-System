

from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "tickets.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "ticket_classifier.pkl"
)

VECTORIZER_PATH = (
    MODEL_DIR
    / "tfidf_vectorizer.pkl"
)

ENCODER_PATH = (
    MODEL_DIR
    / "label_encoder.pkl"
)


sys.path.append(
    str(BASE_DIR)
)

from src.preprocessing import preprocess_ticket


def load_dataset():
   

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        comment=chr(96)
    )

    required_columns = [
        "ticket_text",
        "issue_category"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            
            + ", ".join(
                missing_columns
            )
        )

    data = data[
        required_columns
    ].copy()

    data = data.dropna(
        subset=[
            "ticket_text",
            "issue_category"
        ]
    )

    return data



def preprocess_dataset(
    data: pd.DataFrame
) -> pd.DataFrame:
    

    data["ticket_text"] = (
        data["ticket_text"]
        .apply(
            preprocess_ticket
        )
    )

    data = data[
        data["ticket_text"].str.len() >= 3
    ]

    return data



def train_model():
    

    print(
        "\n"
        + "=" * 65
    )

    print(
        "INTELLIGENT IT TICKET CLASSIFIER"
    )

    print(
        "=" * 65
        + "\n"
    )

   
    print(
        "[1/7] Loading dataset..."
    )

    data = load_dataset()

    print(
        f"Loaded {len(data)} tickets."
    )

    

    print(
        "\n[2/7] Preprocessing ticket text..."
    )

    data = preprocess_dataset(
        data
    )

    print(
        f"Usable tickets: {len(data)}"
    )

    
    X = data[
        "ticket_text"
    ]

    y = data[
        "issue_category"
    ]

    print(
        f"Number of issue categories: "
        f"{y.nunique()}"
    )

    

    print(
        "\n[3/7] Encoding issue categories..."
    )

    label_encoder = LabelEncoder()

    y_encoded = (
        label_encoder.fit_transform(
            y
        )
    )

    
    print(
        "\n[4/7] Splitting dataset..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )



    print(
        "\n[5/7] Creating TF-IDF features..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=20000
    )

    X_train_tfidf = (
        vectorizer.fit_transform(
            X_train
        )
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )

    print(
        f"TF-IDF features: "
        f"{X_train_tfidf.shape[1]}"
    )

    

    print(
        "\n[6/7] Training Logistic Regression model..."
    )

    classifier = LogisticRegression(
        max_iter=2000,
        C=5.0,
        class_weight="balanced",
        random_state=42
    )

    classifier.fit(
        X_train_tfidf,
        y_train
    )

   
    print(
        "\n[7/7] Evaluating model..."
    )

    y_pred = classifier.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        "\n"
        + "-" * 65
    )

    print(
        f"Classification Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "-" * 65
    )

    print(
        "\nClassification Report:"
    )

    target_names = (
        label_encoder.inverse_transform(
            sorted(
                set(y_test)
            )
        )
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0
        )
    )

   

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        classifier,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    joblib.dump(
        label_encoder,
        ENCODER_PATH
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "MODEL TRAINING COMPLETED"
    )

    print(
        "=" * 65
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nVectorizer saved to:"
        f"\n{VECTORIZER_PATH}"
    )

    print(
        f"\nLabel encoder saved to:"
        f"\n{ENCODER_PATH}"
    )

    print(
        "\n"
        + "=" * 65
    )

    return {
        "accuracy": accuracy,
        "model_path": str(
            MODEL_PATH
        ),
        "vectorizer_path": str(
            VECTORIZER_PATH
        ),
        "encoder_path": str(
            ENCODER_PATH
        )
    }




if __name__ == "__main__":

    try:

        results = train_model()

        print(
            "\nTraining Summary:"
        )

        print(
            f"Accuracy: "
            f"{results['accuracy'] * 100:.2f}%"
        )

    except Exception as exc:

        print(
            "\nTraining failed:"
        )

        print(
            str(exc)
        )

        raise

