
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "ticket_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

class TicketClassifier:

    def __init__(
        self,
        model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH,
        encoder_path=ENCODER_PATH
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.encoder_path = Path(encoder_path)

        self.model = None
        self.vectorizer = None
        self.label_encoder = None

        self.load_models()

    def load_models(self):
        

        missing_files = []

        if not self.model_path.exists():
            missing_files.append(str(self.model_path))

        if not self.vectorizer_path.exists():
            missing_files.append(str(self.vectorizer_path))

        if not self.encoder_path.exists():
            missing_files.append(str(self.encoder_path))

        if missing_files:
            raise FileNotFoundError(
                "Required model files are missing:\n"
                + "\n".join(missing_files)
                + "\n\nRun train_model.py first to generate them."
            )

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.encoder_path)

    def predict(self, ticket_text: str) -> dict:
        

        if not ticket_text or not ticket_text.strip():
            raise ValueError(
                "Ticket text cannot be empty."
            )

        
        features = self.vectorizer.transform(
            [ticket_text]
        )

        
        prediction = self.model.predict(features)[0]

        
        category = self.label_encoder.inverse_transform(
            [prediction]
        )[0]

        
        confidence = self._get_confidence(features)

        return {
            "category": category,
            "confidence": round(confidence, 4)
        }

    def _get_confidence(self, features) -> float:
       

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                features
            )

            confidence = probabilities.max()

            return float(confidence)

       
        return 0.0

    def predict_top_k(
        self,
        ticket_text: str,
        top_k: int = 3
    ) -> list:
        

        if not ticket_text or not ticket_text.strip():
            raise ValueError(
                "Ticket text cannot be empty."
            )

        features = self.vectorizer.transform(
            [ticket_text]
        )

        if not hasattr(self.model, "predict_proba"):
            result = self.predict(ticket_text)

            return [result]

        probabilities = self.model.predict_proba(
            features
        )[0]

        
        ranked_indices = probabilities.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            category = self.label_encoder.inverse_transform(
                [index]
            )[0]

            confidence = float(
                probabilities[index]
            )

            results.append(
                {
                    "category": category,
                    "confidence": round(
                        confidence,
                        4
                    )
                }
            )

        return results


def load_classifier():
    

    return TicketClassifier()


if __name__ == "__main__":

    print(
        "TicketClassifier module loaded."
    )

    print(
        "Train the model using:"
    )

    print(
        "python src/train_model.py"
    )

