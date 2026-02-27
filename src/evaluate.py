import os
import sys
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    model_path = "models/model.pkl"

    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model not found. Run: python src/train.py", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Load trained model
    model = joblib.load(model_path)

    # Make predictions
    preds = model.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")

    # Quality gate
    threshold = 0.90
    if acc < threshold:
        raise ValueError(
            f"Accuracy too low: {acc:.4f} < {threshold}"
        )

    print("🎉 Evaluation passed.")


if __name__ == "__main__":
    main()
