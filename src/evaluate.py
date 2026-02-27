from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

def main():
    if not os.path.exists("models/model.pkl"):
        print("❌ Model not found. Run: python src/train.py", file=sys.stderr)
        sys.exit(1)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load("models/model.pkl")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")
    
    if acc < 0.9:
    raise ValueError("Accuracy too low!")

if __name__ == "__main__":
    main()
