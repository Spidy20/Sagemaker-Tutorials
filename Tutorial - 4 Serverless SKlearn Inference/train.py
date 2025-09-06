import logging
import os
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import sklearn


def setup_logging(log_dir: str = "logs", log_name: str | None = None) -> None:
    os.makedirs(log_dir, exist_ok=True)
    if log_name is None:
        log_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_name)

    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()  # console
        ],
    )
    logging.info("Logging initialized. Log file: %s", log_path)


def main():
    setup_logging()

    logger = logging.getLogger("trainer")
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("scikit-learn version: %s", sklearn.__version__)

    # 1) Load data
    data = load_iris()
    X, y = data.data, data.target
    logger.info("Loaded Iris dataset: X shape=%s, y shape=%s", X.shape, y.shape)
    logger.info("Classes: %s", list(data.target_names))

    # 2) Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        "Split data: train=%d, test=%d", X_train.shape[0], X_test.shape[0]
    )

    # 3) Build pipeline: Standardize -> LogisticRegression
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None, random_state=42)),
        ]
    )
    logger.info("Pipeline: %s", pipeline)

    # 4) Train
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # 5) Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data.target_names)
    logger.info("Test Accuracy: %.4f", acc)
    logger.info("Classification Report:\n%s", report)

    # 6) Save model
    model_path = "model-inference-1.0.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved trained model to: %s", model_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
