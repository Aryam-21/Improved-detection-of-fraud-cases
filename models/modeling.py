from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import  StratifiedKFold, cross_validate
import numpy as np
class ModelTrainer:
    def __init__(self):
        pass
    def validate_data(self, x, y=None):
        if x is None:
            raise ValueError("Input features cannot be None")

        if not isinstance(x, (np.ndarray, list)):
            raise TypeError("Input features must be a numpy array or list")

        if y is not None:
            if not isinstance(y, (np.ndarray, list)):
                raise TypeError("Target variable must be a numpy array or list")

            if len(x) != len(y):
                raise ValueError("Features and target must have the same length")
    def train_model(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        return model
    def predict(self, model, x_test):
        if model is None:
            raise ValueError("Model instance cannot be None")

        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise TypeError("Model must support predict() and predict_proba()")

        self._validate_data(x_test)

        try:
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
        return y_pred, y_proba    
    def evaluate_model(self, y_test, y_pred, y_proba):
        self.validate_data(y_test)

        if y_pred is None or y_proba is None:
            raise ValueError("Predictions and probabilities cannot be None")

        if len(y_test) != len(y_pred):
            raise ValueError("y_test and y_pred must have the same length")
        try:
            f1_sco = f1_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            auc_pr = auc(recall, precision)
            cm = confusion_matrix(y_test, y_pred)
        except Exception as e:
            raise RuntimeError(f"Model evaluation failed: {e}")
        print("F1-score:", f1_sco)
        print("AUC-PR:", auc_pr)
        print("Confusion Matrix:\n", cm)
        return {
            "f1_score": f1_sco,
            "auc_pr": auc_pr,
            "confusion_matrix": cm
        }
    def cross_validation(self, model, x, y, n_splits=5):
        if model is None:
            raise ValueError("Model instance cannot be None")
        self.validate_data(x,y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = {
            'f1': 'f1',
            'auc_pr':'average_precision'
        }
        try:
            c_validate = cross_validate(model, x, y, cv=skf, scoring=scoring, n_jobs=-1)
        except Exception as e:
            raise RuntimeError(f"Cross-validation failed: {e}")
        results = {
        'f1_mean': c_validate['test_f1'].mean(),
        'f1_std': c_validate['test_f1'].std(),
        'auc_pr_mean': c_validate['test_auc_pr'].mean(),
        'auc_pr_std': c_validate['test_auc_pr'].std()
        }
        return c_validate, results