from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import  StratifiedKFold, cross_validate
class ModelTrainer:
    def __init__(self):
        pass
    def train_model(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        return model
    def predict(self, model, x_test):
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        return y_pred, y_proba    
    def evaluate_model(self, y_test, y_pred, y_proba):
        f1_sco = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auc_pr = auc(recall, precision)
        cm = confusion_matrix(y_test, y_pred)
        print("F1-score:", f1_sco)
        print("AUC-PR:", auc_pr)
        print("Confusion Matrix:\n", cm)
        return {
            "f1_score": f1_sco,
            "auc_pr": auc_pr,
            "confusion_matrix": cm
        }
    def cross_validation(self, model, x, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            'f1': 'f1',
            'auc_pr':'average_precision'
        }
        c_validate = cross_validate(model, x, y, cv=skf, scoring=scoring, n_jobs=-1)
        results = {
        'f1_mean': c_validate['test_f1'].mean(),
        'f1_std': c_validate['test_f1'].std(),
        'auc_pr_mean': c_validate['test_auc_pr'].mean(),
        'auc_pr_std': c_validate['test_auc_pr'].std()
        }
        return c_validate, results