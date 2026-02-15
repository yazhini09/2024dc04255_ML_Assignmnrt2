from xgboost import XGBClassifier
from model.evaluation import evaluate_model

def train_xgboost(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    return model, metrics
