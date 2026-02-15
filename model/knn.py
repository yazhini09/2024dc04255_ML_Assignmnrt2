from sklearn.neighbors import KNeighborsClassifier
from model.evaluation import evaluate_model

def train_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    return model, metrics
