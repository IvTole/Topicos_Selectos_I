from sklearn import metrics

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    # accuracy score
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"{model}: accuracy={accuracy:.2f} %")

    # auc score
    y_probs = model.predict_proba(X_test)[:,1]
    auc_score = metrics.roc_auc_score(y_test, y_probs)
    print(f"{model}: auc={auc_score:.2f} %")