from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def ann(X_train, y_train, X_test, y_test):

    # ann
    #param = {
    #    'alpha': [0.1, 0.0001],
    #    'hidden_layer_sizes': [(13, 13, 13),
    #                           (13, 13)],
    #    'solver': ['adam'],
    #    'activation': ['relu']}
    #model = MLPClassifier()
    #cv_model = GridSearchCV(model, param, cv=10, n_jobs=-1, verbose=2)
    #cv_model.fit(X_train, y_train)
    #print('Best parameters:' + str(cv_model.best_params_))
    model = MLPClassifier(activation='relu', alpha=0.5, hidden_layer_sizes=13, solver='adam',max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("score of the algorithm: " + str(score))

    return model
