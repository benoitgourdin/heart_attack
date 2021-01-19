from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def knn(X_train, X_test, y_train, y_test):

    param = {
        'n_neighbors': range(1, 20),
        'weights': ('distance', 'uniform')}
    model = KNeighborsClassifier()
    cv_model = GridSearchCV(model, param, cv=10, n_jobs=-1, verbose=2)
    cv_model.fit(X_train, y_train)
    print('Best parameters:' + str(cv_model.best_params_))

    num_neighbors = 17
    weights = 'uniform'

    # train algorithm
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights)
    knn.fit(X_train, y_train)

    # get score
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("score of the algorithm: " + str(score))

    return knn
