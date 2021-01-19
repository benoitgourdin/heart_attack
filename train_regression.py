from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def regression(X_train, X_test, y_train, y_test):

    # train algorithm
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    # get score
    y_pred = regressor.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("score of the algorithm: " + str(score))

    return regressor
