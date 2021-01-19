import pandas as pd

from prepare_data import prepareData
from train_regression import regression

if __name__ == '__main__':
    X = [[56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2]]
    X_df = pd.DataFrame(X, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                    'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    X_train, X_test, y_train, y_test, scaler = prepareData()
    model = regression(X_train, X_test, y_train, y_test)
    X_scaled = scaler.transform(X_df)
    result = model.predict(X_scaled)
    if result == 1:
        print("heart attack")
    elif result == 0:
        print("no heart attack")
