import random
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

def bagging(X_train, y_train, X_test, boot_count, depth):
    
    trees = np.array([DTR(max_depth = depth) for _ in range(0, boot_count)])
    X_train_bootstrap = np.array([])
    y_train_bootstrap = np.array([])
    
    # Формирование трейн-выборок бутстрэпом:
    for i in range(0, boot_count):
        for j in range(0, X_train.shape[0]):
            random_index = random.choice([i for i in range(0, X_train.shape[0])])
            X_train_bootstrap = np.append(X_train_bootstrap, X_train[random_index])
            y_train_bootstrap = np.append(y_train_bootstrap, y_train[random_index])
    X_train_bootstrap = X_train_bootstrap.reshape(boot_count, X_train.shape[0], X_train.shape[1])
    y_train_bootstrap = y_train_bootstrap.reshape(boot_count, X_train.shape[0])
    
    # Обучаем деревья на трейн-выборках:
    fitted_trees = np.array([trees[i].fit(X_train_bootstrap[i], y_train_bootstrap[i]) for i in range(0, boot_count)])
    
    # Предсказываем ансамблем деревьев:
    y_predicts = np.array([tree.predict(X_test) for tree in fitted_trees])
    y_predicts = y_predicts.reshape(boot_count, X_test.shape[0])
    # Усреднение
    y_pred = np.array([])
    for i in range(0, X_test.shape[0]):
        mean_value = 0
        for j in range(0, boot_count):
            mean_value += y_predicts[j][i]
        mean_value = mean_value/boot_count
        y_pred = np.append(y_pred, mean_value)
    return y_pred



X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True)
y_pred = bagging(X_train, y_train, X_test, boot_count=200, depth=10)
y_dt_pred = DTR().fit(X_train, y_train).predict(X_test)
y = RandomForestRegressor().fit(X_train, y_train).predict(X_test)

print(mean_squared_error(y, y_test))
print(mean_squared_error(y_dt_pred, y_test))
print(mean_squared_error(y_pred, y_test))