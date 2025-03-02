import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_random_forest_regressor(data):
    """
    Функция для обучения модели случайного леса на данных недвижимости.
    """
    renovation_weights = {
        'Without renovation': 0,
        'Cosmetic': 1,
        'European-style renovation': 2,
        'Designer': 3
    }
    
    data['Renovation'] = data['Renovation'].map(renovation_weights)
    
    data['Price'].plot(kind='box')
    
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    X = pd.get_dummies(X, drop_first=True)
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R²: {r2}')
    
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    print("\nВажность признаков:")
    print(feature_importances)
    
    y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Вывод первых 10 значений
    predicted_values = [(y_pred_original_scale[i][0], y_test_original_scale[i][0]) for i in range(10)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original_scale, y_pred_original_scale)
    plt.plot([min(y_test_original_scale), max(y_test_original_scale)], [min(y_test_original_scale), max(y_test_original_scale)], color='red', linestyle='dashed')
    plt.title('Фактические vs Предсказанные значения')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.show()
    
    return mse, r2, feature_importances, predicted_values

def train_linear_regression(data):
    """
    Функция для обучения линейной регрессии на данных недвижимости.
    """
    renovation_weights = {
        'Without renovation': 0,
        'Cosmetic': 1,
        'European-style renovation': 2,
        'Designer': 3
    }
    
    data['Renovation'] = data['Renovation'].map(renovation_weights)
    
    data['Price'].plot(kind='box')
    
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    X = pd.get_dummies(X, drop_first=True)
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train.ravel())
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R²: {r2}')
    
    feature_importances = pd.DataFrame(model.coef_, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    print("\nВажность признаков:")
    print(feature_importances)
    
    y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Вывод первых 10 значений
    predicted_values = [(y_pred_original_scale[i][0], y_test_original_scale[i][0]) for i in range(10)]
    
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original_scale, y_pred_original_scale)
    plt.plot([min(y_test_original_scale), max(y_test_original_scale)], [min(y_test_original_scale), max(y_test_original_scale)], color='red', linestyle='dashed')
    plt.title('Фактические vs Предсказанные значения')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.show()
    
    return mse, r2, feature_importances, predicted_values

def train_svr(data):
    """
    Функция для обучения модели опорных векторов на данных недвижимости.
    """
    renovation_weights = {
        'Without renovation': 0,
        'Cosmetic': 1,
        'European-style renovation': 2,
        'Designer': 3
    }
    
    data['Renovation'] = data['Renovation'].map(renovation_weights)
    
    data['Price'].plot(kind='box')
    
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    X = pd.get_dummies(X, drop_first=True)
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train.ravel())
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R²: {r2}')
    

    # Оценка важности признаков на меньшей подвыборке (например, 500 объектов)
    X_test_sample = X_test[:500] if len(X_test) > 500 else X_test
    y_test_sample = y_test[:500] if len(y_test) > 500 else y_test

    # Используем Permutation Importance для оценки важности признаков
    perm_importance = permutation_importance(model, X_test_sample, y_test_sample.ravel(), n_repeats=3, random_state=42, n_jobs=-1)
    feature_importances = pd.DataFrame(perm_importance.importances_mean, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    print("\nВажность признаков (Permutation Importance):")
    print(feature_importances)
    
    y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Вывод первых 10 значений
    predicted_values = [(y_pred_original_scale[i][0], y_test_original_scale[i][0]) for i in range(10)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original_scale, y_pred_original_scale)
    plt.plot([min(y_test_original_scale), max(y_test_original_scale)], [min(y_test_original_scale), max(y_test_original_scale)], color='red', linestyle='dashed')
    plt.title('Фактические vs Предсказанные значения')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.show()
    
    return mse, r2, feature_importances, predicted_values

def train_gb(data):
    """
    Функция для обучения модели градиентного бустинга на данных недвижимости.
    """
    renovation_weights = {
        'Without renovation': 0,
        'Cosmetic': 1,
        'European-style renovation': 2,
        'Designer': 3
    }
    
    data['Renovation'] = data['Renovation'].map(renovation_weights)
    
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    X = pd.get_dummies(X, drop_first=True)
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # ОГРАНИЧИВАЕМ ВЫБОРКУ ДЛЯ УСКОРЕНИЯ
    X_train_sample = X_train[:20000]
    y_train_sample = y_train[:20000]

    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.2, random_state=42)
    model.fit(X_train_sample, y_train_sample.ravel())
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R²: {r2}')
    
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    print("\nВажность признаков (Gradient Boosting):")
    print(feature_importances)
    
    y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Вывод первых 10 значений
    predicted_values = [(y_pred_original_scale[i][0], y_test_original_scale[i][0]) for i in range(10)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original_scale, y_pred_original_scale)
    plt.plot([min(y_test_original_scale), max(y_test_original_scale)], [min(y_test_original_scale), max(y_test_original_scale)], color='red', linestyle='dashed')
    plt.title('Фактические vs Предсказанные значения')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.show()
    
    return mse, r2, feature_importances, predicted_values