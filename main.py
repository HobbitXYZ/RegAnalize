import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from func import train_random_forest_regressor, train_linear_regression, train_svr, train_gb

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Обучение моделей")
        self.setGeometry(100, 100, 270, 550)  # Размеры окна можно настроить

        # Инициализация атрибутов для предсказанных и реальных значений
        self.predicted_values = None
        self.actual_values = None

        # Layout для размещения виджетов
        layout = QVBoxLayout()

        # Кнопки для каждого метода
        self.btn_rf = QPushButton("Обучить случайный лес", self)
        self.btn_lr = QPushButton("Обучить линейную регрессию", self)
        self.btn_svr = QPushButton("Обучить опорные векторы", self)
        self.btn_gb = QPushButton("Обучить градиентный бустинг", self)
        
        # Место для вывода результатов
        self.result_label = QLabel("Результаты будут отображены здесь.", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Место для отображения таблицы с важностью признаков и предсказанными значениями
        self.table_widget = QTableWidget(self)

        # Подключаем кнопки к методам
        self.btn_rf.clicked.connect(self.train_random_forest)
        self.btn_lr.clicked.connect(self.train_linear_regression)
        self.btn_svr.clicked.connect(self.train_svr)
        self.btn_gb.clicked.connect(self.train_gb)

        # Добавляем виджеты на layout
        layout.addWidget(self.btn_rf)
        layout.addWidget(self.btn_lr)
        layout.addWidget(self.btn_svr)
        layout.addWidget(self.btn_gb)
        layout.addWidget(self.result_label)
        layout.addWidget(self.table_widget)

        self.setLayout(layout)

    def display_feature_importances(self, feature_importances):
        # Очищаем таблицу перед добавлением новых данных
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Feature", "Importance"])

        # Добавляем данные о важности признаков
        for idx, (feature, importance) in enumerate(feature_importances.iterrows()):
            self.table_widget.insertRow(idx)
            self.table_widget.setItem(idx, 0, QTableWidgetItem(feature[0]))
            self.table_widget.setItem(idx, 1, QTableWidgetItem(f"{importance['Importance']:.4f}"))

    def display_predicted_values(self, predicted_values):
        # Очищаем таблицу перед добавлением новых данных
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Predicted", "Actual"])

        # Добавляем данные о предсказанных и реальных значениях
        for idx, (predicted, actual) in enumerate(predicted_values):
            self.table_widget.insertRow(idx)
            self.table_widget.setItem(idx, 0, QTableWidgetItem(f"{predicted:.2f}"))
            self.table_widget.setItem(idx, 1, QTableWidgetItem(f"{actual:.2f}"))

    def train_random_forest(self):
        data = pd.read_csv("real_estate_data.csv")  # Пример, замените на ваш датасет
        mse, r2, feature_importances, predicted_values = train_random_forest_regressor(data)

        # Обновляем метки с результатами
        self.result_label.setText(f"МSE: {mse:.2f}\nR²: {r2:.2f}")

        # Сохраняем предсказанные и реальные значения
        self.predicted_values, self.actual_values = zip(*predicted_values)

        # Отображаем таблицы с важностью признаков и предсказанными значениями
        self.display_feature_importances(feature_importances)
        self.display_predicted_values(predicted_values)

        # График в окне
        self.plot_graph()

    def train_linear_regression(self):
        data = pd.read_csv("real_estate_data.csv")  # Пример, замените на ваш датасет
        mse, r2, feature_importances, predicted_values = train_linear_regression(data)

        # Обновляем метки с результатами
        self.result_label.setText(f"МSE: {mse:.2f}\nR²: {r2:.2f}")

        # Сохраняем предсказанные и реальные значения
        self.predicted_values, self.actual_values = zip(*predicted_values)

        # Отображаем таблицы с важностью признаков и предсказанными значениями
        self.display_feature_importances(feature_importances)
        self.display_predicted_values(predicted_values)

        # График в окне
        self.plot_graph()

    def train_svr(self):
        data = pd.read_csv("real_estate_data.csv")  # Пример, замените на ваш датасет
        mse, r2, feature_importances, predicted_values = train_svr(data)

        # Обновляем метки с результатами
        self.result_label.setText(f"МSE: {mse:.2f}\nR²: {r2:.2f}")

        # Сохраняем предсказанные и реальные значения
        self.predicted_values, self.actual_values = zip(*predicted_values)

        # Отображаем таблицы с важностью признаков и предсказанными значениями
        self.display_feature_importances(feature_importances)
        self.display_predicted_values(predicted_values)

        # График в окне
        self.plot_graph()

    def train_gb(self):
        data = pd.read_csv("real_estate_data.csv")  # Пример, замените на ваш датасет
        mse, r2, feature_importances, predicted_values = train_gb(data)

        # Обновляем метки с результатами
        self.result_label.setText(f"МSE: {mse:.2f}\nR²: {r2:.2f}")

        # Сохраняем предсказанные и реальные значения
        self.predicted_values, self.actual_values = zip(*predicted_values)

        # Отображаем таблицы с важностью признаков и предсказанными значениями
        self.display_feature_importances(feature_importances)
        self.display_predicted_values(predicted_values)

        # График в окне
        self.plot_graph()

    def plot_graph(self):
        # Создаем новый объект Figure для отображения в PyQt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Фактические vs Предсказанные значения')
        ax.set_xlabel('Фактические значения')
        ax.set_ylabel('Предсказанные значения')

        # Отображаем график в окне
        ax.scatter(self.actual_values, self.predicted_values)
        ax.plot([min(self.actual_values), max(self.actual_values)], 
                [min(self.actual_values), max(self.actual_values)], color='red', linestyle='dashed')

        # Создаем FigureCanvas для отображения
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Добавляем canvas в layout
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())