import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def get_dataframe(url):
    # Descargar el contenido del archivo
    response = requests.get(url)
    data_text = response.text

    # Dividir el contenido en líneas
    lines = data_text.splitlines()

    # Saltar las primeras líneas que contienen la descripción y capturar solo los datos
    # En este caso, los datos empiezan en la línea 23 aproximadamente (ajustar si es necesario)
    data_lines = lines[22:]

    # Los datos están organizados en dos filas por cada entrada
    # Procesamos dos líneas a la vez para formar una fila de datos completa
    data = []
    for i in range(0, len(data_lines), 2):
        line1 = data_lines[i].strip()
        line2 = data_lines[i + 1].strip() if i + 1 < len(data_lines) else ''
        
        # Dividir ambas líneas en valores y combinarlos
        values = line1.split() + line2.split()
        data.append(values)

    # Crear un DataFrame y asignar nombres de columnas
    columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT", "MEDV"]
    df = pd.DataFrame(data, columns=columns)

    # Convertir las columnas a tipo numérico
    df = df.apply(pd.to_numeric)

    # Mostrar las primeras filas del DataFrame
    return df

class RegresionLogistica:
    def __init__(self):
        self.model = LogisticRegression()

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        kappa = cohen_kappa_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Matriz de confunsión para Regresion Logistica')
        plt.show()



class RegresionLogisticaLasso:
    def __init__(self, Cs=10, cv=5, random_state=42):
        self.model = LogisticRegressionCV(Cs=Cs, cv=cv, penalty='l1', solver='liblinear', random_state=random_state)

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        kappa = cohen_kappa_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Matriz de confusión para Regresion Logistica Lasso')
        plt.show()

    def get_optimal_alpha(self):
        return self.model.C_[np.argmax(self.model.scores_[1].mean(axis=0))]
    
    def coefs(self):
        return self.model.coef_
    
    def get_score(self):
        return self.model.scores_

class RegresionLogisticaRidge:
    def __init__(self, Cs=10, cv=5, random_state=42):
        self.model = LogisticRegressionCV(Cs=Cs, cv=cv, penalty='l2', solver='lbfgs', random_state=random_state)

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        kappa = cohen_kappa_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Matriz de confusión para Regresion Logistica Ridge')
        plt.show()

    def get_optimal_alpha(self):
        return self.model.C_[np.argmax(self.model.scores_[1].mean(axis=0))]
    
    def coefs(self):
        return self.model.coef_
    
    def get_score(self):
        return self.model.scores_


class RegresionElasticNet:
    def __init__(self, Cs=10, cv=5, l1_ratios=[0.1, 0.5, 0.9], random_state=42):
        self.model = LogisticRegressionCV(Cs=Cs, cv=cv, penalty='elasticnet', solver='saga', l1_ratios=l1_ratios, random_state=random_state)

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        kappa = cohen_kappa_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Kappa: {kappa:.2f}')
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Matriz de confusión para Regresion Logistica ElasticNet')
        plt.show()