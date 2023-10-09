import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
try:
    datos = pd.read_excel(r"C:\Users\LEONARDO ACUÑA\Desktop\predicion\bateria.xlsx")
    # print(datos)  # O cualquier otra operación que planees hacer con los datos
except Exception as e:
    print(f"Error al leer el archivo Excel: {e}")
x=datos["Tiempo"]
y=datos["Carga"]
X=x.values.reshape(-1, 1)
while True :

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test= train_test_split(X,y)

    mlr=MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(3,3),random_state=1)
    mlr.fit(X_train,Y_train)
    print(mlr.score(X_train,Y_train))
    if mlr.score(X_train,Y_train)>0.99:
        break
print("prediccion en t=20 min",mlr.predict(np.array([20]).reshape(-1, 1)))