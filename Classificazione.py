# Import delle librerie necessarie
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Caricamento dei dati (sostituire 'path_to_dataset' con il percorso effettivo del dataset)
dataset = pd.read_csv('synthetic_tumor_data.csv')

# Separazione delle variabili di input (X) e della classe target (y)
X = dataset.iloc[:, :-1].values  # tutte le colonne tranne l'ultima
y = dataset.iloc[:, -1].values   # solo l'ultima colonna

# Suddivisione in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creazione della rete neurale
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # o 'softmax' se ci sono più classi

# Compilazione del modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # usare 'categorical_crossentropy' per classificazione multi-classe

# Addestramento del modello
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Predizione su nuovi dati
predictions = model.predict(X_test)

# Formattazione delle predizioni per l'output (se necessario)
formatted_predictions = np.round(predictions).astype(int)  # per classificazione binaria
