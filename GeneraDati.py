import numpy as np
import pandas as pd

# Numero di campioni nel set di dati
n_samples = 1000  # Puoi modificare questo numero

# Numero di variabili per campione (escludendo la classe target)
n_variables = 20  # Puoi modificare questo numero

# Generazione di dati casuali
np.random.seed(0)  # Per la riproducibilità
X = np.random.rand(n_samples, n_variables)  # Variabili predittive
y = np.random.randint(0, 2, n_samples)  # Classi target (0 o 1)

# Creazione di un DataFrame
data = pd.DataFrame(X, columns=[f'variable_{i+1}' for i in range(n_variables)])
data['class'] = y  # Aggiunta della colonna della classe target

# Visualizzazione delle prime righe del DataFrame
print(data.head())

# Salvataggio in un file CSV
data.to_csv('synthetic_tumor_data.csv', index=False)
