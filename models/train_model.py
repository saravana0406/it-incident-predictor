import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
df = pd.read_csv("../data/sample_logs.csv")

X = df[['cpu', 'memory', 'errors']]
y = df['label']

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ANN model
model = Sequential([
    Dense(32, activation='relu', input_dim=2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_pca, y, epochs=20)

# save everything
model.save("model.h5")
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))

print("Model trained and saved!")