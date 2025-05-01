# --------------------------
# Imports
# --------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------
# Load and Preprocess Data
# --------------------------
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
    df.rename(columns={'prognosis': 'target'}, inplace=True)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
    return df, le

def split_data_for_clients(df, num_clients=6):
    client_dfs = []
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_size = len(df) // num_clients
    for i in range(num_clients):
        start = i * split_size
        end = start + split_size
        client_dfs.append(shuffled_df.iloc[start:end])
    return client_dfs

# --------------------------
# Define Model
# --------------------------
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------
# Train Local Model
# --------------------------
def train_local_model(X_train, y_train, input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
    return model.get_weights()

# --------------------------
# Federated Averaging
# --------------------------
def federated_averaging(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# --------------------------
# Main Federated Learning Process
# --------------------------
if __name__ == "__main__":
    file_path = 'Training.csv'
    test_file_path = 'Testing.csv'

    df, label_encoder = preprocess_data(file_path)
    test_df, _ = preprocess_data(test_file_path)

    client_dfs = split_data_for_clients(df, num_clients=5)

    input_shape = df.shape[1] - 1
    num_classes = len(df['target'].unique())

    local_weights = []

    for client_df in client_dfs:
        X = client_df.drop(columns=['target']).to_numpy()
        y = client_df['target'].to_numpy()
        weights = train_local_model(X, y, input_shape, num_classes)
        local_weights.append(weights)

    # Federated Averaging
    global_model = create_model(input_shape, num_classes)
    averaged_weights = federated_averaging(local_weights)
    global_model.set_weights(averaged_weights)

    # Evaluation
    X_test = test_df.drop(columns=['target']).to_numpy()
    y_test = test_df['target'].to_numpy()

    y_pred = np.argmax(global_model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"Global Model Accuracy on Testing Data: {acc * 100:.2f}%")
