import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_utils import preprocess_data, split_data_for_clients
from src.model_utils import create_model, train_local_model
from src.federated_learning import federated_averaging

def train_and_save(file_path, test_file_path, num_clients=5):
    df, label_encoder = preprocess_data(file_path)
    test_df, _ = preprocess_data(test_file_path)

    client_dfs = split_data_for_clients(df, num_clients=num_clients)

    input_shape = df.shape[1] - 1
    num_classes = len(df['target'].unique())

    local_weights = []
    client_accuracies = {}

    for i, client_df in enumerate(client_dfs):
        train_df, val_df = train_test_split(client_df, test_size=0.2, random_state=42)
        X_train = train_df.drop(columns=['target']).to_numpy()
        y_train = train_df['target'].to_numpy()
        X_val = val_df.drop(columns=['target']).to_numpy()
        y_val = val_df['target'].to_numpy()

        weights, acc = train_local_model(X_train, y_train, X_val, y_val, input_shape, num_classes)
        local_weights.append(weights)
        client_accuracies[f"Client {i+1}"] = round(acc * 100, 2)

    global_model = create_model(input_shape, num_classes)
    averaged_weights = federated_averaging(local_weights)
    global_model.set_weights(averaged_weights)

    X_test = test_df.drop(columns=['target']).to_numpy()
    y_test = test_df['target'].to_numpy()
    y_pred = np.argmax(global_model.predict(X_test), axis=1)
    global_acc = accuracy_score(y_test, y_pred)

    os.makedirs('models', exist_ok=True)
    global_model.save('models/global_model.h5')

    os.makedirs('results', exist_ok=True)
    with open('results/accuracy.json', 'w') as f:
        json.dump({
            'global_accuracy': global_acc,
            'client_accuracies': client_accuracies
        }, f)

    print(f"✅ Global Model Accuracy: {global_acc * 100:.2f}%")
    print(f"✅ Client Accuracies: {client_accuracies}")
    print("✅ Training complete!")

if __name__ == "__main__":
    train_and_save('data/Training.csv', 'data/Testing.csv', num_clients=5)
