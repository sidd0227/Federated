import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
