import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnamed columns if any
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True, errors='ignore')

    # Rename target column if needed
    if 'cardio' in df.columns:
        df.rename(columns={'cardio': 'target'}, inplace=True)

    if 'target' not in df.columns:
        raise ValueError("Missing target column")

    # Encode target column
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    # === Feature Engineering: Combine features safely ===
    if 'chol' in df.columns and 'thalach' in df.columns:
        df['chol_thalach_ratio'] = df['chol'] / (df['thalach'] + 1)

    if 'oldpeak' in df.columns and 'cp' in df.columns:
        df['oldpeak_cp'] = df['oldpeak'] * df['cp']

    if 'age' in df.columns and 'trestbps' in df.columns:
        df['age_trestbps'] = df['age'] * df['trestbps']

    # === Feature Selection: Drop unwanted columns if necessary
    # Example (uncomment and modify as needed):
    # drop_cols = ['some_col1', 'some_col2']
    # df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df, le
