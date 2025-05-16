import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_utils import preprocess_data  # Make sure src is in your PYTHONPATH or project folder

def prepare_and_save_data(original_file_path, train_path, test_path, test_size=0.2):
    # Preprocess data (returns dataframe and label encoder)
    df, _ = preprocess_data(original_file_path)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Save the train and test data to CSV files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Saved training data to {train_path} with shape {train_df.shape}")
    print(f"✅ Saved testing data to {test_path} with shape {test_df.shape}")

if __name__ == "__main__":
    prepare_and_save_data(
        original_file_path="data/cardio_train.csv",
        train_path="data/Training.csv",
        test_path="data/Testing.csv"
    )
