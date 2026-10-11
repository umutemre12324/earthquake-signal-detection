import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    X = []
    y = []

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if filename.startswith("earthquake_") and filename.endswith(".npy") and not filename.startswith("earthquake_noise"):
            X.append(np.load(path))
            y.append(1)

        elif filename.startswith("noise_") and filename.endswith(".npy"):
            X.append(np.load(path))
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def preprocess_data():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    os.makedirs("processed_data", exist_ok=True)

    np.save("processed_data/X_train.npy", X_train)
    np.save("processed_data/X_test.npy", X_test)
    np.save("processed_data/y_train.npy", y_train)
    np.save("processed_data/y_test.npy", y_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print("Preprocessing completed!")
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    print("Processed arrays saved in processed_data/")

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = preprocess_data()