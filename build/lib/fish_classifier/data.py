import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_metadata(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_folds(metadata, fold_indices, label_key="class"):
    X_list = []
    y_list = []

    for i in fold_indices:
        X = metadata[i]["data"]
        y = metadata[i][label_key]

        X = X.reshape(-1, 75, 200, 3)
        y = np.repeat(y, 9)

        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


def prepare_data(path):
    metadata = load_metadata(path)

    X_train, y_train = load_folds(metadata, range(8))
    X_val, y_val = load_folds(metadata, [8])
    X_test, y_test = load_folds(metadata, [9])

    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))

    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test_enc,
        y_test,
        le,
    )