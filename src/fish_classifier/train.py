from fish_classifier.data import prepare_data
from fish_classifier.model import build_model


def train_model(data_path):
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test_enc,
        y_test,
        le,
    ) = prepare_data(data_path)
    model = build_model(len(le.classes_))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
    )
    return model, history, le