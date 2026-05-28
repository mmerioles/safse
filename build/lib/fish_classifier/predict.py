import numpy as np


def predict_labels(model, X, label_encoder):
    probs = model.predict(X)
    pred_classes = np.argmax(probs, axis=1)
    return label_encoder.inverse_transform(pred_classes)