import joblib
import os

def save_model(model, encoders, model_path="model"):
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, "random_forest_model.joblib"))
    joblib.dump(encoders, os.path.join(model_path, "label_encoders.joblib"))

def load_model(model_path="model"):
    model = joblib.load(os.path.join(model_path, "random_forest_model.joblib"))
    encoders = joblib.load(os.path.join(model_path, "label_encoders.joblib"))
    return model, encoders
