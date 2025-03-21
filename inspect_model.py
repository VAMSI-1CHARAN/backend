import pickle
import os
import numpy as np

def inspect_model(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        print("\nModel type:", type(model))
        print("\nModel attributes:", dir(model))
        if hasattr(model, 'get_config'):
            print("\nModel config:", model.get_config())
        print("\nModel summary:")
        if hasattr(model, 'summary'):
            model.summary()
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    # Inspect CNN model
    cnn_path = os.path.join(model_dir, "cnn_model.pkl")
    print("\nInspecting CNN model...")
    inspect_model(cnn_path) 