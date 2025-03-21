import pickle
import os
import tensorflow as tf
from tensorflow import keras

def load_and_save_model(pickle_path, h5_path):
    try:
        # Load the pickled model
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
            
        # Save as Keras model
        model.save(h5_path)
        print(f"Successfully converted {pickle_path} to {h5_path}")
    except Exception as e:
        print(f"Error converting {pickle_path}: {str(e)}")

def main():
    # Get the model directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    # Convert each model
    models = ['cnn', 'gru', 'hybrid']
    for model_name in models:
        pickle_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        h5_path = os.path.join(model_dir, f"{model_name}_model.h5")
        
        if os.path.exists(pickle_path):
            print(f"\nConverting {model_name} model...")
            load_and_save_model(pickle_path, h5_path)
        else:
            print(f"\n{model_name} model pickle file not found")

if __name__ == "__main__":
    main() 