import pickle
import os
import numpy as np
import tensorflow as tf

def extract_weights(pickle_path, save_path):
    try:
        # Load the pickled model
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get the weights
        weights = model.get_weights()
        
        # Create a temporary model with the same architecture
        if 'cnn' in pickle_path:
            temp_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        elif 'gru' in pickle_path:
            temp_model = tf.keras.Sequential([
                tf.keras.layers.Reshape((13, 1), input_shape=(13,)),
                tf.keras.layers.GRU(64, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GRU(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:  # hybrid
            temp_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(13,)),
                tf.keras.layers.Reshape((128, 1)),
                tf.keras.layers.GRU(64, return_sequences=True),
                tf.keras.layers.Conv1D(32, 3, activation='relu'),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        
        # Compile the model
        temp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Set the weights
        temp_model.set_weights(weights)
        
        # Save the weights
        temp_model.save_weights(save_path)
        print(f"Successfully extracted weights from {pickle_path} and saved to {save_path}")
        
    except Exception as e:
        print(f"Error processing {pickle_path}: {str(e)}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    models = ['cnn', 'gru', 'hybrid']
    for model_name in models:
        pickle_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        weights_path = os.path.join(model_dir, f"{model_name}_model_weights.h5")
        
        if os.path.exists(pickle_path):
            print(f"\nProcessing {model_name} model...")
            extract_weights(pickle_path, weights_path)
        else:
            print(f"\n{model_name} model pickle file not found")

if __name__ == "__main__":
    main() 