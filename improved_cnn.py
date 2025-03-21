import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from urllib.parse import urlparse
import re
import tldextract

def create_url_features(url):
    """Create numerical features from URL for CNN processing."""
    # Initialize feature vector
    max_length = 256  # Maximum URL length to consider
    
    # Create character-level encoding
    char_to_int = {char: i+1 for i, char in enumerate(
        'abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&\'()*+,;=')}
    
    # Initialize feature arrays
    url_encoded = np.zeros(max_length)
    
    # Encode URL characters
    for i, char in enumerate(url.lower()[:max_length]):
        url_encoded[i] = char_to_int.get(char, 0)
    
    # Create additional features
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    
    # Additional numerical features
    additional_features = np.array([
        len(url) / max_length,  # Normalized URL length
        url.count('.') / len(url),  # Dot density
        url.count('-') / len(url),  # Hyphen density
        len(domain_info.domain) / len(url),  # Domain length ratio
        1 if parsed_url.scheme == 'https' else 0,  # HTTPS flag
        len(parsed_url.query) / len(url) if url else 0,  # Query length ratio
        1 if re.search(r'[^a-zA-Z0-9-.]', domain_info.domain) else 0,  # Special chars flag
        url.count('/') / len(url),  # Slash density
    ])
    
    return url_encoded, additional_features

def build_improved_cnn(input_shape, aux_shape):
    """Build an improved CNN model with attention mechanism and auxiliary features."""
    # URL sequence input
    url_input = layers.Input(shape=input_shape, name='url_input')
    
    # Embedding layer
    x = layers.Embedding(100, 64)(url_input)
    
    # Convolutional blocks with residual connections
    conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    
    conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    
    # Add residual connection
    conv2 = layers.Add()([layers.Conv1D(128, 1)(conv1), conv2])
    
    # Self-attention mechanism
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(conv2, conv2)
    attention = layers.BatchNormalization()(attention)
    attention = layers.Dropout(0.2)(attention)
    
    # Combine attention with convolutional features
    combined = layers.Concatenate()([conv2, attention])
    
    # Global pooling
    pooled = layers.GlobalAveragePooling1D()(combined)
    
    # Auxiliary feature input
    aux_input = layers.Input(shape=aux_shape, name='aux_input')
    
    # Combine pooled CNN features with auxiliary features
    merged = layers.Concatenate()([pooled, aux_input])
    
    # Dense layers with skip connections
    dense1 = layers.Dense(256, activation='relu')(merged)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(128, activation='relu')(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(0.3)(dense2)
    
    # Add skip connection
    dense2 = layers.Add()([layers.Dense(128)(dense1), dense2])
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(dense2)
    
    # Create model
    model = models.Model(inputs=[url_input, aux_input], outputs=output)
    
    return model

def train_improved_cnn(file_path):
    """Train an improved CNN model with advanced features and architecture."""
    # Load dataset
    print("Loading dataset...")
    df = pd.read_excel(file_path)
    
    # Prepare features
    print("Preparing features...")
    url_features = []
    aux_features = []
    
    for url in df['url']:
        url_encoded, additional_features = create_url_features(url)
        url_features.append(url_encoded)
        aux_features.append(additional_features)
    
    X_url = np.array(url_features)
    X_aux = np.array(aux_features)
    y = df['status'].map({'legitimate': 0, 'phishing': 1}).values
    
    # Split the data
    X_url_train, X_url_test, X_aux_train, X_aux_test, y_train, y_test = train_test_split(
        X_url, X_aux, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale auxiliary features
    aux_scaler = MinMaxScaler()
    X_aux_train = aux_scaler.fit_transform(X_aux_train)
    X_aux_test = aux_scaler.transform(X_aux_test)
    
    # Build model
    print("Building CNN model...")
    model = build_improved_cnn(X_url.shape[1], X_aux.shape[1])
    
    # Compile model with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        [X_url_train, X_aux_train],
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight={0: 1, 1: 1.5}  # Give more weight to phishing class
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate([X_url_test, X_aux_test], y_test)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    model.save('improved_cnn_model')
    joblib.dump(aux_scaler, 'improved_cnn_scaler.joblib')
    
    return model, aux_scaler, history

if __name__ == "__main__":
    file_path = "dataset/Dataset-2.xlsx"  # Update this path to match your dataset location
    print("Training improved CNN model...")
    model, scaler, history = train_improved_cnn(file_path)
    print("Training completed successfully!") 