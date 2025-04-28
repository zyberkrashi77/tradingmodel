import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def create_sequence_features(sequence):
    """Create features for each sequence"""
    features = []
    for seq in sequence:
        # Original values
        original = seq.reshape(-1)
        
        # Current trend (last 3 values)
        last_3 = original[-3:]
        trend = np.polyfit(range(3), last_3, 1)[0]
        
        # Current value and its position relative to recent history
        current = original[-1]
        recent_mean = np.mean(original[-5:])
        position = (current - recent_mean) / (np.max(original[-5:]) - np.min(original[-5:]) + 1e-8)
        
        # Combine features
        combined = np.column_stack([
            original[:-1],  # Historical values
            np.full(len(original)-1, trend),  # Current trend
            np.full(len(original)-1, position)  # Current position
        ])
        
        features.append(combined)
    
    return np.array(features)

def load_and_preprocess_data(file_path, sequence_length=40):
    """Load and preprocess the bust data"""
    # Read the data and convert to numeric, coercing errors to NaN
    df = pd.read_csv(file_path, header=None, names=['multiplier'], low_memory=False)
    df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Convert to float type
    df['multiplier'] = df['multiplier'].astype(float)
    
    # Create sequences using rolling window
    sequences = []
    for i in range(len(df) - sequence_length):
        sequences.append(df['multiplier'].iloc[i:i+sequence_length].values)
    
    # Convert to numpy array
    sequences = np.array(sequences)
    
    # Create features
    X = create_sequence_features(sequences)
    
    # Get targets - the actual multiplier values
    y = df['multiplier'].iloc[sequence_length:].values
    
    return X, y

def build_model(input_shape):
    """Build a regression model for multiplier prediction"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid activation to constrain predictions
    ])
    
    # Compile model with gradient clipping and lower learning rate
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('history_main.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features and targets
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    
    # Scale and reshape back
    X_train_scaled = X_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Scale targets to [0,1] range
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    # Build model
    print("Building model...")
    model = build_model(X_train.shape[1:])
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Define early stopping with more patience
    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model with more epochs
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=100,
        batch_size=32,  # Smaller batch size
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    loss, mae = model.evaluate(
        X_test_scaled, y_test_scaled, verbose=0
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    # Calculate bust prediction metrics
    y_test_bust = (y_test >= 2.0).astype(int)
    y_pred_bust = (y_pred >= 2.0).astype(int)
    
    # Debug information
    print("\nDebug Information:")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Number of actual busts: {np.sum(y_test_bust)}")
    print(f"Number of predicted busts: {np.sum(y_pred_bust)}")
    print(f"Average predicted value: {np.mean(y_pred):.4f}")
    print(f"Min predicted value: {np.min(y_pred):.4f}")
    print(f"Max predicted value: {np.max(y_pred):.4f}")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_bust, y_pred_bust)
    precision = precision_score(y_test_bust, y_pred_bust)
    recall = recall_score(y_test_bust, y_pred_bust)
    f1 = f1_score(y_test_bust, y_pred_bust)
    cm = confusion_matrix(y_test_bust, y_pred_bust)
    
    print("\nRegression Metrics:")
    print(f"Test MSE: {loss:.6f}")
    print(f"Test MAE: {mae:.6f}")
    
    print("\nBust Prediction Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model and scalers
    model.save('models/bust_prediction_model_regression.h5')
    joblib.dump(X_scaler, 'models/sequence_scaler_regression.pkl')
    joblib.dump(y_scaler, 'models/target_scaler_regression.pkl')
    print("\nModel and scalers saved successfully!")

if __name__ == "__main__":
    main() 