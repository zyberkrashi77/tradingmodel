import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

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
    
    # Convert to numpy array and reshape for LSTM (samples, timesteps, features)
    sequences = np.array(sequences)
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)  # Add feature dimension
    
    # Calculate statistics for all sequences at once
    stats = np.column_stack([
        np.mean(sequences, axis=1),
        np.max(sequences, axis=1),
        np.min(sequences, axis=1),
        np.std(sequences, axis=1),
        np.percentile(sequences, 25, axis=1),
        np.percentile(sequences, 75, axis=1),
        np.median(sequences, axis=1),
        np.mean(sequences >= 2.0, axis=1)  # ratio of values >= 2.0
    ])
    
    # Get targets
    y = (df['multiplier'].iloc[sequence_length:].values >= 2.0).astype(int)
    
    return sequences, stats, y

def build_model(sequence_shape, stats_shape):
    """Build the model for bust prediction"""
    # Sequence input branch
    sequence_input = Input(shape=sequence_shape)
    x = LSTM(128, return_sequences=True)(sequence_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Statistics input branch
    stats_input = Input(shape=stats_shape)
    y = Dense(32, activation='relu')(stats_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    
    # Combine branches
    combined = concatenate([x, y])
    
    # Final layers
    z = Dense(32, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='sigmoid')(z)
    
    # Create model
    model = Model(inputs=[sequence_input, stats_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    sequences, stats, y = load_and_preprocess_data('history_main.csv')
    
    # Split data
    X_seq_train, X_seq_test, X_stats_train, X_stats_test, y_train, y_test = train_test_split(
        sequences, stats, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    seq_scaler = MinMaxScaler()
    stats_scaler = MinMaxScaler()
    
    # Reshape sequences for scaling
    X_seq_train_reshaped = X_seq_train.reshape(-1, 1)  # Reshape to 2D for scaling
    X_seq_test_reshaped = X_seq_test.reshape(-1, 1)
    
    # Scale sequences and reshape back
    X_seq_train_scaled = seq_scaler.fit_transform(X_seq_train_reshaped).reshape(X_seq_train.shape)
    X_seq_test_scaled = seq_scaler.transform(X_seq_test_reshaped).reshape(X_seq_test.shape)
    
    # Scale stats
    X_stats_train_scaled = stats_scaler.fit_transform(X_stats_train)
    X_stats_test_scaled = stats_scaler.transform(X_stats_test)
    
    # Build model
    print("Building model...")
    model = build_model(X_seq_train.shape[1:], X_stats_train.shape[1:])
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train model
    print("Training model...")
    history = model.fit(
        [X_seq_train_scaled, X_stats_train_scaled], y_train,
        validation_data=([X_seq_test_scaled, X_stats_test_scaled], y_test),
        epochs=100,
        batch_size=64,
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(
        [X_seq_test_scaled, X_stats_test_scaled], y_test, verbose=0
    )
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Save model and scalers
    model.save('models/bust_prediction_model.h5')
    joblib.dump(seq_scaler, 'models/sequence_scaler.pkl')
    joblib.dump(stats_scaler, 'models/stats_scaler.pkl')
    print("\nModel and scalers saved successfully!")

if __name__ == "__main__":
    main() 