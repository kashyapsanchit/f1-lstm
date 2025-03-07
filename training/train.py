import mlflow
from preprocess import preprocess_f1_data
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dropout, Dense
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def train_model(year, race_name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.tensorflow.autolog()
    
    SEQ_LENGTH = 15
    EPOCHS = 100
    BATCH_SIZE = 32
    
    X_train, X_test, y_train, y_test, scaler = preprocess_f1_data(
        year, race_name, sequence_length=SEQ_LENGTH
    )
    
    train_size = int(len(X_train) * 0.85)
    X_train, X_val = X_train[:train_size], X_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]
    
    with mlflow.start_run():
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.4),  
            LSTM(32, return_sequences=False),  
            Dropout(0.3),  
            Dense(16, activation='relu'),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=0.00005)  
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        callbacks = [
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,  
                restore_best_weights=True,
                verbose=1
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        test_loss, test_mae = model.evaluate(X_test, y_test)
        mlflow.log_metrics({"test_loss": test_loss, "test_mae": test_mae})
        
        model.save("model_artifact/lstm_f1_model.keras")
        mlflow.log_artifact("model_artifact/lstm_f1_model.keras")

if __name__ == "__main__":
    train_model(2023, 'Spain')