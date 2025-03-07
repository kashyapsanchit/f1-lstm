import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import fastf1
from joblib import dump

def get_fastf1_data(year, race_name, session_type='Race'):

    fastf1.Cache.enable_cache('cache')  
    
    session = fastf1.get_session(year, race_name, session_type)
    session.load(telemetry=False, weather=False)
    

    laps = session.laps
    
    relevant_cols = [
        'LapTime', 'LapNumber', 'Compound', 'TyreLife', 
        'TrackStatus', 'Stint', 'Driver', 'Team'
    ]
    
    df = laps[relevant_cols].copy()
    df = df[df['LapTime'].notna()]
    df = df[df['Compound'].notna()]
    
    df['LapTime'] = df['LapTime'].dt.total_seconds()
    
    df = pd.get_dummies(df, columns=['Compound', 'Team', 'Driver'])
    
    numerical_cols = ['LapTime', 'LapNumber', 'TyreLife', 'Stint']
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric)
    
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length, 0])  
    return np.array(X), np.array(y)

def preprocess_f1_data(year, race_name, sequence_length=10, test_size=0.2):
    
    df = get_fastf1_data(year, race_name)
    
    y_col = 'LapTime'
    features = [col for col in df.columns if col != y_col]
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    dump(scaler, 'model_artifact/scaler.joblib')
    
    X, y = create_sequences(scaled_features,df[y_col].values,sequence_length)
    
    split = int((1-test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler