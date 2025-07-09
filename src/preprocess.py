import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df['Priority_Label'] = df['Priority'].apply(lambda x: 1 if x in [1, 2] else 0)

    drop_cols = ['Priority', 'Incident_ID', 'Close_Time', 'Impact', 'Urgency', 'Open_Time', 'Reopen_Time']
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna("Unknown")

    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders