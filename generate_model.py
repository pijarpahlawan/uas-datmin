import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
import os
import glob

file_path = 'https://gist.githubusercontent.com/MeylaniEP/52022bf121263fe05921e6b114a23612/raw/e2a28786419c69783d177709134f5eb9cd489f9a/online_gamedev_behavior_dataset.csv'
data = pd.read_csv(file_path)

# Menggunakan Label Encoding untuk kolom non-numerik
label_cols = ['EngagementLevel','Gender','GameGenre','GameDifficulty','Location']

for col in label_cols:
    unique_values = data[col].unique()

# Membuat salinan data untuk preprocessing
data_encoded = data.copy()
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    label_encoders[col] = le

# Memilih fitur untuk clustering
features = ['Age', 'Gender','GameGenre', 'GameDifficulty', 'AvgSessionDurationMinutes', 'Location']
X = data_encoded[features]

label = ['EngagementLevel']
y = [item for sublist in data_encoded[label].values.tolist() for item in sublist]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

models = {
    'SVM': SVC(decision_function_shape='ovr'),
    'Logistic Regression': LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

# Inisialisasi hasil
results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
highest_f1 = 0
trained_models = []

for model_name, model in models.items():
    print(f"Train {model_name}...")
    model.fit(X_train, y_train)  # Melatih model
    y_pred = model.predict(X_test)  # Melakukan prediksi

    # Menghitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Menyimpan hasil
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)

    # Membuat list model yang telah ditraining
    if highest_f1 < f1:
      highest_f1 = f1
    trained_models.append({
        'Model Name': model_name,
        'Model': model,
        'F1 Score': f1
    })
    
# Mengambil model dengan hasil terbaik
chosen_model = None
for model in trained_models:
  if model['F1 Score'] == highest_f1:
    chosen_model = model
    break

# Menghapus file .pkl yang telah ada
folder_path = 'dist'
files = glob.glob(os.path.join(folder_path, '*.pkl'))

for file in files:
   os.remove(file)
   print(f'File {file} has been removed')

# Menyimpan model terbaik
model_path = f'{folder_path}/{chosen_model["Model Name"]}.pkl'
joblib.dump(chosen_model['Model'], model_path)
