import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib
import os

# Load dataset
df = pd.read_csv('Data/SampleDataset.csv')

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df = df.dropna(subset=['body'])

# Enable tqdm for pandas apply
tqdm.pandas()

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Clean the email body
df['clean_body'] = df['body'].progress_apply(clean_text)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_body'])

# Encode the labels
label_encoders = {}
y = pd.DataFrame()
for col in ['type', 'queue', 'priority']:
    le = LabelEncoder()
    y[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
target_columns = ['type', 'queue', 'priority']
for i, col in enumerate(target_columns):
    print(f"\nClassification Report for: {col}")
    print(classification_report(y_test[col], y_pred[:, i]))

# Example test input
new_text = "Please check the database issue and respond urgently"
cleaned = clean_text(new_text)
vectorized = vectorizer.transform([cleaned])
pred = model.predict(vectorized)

for i, col in enumerate(['type', 'queue', 'priority']):
    print(f"{col}: {label_encoders[col].inverse_transform([pred[0][i]])[0]}")


#TO save the model
# import pickle
#
# os.makedirs("Models", exist_ok=True)
#
# with open("Models/model.pkl", "wb") as f:
#     pickle.dump(model, f)
#
# with open("Models/vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)
#
# with open("Models/label_encoders.pkl", "wb") as f:
#     pickle.dump(label_encoders, f)
