import imaplib
import email
from email.header import decode_header
import csv
import re

import joblib
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm


# === YOUR CREDENTIALS ===
EMAIL_USER = "testhackathon41@gmail.com"
EMAIL_PASS = "fkjl spak nvsx gjkz"  # Use app password if 2FA is enabled
IMAP_SERVER = "imap.gmail.com"  # Change this for Outlook, Yahoo, etc.

# === Connect and Login ===
imap = imaplib.IMAP4_SSL(IMAP_SERVER)
imap.login(EMAIL_USER, EMAIL_PASS)

# === Select inbox ===
imap.select("inbox")

# === Search for UNSEEN (new) emails ===
status, messages = imap.search(None, 'UNSEEN')
email_ids = messages[0].split()

print(f"📬 Found {len(email_ids)} new emails.")

# === Prepare CSV file ===
with open("Data/emails.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sender", "DateTime", "Body"])  # Header

    for email_id in email_ids:
        # Fetch the email by ID
        status, msg_data = imap.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Extract sender
                sender = msg.get("From")
                sender_email = re.search(r'<(.+?)>', sender)
                sender = sender_email.group(1) if sender_email else sender

                # Extract date
                date = msg.get("Date")

                # Extract email body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_dispo = str(part.get("Content-Disposition"))

                        if content_type == "text/plain" and "attachment" not in content_dispo:
                            try:
                                body = part.get_payload(decode=True).decode()
                            except:
                                body = part.get_payload()
                            break
                else:
                    body = msg.get_payload(decode=True).decode()

                # Clean the body (optional)
                body = body.strip().replace('\r', '').replace('\n', ' ').replace(',', ';')

                # Write to CSV
                writer.writerow([sender, date, body[:1000]])  # Limit body to 1000 chars if long

# === Logout ===
imap.logout()

print("✅ Emails saved to 'emails.csv'")




nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tqdm.pandas()


# # Load saved model, vectorizer, and label encoders
# with open("models/model.pkl", "rb") as f:
#     model = pickle.load(f)
#
# with open("models/vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)
#
# with open("models/label_encoders.pkl", "rb") as f:
#     label_encoders = pickle.load(f)


model = joblib.load("models/model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def sort(group):
    path = 'Output/' + group + '.csv'
    with open(path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(output_df.columns)  # write header

        # Loop through the DataFrame rows
        for index, row in output_df.iterrows():
            if row['queue'] == group:
                writer.writerow(row)
                print('Entry added to the ',group,' Group.')

# Load email data
email_df = pd.read_csv('Data/emails.csv')
# email_df = email_df.dropna(subset=['body'])  # Drop rows with empty email body

# Clean the email bodies
email_df['clean_body'] = email_df['Body'].progress_apply(clean_text)

# Vectorize the cleaned emails
X_new = vectorizer.transform(email_df['clean_body'])

# Predict
predictions = model.predict(X_new)

# Decode predictions
decoded_predictions = []
for row in predictions:
    decoded_row = {}
    for i, col in enumerate(['type', 'queue', 'priority']):
        decoded_row[col] = label_encoders[col].inverse_transform([row[i]])[0]
    decoded_predictions.append(decoded_row)

# Create a DataFrame of predictions
pred_df = pd.DataFrame(decoded_predictions)

# Merge with original data (optional)
output_df = pd.concat([email_df[['Body']], pred_df], axis=1)

queues = output_df['queue'].tolist()

for i in queues:
    sort(i)