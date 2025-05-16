import re
import joblib
from email.utils import parseaddr

class EmailClassifier:
    def __init__(self, model_file='email_classifier.pkl'):
        self.model_file = model_file
        self.model = joblib.load(model_file)
    
    @staticmethod
    def extract_email_features(email_text, has_attachment, attachment_format, sender_email):
        attachment_mapping = {"None": 0, "exe": 1, "apk": 2, "bat": 3, "cmd": 4, "scr": 5, "js": 6}
        attachment_encoded = attachment_mapping.get(attachment_format, 0)
        
        domain_mapping = {"gmail.com": 1, "yahoo.com": 2, "outlook.com": 3, "hotmail.com": 4, "unknown": -1}
        email_domain = parseaddr(sender_email)[1].split('@')[-1] if "@" in sender_email else "unknown"
        domain_encoded = domain_mapping.get(email_domain, -1)
        
        features = [
            len(email_text),
            len(email_text.split()),
            sum(not c.isalnum() and not c.isspace() for c in email_text),
            sum(c.isdigit() for c in email_text),
            sum(c.isupper() for c in email_text),
            int(bool(re.search(r'https?://\S+', email_text))),
            int(bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email_text))),
            sum(email_text.lower().count(word) for word in ["win", "free", "prize", "money", "credit", "offer", "click", "urgent"]),
            int(has_attachment),
            int(attachment_format in {"exe", "apk", "bat", "cmd", "scr", "js"}),
            1 if has_attachment else 0,
            len(re.findall(r'https?://\S+', email_text)),
            len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email_text)),
            len(sender_email),
            sum(email_text.lower().count(word) for word in ["password", "bank", "account", "verify", "login", "security"]),
            sum(not c.isalnum() and not c.isspace() for c in email_text) / max(len(email_text), 1),
            sum(c.isupper() for c in email_text) / max(sum(c.isalpha() for c in email_text), 1),
            sum(c.isdigit() for c in email_text) / max(len(email_text), 1),
            attachment_encoded,
            domain_encoded,
        ]
        return features
    
    def classify(self, email_text, has_attachment=False, attachment_format='None', sender_email='unknown@example.com'):
        features = self.extract_email_features(email_text, has_attachment, attachment_format, sender_email)
        prediction = self.model.predict([features])
        return prediction[0]