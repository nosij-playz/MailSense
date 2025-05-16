from flask import Flask, request, jsonify, render_template
from emailcheck import EmailClassifier

app = Flask(__name__)
classifier = EmailClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.json
    email_text = data.get("email_text", "")
    has_attachment = data.get("has_attachment", False)
    attachment_format = data.get("attachment_format", "")
    sender_email = data.get("sender_email", "")
    
    category = classifier.classify(email_text, has_attachment, attachment_format, sender_email)
    
    return jsonify({"predicted_category": category})

if __name__ == '__main__':
    app.run(debug=True)
