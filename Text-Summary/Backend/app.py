from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

if __name__ == '__main__':
    app.run(debug=True)
