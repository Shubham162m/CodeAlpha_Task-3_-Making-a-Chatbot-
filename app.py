# app.py
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = Flask(__name__)

# Load lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced Knowledge Base
knowledge_base = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
        "responses": [
            "Hello! Welcome to XYZ Corp. How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! I'm here to assist with any questions."
        ],
        "threshold": 0.7
    },

     "name": {
        "patterns": ["Name" , "Full Name" ,"What Is Your Name","what's your name ? ","what's your name","who are you ?","who are you"],
        "responses": [
            "SHUBHAM MODI" ,"MODI SHUBHAM"
        ],
        "threshold": 0.8
    },

     "Age": {
        "patterns": ["Age","age"],
        "responses": [
            "22"
        ],
        "threshold": 0.8
    },

    


    "products": {
        "patterns": [
            "what products do you offer",
            "your services",
            "what do you sell",
            "offerings",
            "product catalog"
        ],
        "responses": [
            "We specialize in three main areas:\n1. AI Solutions\n2. Web Development\n3. Business Consulting\nWhich area interests you?",
            "Our products include:\n- Enterprise AI platforms\n- Custom web applications\n- Digital transformation consulting"
        ],
        "threshold": 0.75
    },
    "contact": {
        "patterns": [
            "how to contact",
            "email",
            "phone number",
            "get in touch",
            "support",
                    ],
        "responses": [
            "You can reach us through:\n📧 Email:modishubham610@gmail.com\n📞 Phone: +91 92657 06957\n📍 Address: Gujarat,palanput ,pin 385421",
            "Our support team is available:\nMon-Fri: 9AM-5PM\nSat: 10AM-2PM\nCall us at 9898167275 \n"
        ],
        "threshold": 0.8
    },
    "hours": {
        "patterns": [
            "business hours",
            "when are you open",
            "operating hours",
            "working hours"
        ],
        "responses": [
            "Our standard business hours are:\nMonday-Friday: 9:00 AM - 5:00 PM\nSaturday: 10:00 AM - 2:00 PM",
            "We're open:\nWeekdays: 9AM-5PM\nSaturdays: 10AM-2PM\nClosed Sundays"
        ],
        "threshold": 0.75
    }
}

# Pre-compute embeddings
for intent in knowledge_base.values():
    intent["embeddings"] = [model.encode(pattern) for pattern in intent["patterns"]]

def get_response(user_input):
    user_embedding = model.encode(user_input.lower())
    best_match = None
    highest_score = 0
    
    for intent, data in knowledge_base.items():
        similarities = cosine_similarity([user_embedding], data["embeddings"])[0]
        max_similarity = np.max(similarities)
        
        if max_similarity > highest_score and max_similarity > data["threshold"]:
            highest_score = max_similarity
            best_match = intent
    
    if best_match:
        return {
            "response": np.random.choice(knowledge_base[best_match]["responses"]),
            "intent": best_match,
            "confidence": float(highest_score)
        }
    
    # Fallback responses
    fallbacks = [
        "I'm not sure I understand. Could you rephrase that?",
        "I didn't catch that. Try asking about our products, services, or contact information.",
        "That's an interesting question! For now, I can help with information about our company, products, or how to contact us."
    ]
    return {
        "response": np.random.choice(fallbacks),
        "intent": "unknown",
        "confidence": 0.0
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    try:
        result = get_response(user_message)
        return jsonify({
            "response": result["response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)