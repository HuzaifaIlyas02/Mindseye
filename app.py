### App.py ###
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pymongo
from qa_system import retriever, ask_question, ocr_physics_question
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET')  # Set JWT secret key from .env file
app.config['UPLOAD_FOLDER'] = 'uploads'
jwt = JWTManager(app)

# MongoDB setup
MONGODB_URI = os.getenv('MONGODB_URI')
client = pymongo.MongoClient(MONGODB_URI)
db = client.get_default_database()  # Get the default database from the URI
users_collection = db["users"]
conversations_collection = db["conversations"]

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400
    
    hashed_password = generate_password_hash(password)
    users_collection.insert_one({"email": email, "password": hashed_password})
    
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user['password'], password):
        access_token = create_access_token(identity=email)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

@app.route('/api/question', methods=['POST'])
@jwt_required()
def handle_question():
    current_user = get_jwt_identity()
    data = request.json
    question = data.get('question')
    
    user_conversations = conversations_collection.find_one({"user": current_user})
    conversation_history = user_conversations['history'] if user_conversations else []
    
    answer, flowchart = ask_question(retriever, question, conversation_history)
    
    conversations_collection.update_one(
        {"user": current_user},
        {"$push": {"history": {"question": question, "answer": answer}}},
        upsert=True
    )
    
    return jsonify({
        'answer': answer,
        'flowchart': flowchart
    })

@app.route('/api/upload-image', methods=['POST'])
@jwt_required()
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        question = ocr_physics_question(filepath)
        
        # Process the extracted question
        current_user = get_jwt_identity()
        user_conversations = conversations_collection.find_one({"user": current_user})
        conversation_history = user_conversations['history'] if user_conversations else []
        
        answer, flowchart = ask_question(retriever, question, conversation_history)
        
        conversations_collection.update_one(
            {"user": current_user},
            {"$push": {"history": {"question": question, "answer": answer}}},
            upsert=True
        )
        
        return jsonify({
            'question': question,
            'answer': answer,
            'flowchart': flowchart
        })

@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_history():
    current_user = get_jwt_identity()
    user_conversations = conversations_collection.find_one({"user": current_user})
    history = user_conversations['history'] if user_conversations else []
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)