import os
import json
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, session
from sklearn.decomposition import PCA
import pickle
import hashlib
import joblib
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['DATASET_FOLDER'] = 'dataset'
app.config['JSON_FILE'] = os.path.join(app.config['DATASET_FOLDER'], 'face_data.json')
app.config['PCA_MODEL_FILE'] = os.path.join(app.config['DATASET_FOLDER'], 'pca_model.pickle')
app.config['SVM_MODEL_FILE'] = os.path.join(app.config['DATASET_FOLDER'], 'svm_model.pickle')

# Check and create dataset folder if it does not exist
if not os.path.exists(app.config['DATASET_FOLDER']):
    os.makedirs(app.config['DATASET_FOLDER'])

# Create an empty JSON file if it does not exist
if not os.path.exists(app.config['JSON_FILE']):
    with open(app.config['JSON_FILE'], 'w') as f:
        json.dump([], f)  # Initialize with an empty list

# Initialize Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Failed to load Haar Cascade.")

# Load SVM and PCA models
model_svm = pickle.load(open(app.config['SVM_MODEL_FILE'], 'rb'))
model_pca = joblib.load(app.config['PCA_MODEL_FILE'])

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Verify password
def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)

# Read user data from JSON
def read_user_data():
    with open(app.config['JSON_FILE'], 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
# Check and create dataset folder if it does not exist
if not os.path.exists(app.config['DATASET_FOLDER']):
    os.makedirs(app.config['DATASET_FOLDER'])

# Create an empty JSON file if it does not exist
app.config['JSON_FILE'] = os.path.join(app.config['DATASET_FOLDER'], 'face_data.json')
if not os.path.exists(app.config['JSON_FILE']):
    with open(app.config['JSON_FILE'], 'w') as f:
        json.dump([], f)  # Initialize with an empty list
    
# Save user data to JSON
def save_user_data(user_info):
    with open(app.config['JSON_FILE'], 'r+') as f:
        data = json.load(f)
        data.append(user_info)
        f.seek(0)
        json.dump(data, f, indent=4)

# Feature extraction from image
def extract_features(image_data):
    header, encoded = image_data.split(',', 1)
    binary_data = base64.b64decode(encoded)
    np_array = np.frombuffer(binary_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    
    # Resize image và chuẩn hoá kích thước đầu vào
    image = cv2.resize(image, (120, 90))  # Điều chỉnh kích thước theo PCA
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=3)
    
    if len(faces) == 0:
        print("No faces detected in the image.")  # Log for debugging
        return None

    # Chọn khuôn mặt đầu tiên nếu phát hiện nhiều
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]

        # Resize mặt để có kích thước chính xác
        face = cv2.resize(face, (120, 90)).flatten()  # Điều chỉnh kích thước cho PCA
        face = face / 255.0  # Normalize to [0, 1]
        
        # Đảm bảo mặt có kích thước đúng
        if len(face) == 10800:  # 120x90 = 10800
            return face

    print("No valid face found after resizing.")  # Log for debugging
    return None

def log_login_to_json(username):
    log_file_path = 'log.json'
    
    # Nếu tệp chưa tồn tại, khởi tạo danh sách rỗng
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            json.dump([], log_file)  # Tạo tệp JSON rỗng

    # Đọc dữ liệu hiện tại từ tệp JSON
    with open(log_file_path, 'r+') as log_file:
        try:
            logs = json.load(log_file)
        except json.JSONDecodeError:
            logs = []  # Nếu tệp rỗng hoặc lỗi, khởi tạo danh sách rỗng

        # Thêm thông tin đăng nhập mới
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.append({'username': username, 'timestamp': timestamp})

        # Ghi đè lại tệp JSON với dữ liệu mới
        log_file.seek(0)
        json.dump(logs, log_file, indent=4)
        log_file.truncate()  # Xóa nội dung cũ sau vị trí con trỏ

# Euclidean distance for similarity
def euclidean_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['username']
        age = request.form['age']
        birth_date = request.form['birth_date']
        occupation = request.form['occupation']
        student_id = request.form['student_id']
        password = request.form['password']
        gender = request.form['gender']

        # Store user info in session
        session['user_info'] = {
            'username': name,
            'age': age,
            'birth_date': birth_date,
            'occupation': occupation,
            'student_id': student_id,
            'password': hash_password(password),
            'gender': gender
        }
        
        # Redirect to capture image page
        return render_template('capture_image.html', username=name)
    return render_template('register.html')

@app.route('/save_user', methods=['POST'])
def save_user():
    user_info = session.get('user_info', {})
    image_data = request.form.get('image_data')

    if not image_data:
        return "Image data not found. Please try again."

    features = extract_features(image_data)
    if features is None:
        return "No face detected. Please try again."

    # Transform features with PCA
    features_pca = model_pca.transform([features])[0]
    user_info['face_features'] = features_pca.tolist()

    # Save user data in JSON
    save_user_data(user_info)
    return "User information saved successfully!"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get username and password from form
        username = request.form['username']
        password = request.form['password']

        # Verify username and password
        users = read_user_data()
        user = next((u for u in users if u['username'] == username), None)
        if not user or not verify_password(user['password'], password):
            return render_template('result.html', success=False, message="Invalid username or password.")

        # Process image data
        image_data = request.form.get('image_data')
        if not image_data:
            return render_template('result.html', success=False, message="No image data provided.")

        features = extract_features(image_data)
        if features is None:
            return render_template('result.html', success=False, message="No face detected in the image.")

        # Apply PCA transformation
        features_pca = model_pca.transform([features])[0]
        
        # Compute distance to stored features
        stored_features = np.array(user['face_features'])
        distance = euclidean_distance(features_pca, stored_features)

        # Login successful if similarity is at least 80%
        threshold = 0.8  # Adjust based on similarity requirements
        if distance <= threshold * np.linalg.norm(stored_features):
            log_login_to_json(username)  # Ghi lại thời gian đăng nhập của người dùng vào tệp log.json
            return render_template('result.html', success=True, user={
                'name': user['username'],
                'age': user['age'],
                'birth_date': user['birth_date'],
                'occupation': user['occupation'],
                'student_id': user['student_id'],
                'gender': user['gender']
            })

        return render_template('result.html', success=False, message="Face recognition failed.")

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)