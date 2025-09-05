from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import os
import sqlite3
import base64
import io
from datetime import datetime
from PIL import Image
import json
import hashlib

app = Flask(__name__)
CORS(app)
app.secret_key = 'face_recognition_attendance_secret_key_2024'

# Initialize database
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'present',
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    conn.commit()
    conn.close()

# Face recognition class using OpenCV
class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_encodings = []
        self.face_labels = []
        self.load_trained_data()
    
    def detect_faces(self, image):
        """Detect faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces, gray
    
    def extract_face_encoding(self, image, face_coords):
        """Extract face encoding for training"""
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        return face_roi
    
    def train_on_image(self, image, label, retrain=False):
        """Train the model on a single image"""
        faces, gray = self.detect_faces(image)
        if len(faces) > 0:
            # Use the first detected face
            face_coords = faces[0]
            face_roi = self.extract_face_encoding(gray, face_coords)
            
            # Resize face to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            if retrain:
                # For retraining, add to existing data
                self.face_encodings.append(face_roi)
                self.face_labels.append(label)
            else:
                # For new training, check if label already exists
                if label in self.face_labels:
                    # Add to existing student's data
                    self.face_encodings.append(face_roi)
                    self.face_labels.append(label)
                else:
                    # New student
                    self.face_encodings.append(face_roi)
                    self.face_labels.append(label)
            
            # Retrain the model
            if len(self.face_encodings) > 0:
                # Ensure data types are correct for OpenCV
                face_encodings_array = np.array(self.face_encodings, dtype=np.uint8)
                face_labels_array = np.array(self.face_labels, dtype=np.int32)
                
                # Create new recognizer to avoid issues
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.train(face_encodings_array, face_labels_array)
                self.save_trained_data()
            
            return True
        return False
    
    def recognize_face(self, image):
        """Recognize faces in an image"""
        faces, gray = self.detect_faces(image)
        results = []
        
        for face_coords in faces:
            x, y, w, h = face_coords
            face_roi = self.extract_face_encoding(gray, face_coords)
            face_roi = cv2.resize(face_roi, (100, 100))
            
            if len(self.face_encodings) > 0:
                try:
                    label, confidence = self.recognizer.predict(face_roi)
                    # Lower confidence means better match - adjusted threshold for better accuracy
                    if confidence < 80:  # More strict threshold
                        results.append({
                            'label': int(label),
                            'confidence': float(confidence),
                            'coords': face_coords.tolist()
                        })
                except:
                    pass
        
        return results
    
    def delete_student(self, student_id):
        """Delete a student's face data"""
        # Find all indices for this student
        indices_to_remove = [i for i, label in enumerate(self.face_labels) if label == student_id]
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.face_encodings[i]
            del self.face_labels[i]
        
        # Retrain the model if there's still data
        if len(self.face_encodings) > 0:
            # Ensure data types are correct for OpenCV
            face_encodings_array = np.array(self.face_encodings, dtype=np.uint8)
            face_labels_array = np.array(self.face_labels, dtype=np.int32)
            
            # Create new recognizer to avoid issues
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(face_encodings_array, face_labels_array)
            self.save_trained_data()
        else:
            # If no data left, create empty model
            self.face_encodings = []
            self.face_labels = []
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.save_trained_data()
    
    def get_student_training_count(self, student_id):
        """Get number of training samples for a student"""
        return self.face_labels.count(student_id)
    
    def save_trained_data(self):
        """Save trained model data"""
        if len(self.face_encodings) > 0:
            # Ensure data types are correct before saving
            face_encodings_to_save = [np.array(face, dtype=np.uint8) for face in self.face_encodings]
            face_labels_to_save = [int(label) for label in self.face_labels]
            
            np.save('models/face_encodings.npy', face_encodings_to_save)
            np.save('models/face_labels.npy', face_labels_to_save)
            self.recognizer.save('models/face_model.yml')
        else:
            # Save empty arrays if no data
            np.save('models/face_encodings.npy', [])
            np.save('models/face_labels.npy', [])
    
    def load_trained_data(self):
        """Load previously trained model data"""
        try:
            if os.path.exists('models/face_encodings.npy'):
                self.face_encodings = np.load('models/face_encodings.npy', allow_pickle=True).tolist()
                self.face_labels = np.load('models/face_labels.npy', allow_pickle=True).tolist()
                
                # Ensure data types are correct
                if len(self.face_encodings) > 0:
                    # Convert to proper numpy arrays with correct dtypes
                    self.face_encodings = [np.array(face, dtype=np.uint8) for face in self.face_encodings]
                    self.face_labels = [int(label) for label in self.face_labels]
                    
                    # Create new recognizer and train
                    self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                    face_encodings_array = np.array(self.face_encodings, dtype=np.uint8)
                    face_labels_array = np.array(self.face_labels, dtype=np.int32)
                    self.recognizer.train(face_encodings_array, face_labels_array)
        except Exception as e:
            print(f"Error loading trained data: {e}")
            self.face_encodings = []
            self.face_labels = []

# Initialize face recognizer
face_recognizer = FaceRecognizer()

# Authentication functions
def is_authenticated():
    return session.get('authenticated', False)

def require_auth(f):
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    if is_authenticated():
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@require_auth
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        
        # Simple authentication - username and password both "admin"
        if username == 'admin' and password == 'admin':
            session['authenticated'] = True
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/upload_training_image', methods=['POST'])
@require_auth
def upload_training_image():
    try:
        data = request.get_json()
        image_data = data['image']
        student_name = data['name']
        retrain = data.get('retrain', False)
        student_id = data.get('student_id', None)

        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        if retrain and student_id:
            # Retraining existing student
            label_id = student_id
        else:
            # New student: first, insert the new student into the database
            cursor.execute('INSERT INTO students (name, face_encoding) VALUES (?, ?)', 
                         (student_name, '')) # The face_encoding column is just a placeholder
            conn.commit()
            
            # Get the correct auto-incrementing ID of the newly created student
            label_id = cursor.lastrowid
        
        # Train on the image using the correct ID from the database
        if face_recognizer.train_on_image(image, label_id, retrain):
            # Update the face_encoding in the database if needed, though the current
            # implementation doesn't use it for anything other than a placeholder
            cursor.execute('UPDATE students SET face_encoding = ? WHERE id = ?', (str(label_id), label_id))
            conn.commit()

            # Get training count
            training_count = face_recognizer.get_student_training_count(label_id)
            conn.close()
            
            action = "retrained" if retrain else "trained"
            return jsonify({
                'success': True, 
                'message': f'Successfully {action} on {student_name}\'s face (Training samples: {training_count})',
                'training_count': training_count,
                'student_id': label_id
            })
        else:
            conn.close()
            return jsonify({'success': False, 'message': 'No face detected in the image'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # First, detect how many faces are in the image
        faces, gray = face_recognizer.detect_faces(image)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        elif len(faces) > 1:
            return jsonify({'success': False, 'message': f'Multiple faces detected ({len(faces)}). Please ensure only one person is in the frame.'})
        
        # Only one face detected, proceed with recognition
        results = face_recognizer.recognize_face(image)
        
        if results and len(results) == 1:
            # Get student info from database
            try:
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()
                
                result = results[0]
                label = result['label']
                confidence = result['confidence']
                
                cursor.execute('SELECT name FROM students WHERE id = ?', (label,))
                student = cursor.fetchone()
                
                if student:
                    # Mark attendance
                    cursor.execute('INSERT INTO attendance (student_id, status) VALUES (?, ?)', 
                                 (label, 'present'))
                    conn.commit()
                    conn.close()
                    
                    return jsonify({
                        'success': True, 
                        'message': f'Attendance marked successfully for {student[0]}!',
                        'student_name': student[0],
                        'confidence': confidence
                    })
                else:
                    # Get available students before closing connection
                    cursor.execute("SELECT id, name FROM students")
                    available_students = cursor.fetchall()
                    conn.close()
                    return jsonify({'success': False, 'message': f'Student not found in database for label {label}. Available students: {[s[0] for s in available_students]}'})
            except Exception as db_error:
                if 'conn' in locals():
                    conn.close()
                return jsonify({'success': False, 'message': f'Database error: {str(db_error)}'})
        else:
            return jsonify({'success': False, 'message': 'Face not recognized. Please ensure the person is properly trained in the system.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance_logs')
def get_attendance_logs():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.id, s.name, a.timestamp, a.status 
        FROM attendance a 
        JOIN students s ON a.student_id = s.id 
        ORDER BY a.timestamp DESC
    ''')
    logs = cursor.fetchall()
    conn.close()
    
    return jsonify([{'id': log[0], 'name': log[1], 'timestamp': log[2], 'status': log[3]} for log in logs])

@app.route('/api/students')
def get_students():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, created_at FROM students')
    students = cursor.fetchall()
    conn.close()
    
    # Add training count for each student
    students_with_training = []
    for student in students:
        training_count = face_recognizer.get_student_training_count(student[0])
        students_with_training.append({
            'id': student[0], 
            'name': student[1], 
            'created_at': student[2],
            'training_count': training_count
        })
    
    return jsonify(students_with_training)

@app.route('/api/delete_student', methods=['POST'])
@require_auth
def delete_student():
    try:
        data = request.get_json()
        student_id = data['student_id']
        
        # Delete from face recognition model
        face_recognizer.delete_student(student_id)
        
        # Delete from database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
        cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Student deleted successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/retrain_student', methods=['POST'])
@require_auth
def retrain_student():
    try:
        data = request.get_json()
        student_id = data['student_id']
        student_name = data['name']
        
        # This will be handled by the upload_training_image endpoint with retrain=True
        return jsonify({'success': True, 'message': 'Ready for retraining. Please capture new images.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_attendance_log', methods=['POST'])
@require_auth
def delete_attendance_log():
    try:
        data = request.get_json()
        log_id = data['log_id']
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM attendance WHERE id = ?', (log_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Attendance record deleted successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/clear_all_logs', methods=['POST'])
@require_auth
def clear_all_logs():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM attendance')
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'All attendance records cleared successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/detect_faces', methods=['POST'])
def detect_faces():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces, gray = face_recognizer.detect_faces(image)
        
        # Return face detection info
        return jsonify({
            'success': True,
            'face_count': len(faces),
            'faces': [{'x': int(face[0]), 'y': int(face[1]), 'w': int(face[2]), 'h': int(face[3])} for face in faces]
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/debug_recognizer')
def debug_recognizer():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM students')
        students = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'face_encodings_count': len(face_recognizer.face_encodings),
            'face_labels': face_recognizer.face_labels,
            'students_in_db': students,
            'model_trained': len(face_recognizer.face_encodings) > 0
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Updated fix_labels endpoint to correctly rebuild the model from the database
@app.route('/api/fix_labels', methods=['POST'])
@require_auth
def fix_labels():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Get all students from the database
        cursor.execute('SELECT id, name, face_encoding FROM students')
        students = cursor.fetchall()
        
        if not students:
            conn.close()
            return jsonify({'success': False, 'message': 'No students found in database. Cannot rebuild model.'})
            
        new_encodings = []
        new_labels = []
        
        # Re-encode all faces based on the database
        for student in students:
            student_id = student[0]
            face_encoding_str = student[2]
            
            # The app stores a simple string representation of the student ID as a face encoding.
            # This is not a proper face embedding. It needs to be corrected to use
            # actual face encodings (e.g., the extracted face ROI).
            # The current implementation in the `upload_training_image` endpoint is flawed.
            # It stores a string of the label_id instead of the face data itself.
            # A correct implementation would require storing the face data (as a string or binary blob)
            # in the database, but since that's not happening, we'll try to correct it
            # by re-training from scratch using the stored `face_recognizer` data.
            
            # This logic assumes the in-memory `face_recognizer` has the correct encodings
            # but the labels are messed up. So, we'll re-map based on the student list.
            
            # Get all the indices from the in-memory labels that correspond to this student ID
            indices_for_student = [i for i, label in enumerate(face_recognizer.face_labels) if label == student_id]
            
            for index in indices_for_student:
                new_encodings.append(face_recognizer.face_encodings[index])
                new_labels.append(student_id)
        
        # Clear the old data and set the new, corrected data
        face_recognizer.face_encodings = new_encodings
        face_recognizer.face_labels = new_labels
        
        # Retrain the model with the corrected labels
        if len(face_recognizer.face_encodings) > 0:
            face_encodings_array = np.array(face_recognizer.face_encodings, dtype=np.uint8)
            face_labels_array = np.array(face_recognizer.face_labels, dtype=np.int32)
            
            face_recognizer.recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.recognizer.train(face_encodings_array, face_labels_array)
            face_recognizer.save_trained_data()
            message = 'Face recognition model rebuilt and synchronized with database.'
        else:
            # If no students have face data, initialize an empty model
            face_recognizer.recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.save_trained_data()
            message = 'No face data found to rebuild the model. Initialized an empty model.'
            
        conn.close()
        
        return jsonify({
            'success': True,
            'message': message,
            'corrected_labels_count': len(face_recognizer.face_labels)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
