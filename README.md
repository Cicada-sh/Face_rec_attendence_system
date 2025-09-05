# Face Recognition Attendance System

A simple and effective face recognition attendance system that allows you to train the model with your own images and mark attendance in real-time using your webcam.

## Features

- **Live Training**: Train the system with your own face using your webcam
- **Real-time Recognition**: Mark attendance by simply showing your face to the camera
- **Dashboard**: View attendance logs and student statistics
- **Simple Interface**: Easy-to-use web interface with modern design
- **No Dummy Data**: Uses your actual face for training and recognition

## Setup Instructions

### 1. Activate the Virtual Environment

```bash
conda activate face_recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## How to Use

### Training Mode
1. Open the application in your browser
2. Go to the "Training Mode" tab
3. Enter your name
4. Click "Start Camera" to activate your webcam
5. Position your face in the frame
6. Click "Capture & Train" to train the system on your face
7. Repeat this process with different angles/lighting for better accuracy

### Attendance Mode
1. Switch to the "Attendance Mode" tab
2. Click "Start Camera" to activate your webcam
3. Position your face in the frame
4. Click "Mark Attendance" to record your attendance
5. The system will recognize your face and mark you present

### Dashboard
1. Click "View Dashboard" to see attendance statistics
2. View all registered students
3. Check attendance logs with timestamps
4. Monitor daily attendance counts

## Technical Details

- **Backend**: Flask with SQLite database
- **Face Recognition**: OpenCV with LBPH (Local Binary Patterns Histograms) face recognizer
- **Frontend**: HTML5, CSS3, JavaScript with webcam integration
- **Database**: SQLite for storing student data and attendance records

## File Structure

```
Face_rec_hackathon/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
│   ├── index.html        # Main interface
│   └── dashboard.html    # Dashboard interface
├── uploads/              # Uploaded images (if any)
├── models/               # Trained face recognition models
└── attendance.db         # SQLite database (created automatically)
```

## API Endpoints

- `GET /` - Main application interface
- `GET /dashboard` - Attendance dashboard
- `POST /api/upload_training_image` - Train on a new face
- `POST /api/mark_attendance` - Mark attendance for recognized faces
- `GET /api/attendance_logs` - Get attendance records
- `GET /api/students` - Get registered students

## Troubleshooting

### Camera Issues
- Make sure your webcam is connected and not being used by other applications
- Grant camera permissions when prompted by your browser
- Try refreshing the page if the camera doesn't start

### Recognition Issues
- Train with multiple images at different angles and lighting conditions
- Ensure good lighting when training and marking attendance
- Make sure your face is clearly visible and centered in the frame

### Performance
- The system works best with good lighting conditions
- For better accuracy, train with 3-5 different images of the same person
- The face recognition model improves with more training data

## Demo Workflow

1. **Setup**: Run the application and open in browser
2. **Training**: Use "Training Mode" to register your face
3. **Testing**: Switch to "Attendance Mode" and test recognition
4. **Monitoring**: Check the dashboard to see attendance records
5. **Live Demo**: Show real-time face detection and attendance marking

## Notes

- The system uses OpenCV's built-in face recognition which is lightweight and doesn't require external dependencies like dlib
- All data is stored locally in SQLite database
- The face recognition model is saved in the `models/` directory
- The system is designed to be simple and educational, perfect for demonstrations
