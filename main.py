from flask import Flask, request, jsonify, render_template
import whisper
import os
from werkzeug.utils import secure_filename
import subprocess
import sys
from datetime import datetime

app = Flask(__name__)
app.debug = True

# Add FFmpeg to PATH explicitly
FFMPEG_PATH = r"C:\ffmpeg"
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Configure upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
print(f"Upload directory path: {UPLOADS_DIR}")

def create_response(success=True, data=None, error=None, status_code=200):
    """Create a standardized JSON response"""
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data if data else {},
        "error": {
            "message": str(error) if error else None,
            "code": status_code if not success else None
        }
    }
    return jsonify(response), status_code

# Check FFmpeg installation
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print("FFmpeg found:", result.stdout.split('\n')[0])
        return True
    except Exception as e:
        print(f"FFmpeg check failed: {str(e)}")
        return False

print("Checking FFmpeg installation...")
if not check_ffmpeg():
    print("WARNING: FFmpeg not found in PATH")
    sys.exit(1)

# Initialize Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("\n=== Starting Upload Process ===")
        
        if 'file' not in request.files:
            print("Error: No file in request.files")
            return create_response(
                success=False,
                error="No file provided",
                status_code=400
            )
        
        file = request.files['file']
        if file.filename == '':
            print("Error: Empty filename")
            return create_response(
                success=False,
                error="No selected file",
                status_code=400
            )
        
        if file:
            # Create a unique filename using timestamp
            import time
            timestamp = int(time.time())
            original_filename = file.filename
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"Original filename: {original_filename}")
            print(f"Saving as: {filename}")
            print(f"Full filepath: {filepath}")
            
            try:
                # Ensure directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save file and verify
                file.save(filepath)
                
                if not os.path.exists(filepath):
                    raise Exception("File was not saved properly")
                
                file_size = os.path.getsize(filepath)
                print(f"File saved successfully. Size: {file_size/1024/1024:.2f} MB")
                print(f"File exists at path: {os.path.exists(filepath)}")
                
                # Print current working directory and PATH
                print(f"Current working directory: {os.getcwd()}")
                print(f"PATH: {os.environ['PATH']}")
                
                # Verify audio file can be opened
                try:
                    subprocess.run(['ffmpeg', '-i', filepath, '-f', 'null', '-'], 
                                 capture_output=True, 
                                 check=True)
                    print("Audio file validity check passed")
                except subprocess.CalledProcessError as e:
                    print(f"Audio file check failed: {str(e)}")
                    return create_response(
                        success=False,
                        error="Invalid audio file format",
                        status_code=400
                    )
                
                # Transcribe
                print("Starting transcription...")
                result = model.transcribe(filepath)
                transcription = result["text"]
                print("Transcription completed:", transcription[:100] + "...")
                
                # Prepare response data
                response_data = {
                    "transcription": transcription,
                    "metadata": {
                        "filename": original_filename,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "processing_timestamp": timestamp,
                        "language": result.get("language", "unknown"),
                        "duration_seconds": result.get("duration", 0)
                    }
                }
                
                # Cleanup
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print("File cleaned up")
                
                return create_response(
                    success=True,
                    data=response_data
                )
                
            except Exception as e:
                import traceback
                print("Error occurred:")
                print(traceback.format_exc())
                
                # Cleanup on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print("Cleaned up file after error")
                
                return create_response(
                    success=False,
                    error=f"Transcription failed: {str(e)}",
                    status_code=500
                )
                
    except Exception as e:
        print("Upload error:", str(e))
        return create_response(
            success=False,
            error=f"Server error: {str(e)}",
            status_code=500
        )

if __name__ == '__main__':
    # Add host and port explicitly
    print("Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=True)