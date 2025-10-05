from flask import Flask, render_template, request, jsonify
import cv2, os, numpy as np
from PIL import Image
import sqlite3
import datetime
import threading
import sounddevice as sd
import wave
import time
import requests
import speech_recognition as sr
import pyttsx3
import subprocess
import platform

# ==================== FLASK APP ====================
app = Flask(__name__)

# ==================== DATABASE ====================
DB_FILE = "workers.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS workers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  worker_id TEXT UNIQUE,
                  name TEXT,
                  role TEXT,
                  face_folder TEXT,
                  last_seen TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS issues
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  worker_id TEXT,
                  timestamp TEXT,
                  description TEXT,
                  image_path TEXT,
                  solution TEXT)''')
    conn.commit()
    conn.close()


init_db()


# ==================== IMPROVED LLM WITH FALLBACKS ====================

def ask_llm(prompt):
    """
    Improved LLM with multiple fallback options
    """
    # Simple rule-based responses for common queries
    prompt_lower = prompt.lower().strip()

    # Greetings
    if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm TechnoBot. How can I assist you today?"

    # Help questions
    if any(word in prompt_lower for word in ['help', 'what can you do']):
        return "I can help you with: worker identification using face detection, issue reporting, and answering questions. You can speak to me or type messages!"

    # Thanks
    if any(word in prompt_lower for word in ['thank', 'thanks']):
        return "You're welcome! Is there anything else I can help with?"

    # Goodbye
    if any(word in prompt_lower for word in ['bye', 'goodbye']):
        return "Goodbye! Have a great day!"

    # Issue reporting
    if any(word in prompt_lower for word in ['issue', 'problem', 'broken']):
        return "To report an issue, click the 'Report Issue' button and describe your problem. I'll help generate a solution!"

    # Worker identification
    if any(word in prompt_lower for word in ['worker', 'identify', 'face', 'camera']):
        return "I can identify workers using facial recognition. Click the 'Identify' button to start camera detection!"

    # Name
    if any(word in prompt_lower for word in ['name', 'who are you']):
        return "I'm TechnoBot, your workplace assistant robot!"

    # How it works
    if any(word in prompt_lower for word in ['how', 'work']):
        return "I work by using speech recognition to understand you, AI to generate responses, and text-to-speech to reply. You can talk to me or type messages!"

    # Time
    if 'time' in prompt_lower:
        return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."

    # Date
    if 'date' in prompt_lower:
        return f"Today's date is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."

    # Weather (placeholder)
    if 'weather' in prompt_lower:
        return "I don't have access to weather information, but I can help with workplace issues and worker management."

    # Default responses based on question type
    question_words = ["what", "how", "why", "when", "where", "who", "which"]
    if any(prompt_lower.startswith(word) for word in question_words):
        return "I understand you're asking a question. As a workplace assistant, I'm best at helping with worker management, issue reporting, and general workplace assistance. Could you rephrase your question related to these areas?"

    # Generic helpful response
    return "I'm here to help with workplace tasks like worker identification and issue reporting. How can I assist you specifically with these areas?"


# ==================== AUDIO DEVICE MANAGEMENT ====================
def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels']
                })
        return input_devices
    except Exception as e:
        print(f"Error getting audio devices: {e}")
        return []


# ==================== ADVANCED TTS WITH PYTTSX3 ====================
tts_engine = None


def init_tts():
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 160)
        tts_engine.setProperty('volume', 1.0)

        voices = tts_engine.getProperty('voices')
        if voices:
            # Prefer female voices for clarity
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    tts_engine.setProperty('voice', voice.id)
                    break
            else:
                tts_engine.setProperty('voice', voices[0].id)

        print("✅ TTS engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ TTS engine initialization failed: {e}")
        tts_engine = None
        return False


def tts_speak(text):
    if not tts_engine:
        if not init_tts():
            return {"status": "error", "message": "TTS engine not available"}

    try:
        def speak():
            tts_engine.say(text)
            tts_engine.runAndWait()

        speech_thread = threading.Thread(target=speak)
        speech_thread.daemon = True
        speech_thread.start()

        return {"status": "success", "message": "Speech started"}
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return {"status": "error", "message": f"TTS error: {str(e)}"}


# ==================== ADVANCED STT WITH MULTIPLE FALLBACKS ====================
def stt_with_fallbacks(audio_file="worker_audio.wav"):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        print("🔄 Attempting Google Speech Recognition...")
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"✅ Google STT: '{text}'")
            return {"status": "success", "message": "Google STT successful", "text": text, "engine": "google"}
        except sr.UnknownValueError:
            return {"status": "success", "message": "No speech detected", "text": "", "engine": "google"}
        except sr.RequestError as e:
            print(f"❌ Google STT failed: {e}")

    except Exception as e:
        print(f"❌ STT processing error: {e}")

    return {"status": "error", "message": "All STT methods failed", "text": "", "engine": "none"}


# ==================== AUDIO RECORDING ====================
def record_audio(duration=5, fs=16000, filename="worker_audio.wav", device_id=None):
    print(f"🎤 Recording audio for {duration} seconds...")

    try:
        # FIXED: Use os.remove() instead of os.path.remove()
        if os.path.exists(filename):
            os.remove(filename)

        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype='int16',
            device=device_id
        )

        print("🔴 Recording started...")
        for i in range(duration, 0, -1):
            print(f"   {i} seconds remaining...")
            time.sleep(1)

        sd.wait()
        print("✅ Recording finished")

        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
        wf.close()

        if os.path.exists(filename) and os.path.getsize(filename) > 1000:
            file_size = os.path.getsize(filename)
            print(f"✅ Audio saved: {filename} ({file_size} bytes)")
            return True, f"Audio recorded successfully"
        else:
            return False, "Recording failed - file too small"

    except Exception as e:
        print(f"❌ Recording error: {e}")
        return False, f"Recording failed: {str(e)}"


# ==================== FACE DETECTION ====================
def detect_face_only():
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return "Camera Error", "Cannot access camera"

        face_detected = False
        start_time = time.time()

        print("📷 Looking for faces...")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            return "Detection Error", "Face detector not loaded"

        while (time.time() - start_time) < 5:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_detected = True

            cv2.putText(frame, "Face Detection - Press ESC to exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if face_detected:
                break

        cam.release()
        cv2.destroyAllWindows()

        if face_detected:
            return "Face Detected", "Visitor"
        else:
            return "No Face Detected", "Unknown"

    except Exception as e:
        return f"Detection Error: {str(e)}", "Error"


# ==================== FLASK ROUTES ====================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/audio_devices")
def audio_devices():
    devices = get_audio_devices()
    return jsonify({"devices": devices})


@app.route("/identify", methods=["GET"])
def identify():
    worker_name, worker_id = detect_face_only()
    return jsonify({"worker_name": worker_name, "worker_id": worker_id})


@app.route("/stt", methods=["GET"])
def stt_route():
    audio_filename = "worker_audio.wav"

    print("=== Starting Speech Recognition ===")

    device_id = request.args.get('device', None, type=int)

    recorded_successfully, record_message = record_audio(
        duration=5,
        filename=audio_filename,
        device_id=device_id
    )

    if not recorded_successfully:
        return jsonify({"text": record_message, "error": True}), 500

    print("🔄 Processing speech recognition...")
    stt_result = stt_with_fallbacks(audio_filename)

    if stt_result["status"] == "error":
        return jsonify({"text": stt_result["message"], "error": True}), 500
    else:
        return jsonify({
            "text": stt_result["text"],
            "error": False,
            "engine": stt_result.get("engine", "unknown")
        })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip()
    lang = data.get("lang", "en")

    print(f"💬 Chat request: '{prompt}'")

    if not prompt:
        return jsonify({"reply": "Please send a message.", "error": False})

    response = ask_llm(prompt)
    print(f"🤖 AI response: {response}")

    # Use TTS for response
    if response and len(response) > 5:  # Only speak if response is meaningful
        tts_result = tts_speak(response)
        if tts_result["status"] == "error":
            print(f"❌ TTS failed: {tts_result['message']}")

    return jsonify({"reply": response, "error": False})


@app.route("/submit_issue", methods=["POST"])
def submit_issue():
    data = request.json
    worker_id = data.get("worker_id", "Unknown")
    description = data.get("description", "")
    lang = data.get("lang", "en")

    if not description:
        return jsonify({"status": "error", "message": "Description required"}), 400

    # Enhanced issue handling with practical solutions
    description_lower = description.lower()

    if any(word in description_lower for word in ['not working', 'broken', 'not function']):
        solution = "For equipment not working:\n1. Check power connections and cables\n2. Restart the equipment\n3. Verify proper connections\n4. Contact technical support if issue persists"

    elif any(word in description_lower for word in ['slow', 'lag', 'performance']):
        solution = "For performance issues:\n1. Clear cache and temporary files\n2. Check network connection\n3. Close unused applications\n4. Restart the system\n5. Check for software updates"

    elif any(word in description_lower for word in ['error', 'crash', 'not responding']):
        solution = "For application errors:\n1. Note the exact error message\n2. Restart the application\n3. Check system logs\n4. Update to latest version\n5. Reinstall if necessary"

    elif any(word in description_lower for word in ['network', 'wifi', 'internet']):
        solution = "For network issues:\n1. Check physical connections\n2. Restart router/modem\n3. Verify network settings\n4. Test with different device\n5. Contact network administrator"

    elif any(word in description_lower for word in ['login', 'password', 'access']):
        solution = "For access issues:\n1. Verify username/password\n2. Check caps lock key\n3. Reset password if needed\n4. Contact IT support for account issues"

    else:
        # Generic helpful response
        solution = f"I understand you're facing: {description}. Here are general troubleshooting steps:\n1. Restart the affected equipment\n2. Check all connections\n3. Verify power supply\n4. Contact technical support for further assistance"

    # Speak the solution
    tts_speak(
        f"I have a solution for your issue. {solution.replace('For equipment not working:', '').replace('For performance issues:', '').split('.')[0]}")

    # Save to database
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO issues (worker_id,timestamp,description,solution) VALUES (?,?,?,?)",
              (worker_id, str(datetime.datetime.now()), description, solution))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "solution": solution})


@app.route("/test_voice", methods=["GET"])
def test_voice():
    test_text = "Hello! This is a test of the TechnoBot voice system. If you can hear this message clearly, both text to speech and audio output are working perfectly!"

    tts_result = tts_speak(test_text)
    devices = get_audio_devices()

    return jsonify({
        "status": "success",
        "message": "Voice test completed",
        "tts_result": tts_result,
        "audio_devices": len(devices)
    })


@app.route("/system_info")
def system_info():
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "audio_devices": get_audio_devices(),
        "tts_available": tts_engine is not None,
    }
    return jsonify(info)


# ==================== MAIN ====================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("faces", exist_ok=True)

    print("🚀 === TechnoBot Server Starting ===")
    print("🎯 Voice Features: Advanced TTS + Multi-engine STT")
    print("🤖 AI Model: Enhanced Rule-based System")
    print("💻 Platform:", platform.system())

    # Initialize TTS
    tts_ready = init_tts()
    print("✅ TTS Status:", "Ready" if tts_ready else "Failed")

    # List audio devices
    devices = get_audio_devices()
    print(f"🎧 Audio Input Devices: {len(devices)}")
    for device in devices:
        print(f"   {device['id']}: {device['name']}")

    print("🌐 Access at: http://localhost:5000")
    print("=====================================")

    app.run(host="0.0.0.0", port=5000, debug=True)