# TechnoBot Chatbot – Raspberry Pi + Arduino Mega Compatible
# Modules: STT, TTS, LLM, Serial, Vision integration (optional)

import serial
import time
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import openai  # Replace with your LLM API if using Ollama/Gemma/Gemini
from langdetect import detect

# -------------------------
# 1. Arduino Serial Setup
# -------------------------
arduino_port = '/dev/ttyUSB0'  # Check with `ls /dev/tty*`
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # wait for serial connection to initialize

# -------------------------
# 2. LLM / Chatbot Setup
# -------------------------
# Example: OpenAI GPT-4 API (replace with Ollama/Gemma API as needed)
openai.api_key = 'YOUR_API_KEY'

def query_llm(prompt_text):
    """
    Query your LLM for response.
    Replace with your local/offline LLM integration if needed.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.5
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Error querying LLM: {e}"

# -------------------------
# 3. Text-to-Speech Function
# -------------------------
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts_file = "tts_output.mp3"
    tts.save(tts_file)
    playsound.playsound(tts_file)
    os.remove(tts_file)

# -------------------------
# 4. Speech-to-Text Function
# -------------------------
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"STT request failed: {e}")
        return ""

# -------------------------
# 5. Arduino Communication
# -------------------------
def send_to_arduino(command):
    ser.write(f"{command}\n".encode())
    time.sleep(0.5)
    if ser.in_waiting:
        feedback = ser.readline().decode().strip()
        return feedback
    return ""

# -------------------------
# 6. Main TechnoBot Loop
# -------------------------
def main():
    print("TechnoBot activated!")
    speak("Hello! I am TechnoBot. Show me your medicine or tell me your query.")

    while True:
        # 1. Listen to user
        user_input = listen()
        if user_input == "":
            speak("Sorry, I did not hear you. Please repeat.")
            continue

        # 2. Detect language for multilingual support
        try:
            lang_code = detect(user_input)
        except:
            lang_code = 'en'

        # 3. Check if user wants robot to do a hardware action
        if "scan" in user_input.lower():
            # Example: move scanning arm via Arduino
            feedback = send_to_arduino("SCAN")
            speak(f"Scanning medicine. {feedback}", lang=lang_code)
            # Optional: integrate OpenCV medicine recognition here
            # medicine_name = scan_medicine()
            # user_input += f" I want info about {medicine_name}"
        elif "move" in user_input.lower():
            # Example: move forward/backward/turn
            feedback = send_to_arduino("MOVE")
            speak(f"Robot moving. {feedback}", lang=lang_code)
        else:
            # 4. Query chatbot LLM for response
            answer = query_llm(user_input)
            speak(answer, lang=lang_code)

# -------------------------
# Run Main Loop
# -------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down TechnoBot...")
        ser.close()
