from flask import Flask, request, jsonify
import google.generativeai as genai
from config1 import GEMINI_API_KEY
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import playsound
from flask import send_file
import pyttsx3
import time

# Check if PyAudio is available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio is not installed. Voice input will not be available.")
    print("To enable voice input, please install PyAudio using:")
    print("pip install pipwin")
    print("pipwin install pyaudio")

class TuringInsuranceBot:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 80192,
            }
        )
        self.system_instructions = """You're Turing, an AI insurance assistant. Your role is to:
        1. Help users understand insurance policies and concepts
        2. Guide them through the claims process
        3. Provide information about different insurance products
        4. Answer questions about coverage and premiums
        5. Assist with policy management
        6. Offer personalized insurance recommendations
        Be professional, friendly, and clear in your explanations. Always prioritize accuracy and compliance with insurance regulations."""
        self.conversation_history = []
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise
        if PYAUDIO_AVAILABLE:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("Ambient noise adjustment complete.")

    def listen_for_command(self):
        if not PYAUDIO_AVAILABLE:
            print("Voice input is not available. Please use text input instead.")
            return None
            
        try:
            with sr.Microphone() as source:
                print("\nListening... (Speak now)")
                # Reduce the timeout and phrase_time_limit for faster response
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing speech...")
                try:
                    query = self.recognizer.recognize_google(audio)
                    print(f"You said: {query}")
                    return query
                except sr.UnknownValueError:
                    print("Could not understand audio. Please try again.")
                    return None
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def speak_response(self, text, voice_gender='male'):
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            voice_id = voices[0].id if voice_gender == 'male' else voices[1].id
            engine.setProperty('voice', voice_id)
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            print("Response:", text)

    def process_input(self, input_type="text"):
        if input_type == "voice" and PYAUDIO_AVAILABLE:
            while True:
                query = self.listen_for_command()
                if query:
                    return query
                print("Would you like to try again? (y/n)")
                retry = input().lower()
                if retry != 'y':
                    return None
        else:
            return input("\nEnter your insurance question (or 'quit' to exit): ")

    def deliver_response(self, response, output_type="text", voice_gender="male"):
        print("\nInsurance Assistant:", response)
        if output_type == "voice":
            self.speak_response(response, voice_gender)

    def get_response(self, query):
        # Add the user's query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Create the prompt with conversation history
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in self.conversation_history[-5:]  # Keep last 5 exchanges
        ])
        
        prompt = f"""
        {self.system_instructions}
        
        Previous conversation:
        {conversation_context}
        
        Current question: {query}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Add the assistant's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(error_msg)
            return "I apologize, but I encountered an error. Please try again."

# Flask app
app = Flask(__name__)
insurance_bot = TuringInsuranceBot(api_key=GEMINI_API_KEY)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    response = insurance_bot.get_response(query)
    return jsonify({"response": response})

@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    if not PYAUDIO_AVAILABLE:
        return jsonify({"error": "Voice input is not available. Please install PyAudio."}), 400
        
    audio_file = request.files.get('audio')
    output_type = request.form.get('output_type', 'text')
    
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio = insurance_bot.recognizer.record(source)
            query = insurance_bot.recognizer.recognize_google(audio)
            response = insurance_bot.get_response(query)
            
            if output_type == 'voice':
                tts = gTTS(text=response, lang='en')
                temp_file = "response.mp3"
                tts.save(temp_file)
                return send_file(
                    temp_file,
                    mimetype="audio/mpeg",
                    as_attachment=True,
                    download_name="response.mp3"
                )
            else:
                return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Welcome to Turing Insurance Assistant!")
    
    if PYAUDIO_AVAILABLE:
        print("Voice capabilities are enabled.")
        print("\nChoose your interaction mode:")
        print("1. Text to Text")
        print("2. Voice to Voice")
        print("\nSelect voice gender:")
        print("1. Male")
        print("2. Female")
        
        voice_choice = input("Select voice (1-2): ")
        voice_gender = 'male' if voice_choice == '1' else 'female'
        
        mode = input("Select mode (1-2): ")
    else:
        print("Running in text-only mode.")
        mode = "1"
        voice_gender = "male"
    
    while True:
        input_type = "voice" if mode in ["2", "4"] and PYAUDIO_AVAILABLE else "text"
        output_type = "voice" if mode in ["2", "3"] else "text"
        
        query = insurance_bot.process_input(input_type)
        if not query or query.lower() == 'quit':
            break
            
        response = insurance_bot.get_response(query)
        insurance_bot.deliver_response(response, output_type, voice_gender)
        print("\n" + "="*50)