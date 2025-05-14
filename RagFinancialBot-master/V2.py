import google.generativeai as genai
import speech_recognition as sr
from config1 import GEMINI_API_KEY, ELEVEN_LABS_KEY
from elevenlabs import generate, set_api_key, Voice, VoiceSettings
import pygame
from io import BytesIO
import time

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
set_api_key(ELEVEN_LABS_KEY)

class DeoxFoodBot:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.history = []
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Configure the model
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
            system_instruction="""You're a friendly and helpful assistant for DeoxFood, a food delivery service around Egerton University. 
            Your role is to:
            1. Help customers place food orders
            2. Provide information about available menu items
            3. Answer questions about delivery times and areas
            4. Handle special requests and dietary requirements
            5. Provide order status updates
            6. Offer personalized food recommendations
            Be friendly, enthusiastic, and helpful while maintaining a professional tone."""
        )
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ambient noise adjustment complete.")

    def speak_response(self, text):
        try:
            # Generate audio using the current ElevenLabs API
            audio = generate(
                text=text,
                voice="Rachel",  # Using a default voice
                model="eleven_monolingual_v1"
            )
            
            # Play using pygame
            pygame.mixer.music.load(BytesIO(audio))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            print("Response:", text)

    def get_voice_input(self):
        try:
            with sr.Microphone() as source:
                print("\nListening... (Speak now)")
                # Reduced timeout and phrase time limit for faster response
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing speech...")
                try:
                    text = self.recognizer.recognize_google(audio)
                    print("You said:", text)
                    return text
                except sr.UnknownValueError:
                    print("Could not understand audio. Please try again.")
                    return None
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def process_input(self, input_mode):
        if input_mode == "voice":
            while True:
                user_input = self.get_voice_input()
                if user_input:
                    return user_input
                print("Would you like to try again? (y/n)")
                retry = input().lower()
                if retry != 'y':
                    return None
        else:
            return input("You: ")

    def get_response(self, user_input):
        try:
            chat_session = self.model.start_chat(history=self.history)
            response = chat_session.send_message(user_input)
            model_response = response.text
            
            # Update conversation history
            self.history.append({"role": "user", "parts": [user_input]})
            self.history.append({"role": "model", "parts": [model_response]})
            
            return model_response
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(error_msg)
            return "I apologize, but I encountered an error. Please try again."

def main():
    bot = DeoxFoodBot()
    print("Welcome to DeoxFoods Delivery!")
    print("\nChoose your preferred input method:")
    print("1. Voice commands")
    print("2. Text input")
    
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    input_mode = "voice" if choice == "1" else "text"
    print(f"\nGreat! Using {input_mode} input mode.")
    print("Type or say 'switch' to change modes, 'quit' to exit")
    
    # Welcome message
    welcome_msg = "Welcome to Deoxfoods! How can I help you today?"
    print("Bot:", welcome_msg)
    bot.speak_response(welcome_msg)
    
    while True:
        user_input = bot.process_input(input_mode)
        
        if not user_input:
            continue
            
        if user_input.lower() == 'switch':
            input_mode = "text" if input_mode == "voice" else "voice"
            print(f"\nSwitched to {input_mode} input mode")
            continue
        
        if user_input.lower() == 'quit':
            goodbye_msg = "Thank you for using DeoxFoods! Have a great day!"
            print("Bot:", goodbye_msg)
            bot.speak_response(goodbye_msg)
            break
        
        response = bot.get_response(user_input)
        print("Bot:", response)
        bot.speak_response(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 