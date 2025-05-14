import easyocr
import cv2
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


print("Initializing OCR reader...")
reader = easyocr.Reader(['en'], gpu=False)

image_path = 'images/passportform.jpg'
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit(1)

print(f"Reading image from {image_path}...")
img = cv2.imread(image_path)
result = reader.readtext(img, detail=0)
ocr_data = " ".join(result)
print(f"OCR extracted text: {ocr_data[:100]}..." if len(ocr_data) > 100 else ocr_data)


prompt = PromptTemplate(
    input_variables=["ocr_data"],
    template="""
    You are a helpful assistant.
    You are given the following OCR data: {ocr_data}
    Please provide a concise summary of the OCR data.
    """
)

try:
    print("Connecting to DeepSeek API...")
    # Configure the LLM with DeepSeek
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key="sk-..................",
        base_url="https://api.deepseek.com/v1"  # Updated URL
    )

    chain = (
        {"ocr_data": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    print("Processing OCR data with LLM...")
    summary = chain.invoke(ocr_data)
    print("\nSummary:")
    print(summary)
except Exception as e:
    print(f"Error connecting to DeepSeek API: {e}")
    print("\nTrying with alternative API URL...")
    try:
        # Try with alternative URL
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            api_key="sk-...........................",
            base_url="https://api.deepseek.ai/v1"  # Alternative URL
        )
        chain = (
            {"ocr_data": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        summary = chain.invoke(ocr_data)
        print("\nSummary:")
        print(summary)
    except Exception as e:
        print(f"Error with alternative API URL: {e}")
        print("\nPlease check your internet connection and API key validity.")
