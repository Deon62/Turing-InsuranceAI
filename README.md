
---

# Turing Insurance AI

## Overview

Turing Insurance AI is a comprehensive solution for automating vehicle insurance claims using machine learning and AI-powered tools. The project includes multiple features:

* **Car Damage Analysis**: Uses a deep learning model hosted on Hugging Face to analyze images of vehicles for damage detection and provide insights.
* **OCR (Optical Character Recognition)**: Extracts text from images of documents like IDs, driver’s licenses, and police reports, which is then processed for better understanding.
* **Turing Chatbot**: A conversational AI powered by Gemini, designed to answer user questions regarding the insurance process and related queries.

### Key Features

1. **Car Damage Detection**:
   The backend includes a **ResNet-50 model** for detecting car damage from images, hosted on Hugging Face at [car-damage-resnet50](https://huggingface.co/chinesemusk/car-damage-resnet50). This model processes images uploaded via a simple web interface and returns an analysis of the damage.

2. **OCR (Document Scanning)**:
   The OCR module extracts text from scanned images of **IDs**, **driver’s licenses**, and **police reports**, and then passes this data to a **DeepSeek model** for structured information display. This feature is fully integrated into the MVP frontend.

3. **Turing Chatbot**:
   The **Turing Chatbot** is a conversational assistant built using Gemini, designed to help users understand and interact with the insurance process. It can answer various questions about insurance claims, coverage, and policy details.

### MVP (Minimum Viable Product)

The MVP provides a **simple frontend** (written in **HTML** and **CSS**) for users to interact with the system. The MVP allows users to:

* Upload images of vehicles for damage analysis.
* Use the OCR feature to scan and extract text from uploaded document images.
* Interact with the **Turing Chatbot** to get answers to insurance-related queries.

The MVP is located in the `MVP/` folder.

---

## Project Structure

```plaintext
Turing-InsuranceAI/
│
├── MVP/
│   ├── index.html        # Frontend for user interactions
│   └── style.css         # Styles for the frontend
│
├── ragfinancialchatbot/  # Chatbot implementation
│   └── chatbot.py        # Chatbot backend (currently Gemini)
│
├── car_damage_model_hf/  # Folder to store the Hugging Face model and related files
│   └── model.py          # Script to load and run the model
│
└── README.md             # Project documentation
```

---

## Getting Started

### Prerequisites

* Python 3.8+
* Required Python libraries:

  * `huggingface_hub`
  * `transformers`
  * `deepseek`
  * `flask` (or any web framework you're using for deployment)
  * `opencv-python` (for OCR)
  * `gemini` (for chatbot integration)

### 1. Clone the Repository

```bash
git clone https://github.com/Deon62/Turing-InsuranceAI.git
cd Turing-InsuranceAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up the Model

The car damage model is hosted on Hugging Face. To use the model, you'll need to load it in your backend script (e.g., `car_damage_model_hf/model.py`).

```python
from transformers import AutoModelForImageClassification, AutoProcessor

# Load model from Hugging Face
processor = AutoProcessor.from_pretrained("chinesemusk/car-damage-resnet50")
model = AutoModelForImageClassification.from_pretrained("chinesemusk/car-damage-resnet50")
```

### 4. Run the Frontend

The frontend is located in the `MVP/` folder. You can open `index.html` in your browser to test the MVP locally.

### 5. Use the OCR Feature

To use the OCR for extracting text from images, ensure that the image path is correctly passed to the OCR module. The results are passed to the **DeepSeek model** for better organization.

### 6. Start the Turing Chatbot

The chatbot can be run with the provided Python script in `ragfinancialchatbot/chatbot.py`. Ensure you've set up your Gemini model or any chatbot backend you're using.

```bash
python ragfinancialchatbot/chatbot.py
```

---

## Contributing

If you wish to contribute to this project, feel free to fork the repository, submit issues, and make pull requests. All contributions are welcome!

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
