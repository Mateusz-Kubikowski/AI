from flask import Flask, request, jsonify
import google.generativeai as genai
import logging
import re
import os

# Inicjalizacja Flask
app = Flask(__name__)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)

# Ustaw swój API key tutaj
GENAI_API_KEY = os.environ.get("GOOGLE_API_KEY", "Your_Gemini_API_Key

# Konfiguracja Gemini
genai.configure(api_key=GENAI_API_KEY)

# Wybór modelu (np. gemini-pro)
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "")
        if not prompt:
            logging.warning("Brak promptu w żądaniu.")
            return jsonify({"error": "No prompt provided"}), 400

        logging.info(f"Prompt received:\n{prompt}")

        response = model.generate_content(prompt)
        generated_text = response.text.strip()

        # Filtrowanie do pierwszego pytania (opcjonalne, jak w oryginale)
        match = re.search(r"(1\..*)", generated_text, re.DOTALL)
        if match:
            generated_text = match.group(1).strip()

        logging.info(f"Generated response:\n{generated_text}")

        return jsonify({"response": generated_text})

    except Exception as e:
        logging.exception("Wystąpił błąd podczas generowania odpowiedzi.")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    logging.info("Uruchamianie serwera z Gemini AI...")
    app.run(host="0.0.0.0", port=5000)
