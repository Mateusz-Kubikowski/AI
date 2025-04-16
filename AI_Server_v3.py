from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

app = Flask(__name__)

model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
cache_dir = "PATH_TO_STORE_YOUR_AI_MODEL"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    cache_dir=cache_dir
)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "")
        if not prompt:
            logging.warning("Brak promptu w żądaniu.")
            return jsonify({"error": "No prompt provided"}), 400

        logging.info(f"Prompt received:\n{prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )

        # Obcięcie powtórzonego promptu
        prompt_len = inputs["input_ids"].shape[-1]
        # Obcięcie promptu
        generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        # Filtrowanie do pierwszego pytania
        import re
        match = re.search(r"(1\..*)", generated_text, re.DOTALL)
        if match:
            generated_text = match.group(1).strip() 

        logging.info(f"Generated response:\n{generated_text}")

        return jsonify({"response": generated_text})

    except Exception as e:
        logging.exception("Wystąpił błąd podczas generowania odpowiedzi.")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    logging.info("Uruchamianie serwera...")
    app.run(host="0.0.0.0", port=5000)
