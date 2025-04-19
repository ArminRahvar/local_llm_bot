import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

def ask_ollama(question, context=None):
    if context:
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    else:
        prompt = question

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })

    try:
        return response.json()["message"]["content"].strip()
    except Exception as e:
        print("Ollama error:", e)
        print("Response:", response.text)
        return "Sorry, I couldn't get a response from the local model."
