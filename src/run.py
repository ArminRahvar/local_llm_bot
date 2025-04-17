import os
import telebot
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ollama_chat import ask_ollama
from src.utils import extract_text_from_pdf,chunk_text,embed_chunks,build_faiss_index

# --- CONFIG ---
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"


bot = telebot.TeleBot(TELEGRAM_TOKEN)
model = SentenceTransformer("all-MiniLM-L6-v2")

user_data = {}  # Stores chunks & faiss index per user


# --- TELEGRAM HANDLERS ---
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Hey! Send me a PDF and I’ll answer questions about it.")

@bot.message_handler(content_types=["document"])
def handle_pdf(message):
    if message.document.mime_type != "application/pdf":
        bot.reply_to(message, "Please send a valid PDF file.")
        return

    file_info = bot.get_file(message.document.file_id)
    downloaded = bot.download_file(file_info.file_path)

    os.makedirs("files", exist_ok=True)
    file_path = f"files/{message.chat.id}.pdf"
    with open(file_path, "wb") as f:
        f.write(downloaded)

    bot.reply_to(message, "Processing your PDF...")

    # Process
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))

    user_data[message.chat.id] = {
        "chunks": chunks,
        "index": index
    }

    bot.send_message(message.chat.id, "PDF loaded! Now ask me something about it.")

@bot.message_handler(func=lambda message: True)
def handle_question(message):
    data = user_data.get(message.chat.id)
    if not data:
        bot.reply_to(message, "Please send a PDF first.")
        return

    question = message.text
    q_embedding = model.encode([question])
    D, I = data["index"].search(np.array(q_embedding), k=3)
    relevant_chunks = [data["chunks"][i] for i in I[0]]
    context = "\n\n".join(relevant_chunks)

    answer = ask_ollama(question, context)
    bot.send_message(message.chat.id, answer)

# --- RUN ---
if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling()
