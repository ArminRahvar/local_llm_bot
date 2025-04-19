import os
import telebot
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ollama_chat import ask_ollama
from src.utils import extract_text_from_pdf,chunk_text,embed_chunks,build_faiss_index

# --- CONFIG ---
TELEGRAM_TOKEN = os.environ['BOT_TOKEN']


bot = telebot.TeleBot(TELEGRAM_TOKEN)
model = SentenceTransformer("all-MiniLM-L6-v2")

user_data = {}  # Stores chunks & faiss index per user


# --- TELEGRAM HANDLERS ---
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Hi! I'm your Ollama-powered chatbot. Just type a message and I'll reply")

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

    user_data[message.message_id] = {
        "chunks": chunks,
        "index": index
    }

    bot.send_message(message.chat.id, "PDF loaded! Now ask me something about it.")

@bot.message_handler(func=lambda message: True)
def handle_question(message):
    if message.reply_to_message and message.reply_to_message.message_id in user_data:
        # This is a reply to a PDF message
        pdf_info = user_data[message.reply_to_message.message_id]
        question = message.text
        q_embedding = model.encode([question])
        D, I = pdf_info["index"].search(np.array(q_embedding), k=3)
        relevant_chunks = [pdf_info["chunks"][i] for i in I[0]]
        context = "\n\n".join(relevant_chunks)

        answer = ask_ollama(question, context)
        bot.send_message(message.chat.id, answer)
    else:
        # No PDF context: fallback to regular chat
        question = message.text
        answer = ask_ollama(question)
        bot.send_message(message.chat.id, answer)

# --- RUN ---
if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(timeout=300, long_polling_timeout=120)
