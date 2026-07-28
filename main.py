import os
import json
import logging
import asyncio
import traceback
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configs
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN")
# Fixed URL: changed api.aipipe.org -> aipipe.org
AI_PIPE_URL = os.getenv("AI_PIPE_URL", "https://aipipe.org/openai/v1")
PUBLIC_HOST_URL = os.getenv("PUBLIC_HOST_URL", "http://localhost:8000").rstrip("/")

# Log Storage Setup (Ensure file exists immediately on startup)
LOG_DIR = Path("public")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "run.jsonl"
LOG_FILE.touch(exist_ok=True)


def log_agent_run(user_message: str, parsed_answer: dict, raw_llm_response: str):
    """Appends execution details to run.jsonl for auditability."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": user_message,
        "llm_response": raw_llm_response,
        "parsed_answer": parsed_answer
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# Add MODEL_NAME to your Configs section in main.py
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

def call_ai_pipe(message: str) -> str:
    headers = {
        "Authorization": f"Bearer {AI_PIPE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are an expert Data Analyst LLM agent. "
        "Analyze the input data/question carefully. "
        "Strict Rule: Always return ONLY valid JSON matching the requested payload format specified in the prompt. "
        "No prose, no code blocks (```json), no explanations."
    )

    payload = {
        "model": MODEL_NAME,  # Use the configurable model name variable
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.0
    }

    response = requests.post(AI_PIPE_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    res_data = response.json()
    return res_data["choices"][0]["message"]["content"].strip()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    logger.info(f"Received message: {user_text}")

    try:
        raw_llm_reply = call_ai_pipe(user_text)
        
        # Strip code block markers if present
        clean_reply = raw_llm_reply
        if clean_reply.startswith("```"):
            clean_reply = clean_reply.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
            if clean_reply.startswith("json"):
                clean_reply = clean_reply[4:].strip()

        # Parse extracted answer schema
        parsed_answer = json.loads(clean_reply)

        # Log successful run
        log_agent_run(user_text, parsed_answer, raw_llm_reply)

        final_payload = {
            "answer": parsed_answer,
            "log_url": f"{PUBLIC_HOST_URL}/run.jsonl"
        }
        await update.message.reply_text(json.dumps(final_payload))

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        logger.error(traceback.format_exc())
        
        err_answer = {"error": str(e)}
        
        # Log failed run to run.jsonl so the file is not empty
        log_agent_run(user_text, err_answer, f"Error: {str(e)}")

        err_payload = {
            "answer": err_answer,
            "log_url": f"{PUBLIC_HOST_URL}/run.jsonl"
        }
        await update.message.reply_text(json.dumps(err_payload))


# Lifespan for managing Telegram Bot lifecycle alongside FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Telegram Bot Polling
    telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()
    logger.info("Telegram Bot polling started.")
    
    yield
    
    # Shutdown Telegram Bot
    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()


# Initialize FastAPI App
app = FastAPI(lifespan=lifespan)


# Health Check Root Endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Telegram Data Analyst Bot is running!"}


# Mount Static Files directory for serving run.jsonl
app.mount("/", StaticFiles(directory="public"), name="public")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
