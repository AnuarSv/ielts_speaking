import sqlite3
import hashlib
import google.generativeai as genai

# =====================
# Настройки Gemini и SQLite
# =====================
API_KEY = "API"  # Укажи свой ключ
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.3
TOP_P = 0.9
TOP_K = 50
HISTORY_LIMIT = 5  # сколько последних сообщений подставлять

INSTRUCTION = '''
You are an IELTS Speaking teacher. Your goal is to help the student improve their speaking skills for the IELTS exam. 
Do the following:
1. Ask questions from Part 1, Part 2, and Part 3 of the IELTS Speaking test. 
2. Evaluate the student's answers and give detailed feedback, including grammar, vocabulary, pronunciation tips, and fluency suggestions. 
3. Provide model answers and tips on how to improve.
4. Always respond in a friendly, encouraging, and patient way. 
5. Only speak in English.
6. Ask one question at a time and wait for the student's answer.
7. Do not use any emojis, emoticons, or punctuation marks that indicate emotion such as !, ?, *, or similar symbols.
'''

# =====================
# Настройка API один раз
# =====================
genai.configure(api_key=API_KEY)

# =====================
# Настройка и создание базы SQLite
# =====================
conn = sqlite3.connect("ielts_memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS dialogue_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT,
        model_response TEXT,
        hash TEXT UNIQUE
    )
''')
conn.commit()

# =====================
# Функции памяти и хеширования
# =====================
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def save_to_memory(user_msg: str, model_resp: str):
    h = hash_text(user_msg + model_resp)
    try:
        cursor.execute(
            "INSERT INTO dialogue_memory (user_message, model_response, hash) VALUES (?, ?, ?)",
            (user_msg, model_resp, h)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # уже сохранено

def get_history(limit: int):
    cursor.execute(
        "SELECT user_message, model_response FROM dialogue_memory ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    history_text = ""
    for user_msg, model_resp in reversed(rows):
        history_text += f"User: {user_msg}\nTeacher: {model_resp}\n"
    return history_text.strip()

# =====================
# Основная функция общения с Gemini
# =====================
def ask_gemini(prompt: str) -> str:
    try:
        # Инициализация модели
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "response_mime_type": "text/plain"
            },
            system_instruction=INSTRUCTION
        )

        # Собираем историю
        history_context = get_history(HISTORY_LIMIT)
        full_prompt = (history_context + "\n" if history_context else "") + f"User: {prompt}\nTeacher:"

        # Получаем ответ
        response = model.generate_content(full_prompt)

        # Извлекаем текст
        answer = response.text.strip() if hasattr(response, "text") else str(response)
        answer = ' '.join(line.strip() for line in answer.splitlines())

        # Сохраняем в базу
        save_to_memory(prompt, answer)

        return answer

    except Exception as e:
        return f"ERROR: {e}"
