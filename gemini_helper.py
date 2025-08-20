import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def categorize_email(email_text):
    prompt = f"""
    Categorize the following email into one of these priorities:
    - High Priority
    - Medium Priority
    - Low Priority

    Email:
    \"\"\"{email_text}\"\"\"

    Respond with only the priority label.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


