import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수 설정
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise Exception("GEMINI_API_KEY 환경변수가 설정되어야 합니다.")

# API 키로 Gemini 설정
genai.configure(api_key=API_KEY)

# Gemini 모델 설정
generation_config = types.GenerationConfig(
    temperature=0.7,
    max_output_tokens=8192,
)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are a helpful assistant. Remember the context of our conversation and refer to previous messages when relevant. Always maintain conversation continuity.",
)
