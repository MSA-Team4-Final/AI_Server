import google.generativeai as genai
from google.generativeai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv  # 추가
import os

# .env 파일 로드
load_dotenv()
app = FastAPI()

# 환경변수에서 API 키 로드
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise Exception("GEMINI_API_KEY 환경변수가 설정되어야 합니다.")


# client = genai.Client(api_key=API_KEY)
# API 키로 Gemini 설정
genai.configure(api_key=API_KEY)


# 요청 데이터 모델 정의
class PromptRequest(BaseModel):
    prompt: str

# 사용할 Gemini 모델 설정
# GenerationConfig를 통해 모델의 동작을 세부적으로 제어할 수 있습니다.
generation_config_dict  = {
  "temperature": 0.7,
  "max_output_tokens": 8192, # 모델에 따라 최대값이 다를 수 있습니다.
}
generation_config = types.GenerationConfig(**generation_config_dict)

# 모델 초기화 시 변환된 객체를 전달합니다.
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config, # ✅ 타입이 맞는 객체를 전달
    system_instruction="You are a helpful assistant.",
)


# FastAPI 엔드포인트 정의
@app.post("/api/chatbot")
async def generate_response_endpoint(request: PromptRequest):
    try:
        # ✅ 수정된 모델 호출 방식
        response = model.generate_content(request.prompt)
        # ----- 👇 디버깅을 위한 코드 추가 -----
        print("--- Full Gemini Response ---")
        print(response)
        print("--------------------------")
        # ------------------------------------

        # 응답이 비어있는지 확인하고, 비어있다면 차단 이유를 확인
        if not response.parts:
            # response.prompt_feedback 에서 차단 이유를 찾을 수 있습니다.
            block_reason = response.prompt_feedback.block_reason
            print(f"Response was blocked. Reason: {block_reason}")
            # 클라이언트에게도 차단 사실을 알려주는 것이 좋습니다.
            raise HTTPException(status_code=400, detail=f"Response blocked by safety filters: {block_reason}")

        return {"response": response.text}

    except HTTPException as http_exc:
        # 이미 처리된 HTTP 예외는 다시 발생시킵니다.
        raise http_exc

    except Exception as e:
        # 에러가 발생하면 서버 로그에 자세한 내용을 출력합니다.
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")
