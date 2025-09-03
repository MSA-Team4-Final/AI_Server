import google.generativeai as genai
from google.generativeai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uuid
from typing import Optional, Dict, Any

# .env 파일 로드
load_dotenv()
app = FastAPI()

# 환경변수에서 API 키 로드
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise Exception("GEMINI_API_KEY 환경변수가 설정되어야 합니다.")

# API 키로 Gemini 설정
genai.configure(api_key=API_KEY)

# 채팅 세션을 저장할 딕셔너리 (실제 환경에서는 Redis 등 사용 권장)
chat_sessions: Dict[str, Any] = {}


# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    prompt: str
    sessionId: Optional[str] = None


# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    response: str
    sessionId: str


# Gemini 모델 설정
generation_config_dict = {
    "temperature": 0.7,
    "max_output_tokens": 8192,
}
generation_config = types.GenerationConfig(**generation_config_dict)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are a helpful assistant. Remember the context of our conversation and refer to previous messages when relevant. Always maintain conversation continuity.",
)


def get_or_create_chat_session(session_id: Optional[str] = None):
    """채팅 세션을 가져오거나 새로 생성하는 함수"""

    # 기존 세션이 있는 경우 반환
    if session_id and session_id in chat_sessions:
        chat_session = chat_sessions[session_id]
        print(f"--- Continuing existing chat session: {session_id} ---")
        print(f"--- Current history length: {len(chat_session.history)} messages ---")
        return chat_session, session_id

    # 새로운 세션 생성
    new_session_id = session_id or str(uuid.uuid4())

    # 새로운 채팅 세션 시작 (빈 히스토리로 시작)
    chat_session = model.start_chat(history=[])

    # 세션 딕셔너리에 저장
    chat_sessions[new_session_id] = chat_session

    print(f"--- New chat session created: {new_session_id} ---")
    return chat_session, new_session_id


@app.post("/api/chatbot", response_model=ChatResponse)
async def generate_response_endpoint(request: ChatRequest):
    try:
        # 채팅 세션 가져오기 또는 생성
        chat_session, session_id = get_or_create_chat_session(request.sessionId)

        print(f"User message: {request.prompt}")
        print(f"Session ID: {session_id}")

        # 기존 대화 히스토리 출력 (디버깅용)
        if hasattr(chat_session, 'history') and chat_session.history:
            print("=== Current Conversation History ===")
            for i, message in enumerate(chat_session.history):
                role = message.role
                content = message.parts[0].text if message.parts else "No content"
                print(f"{i + 1}. {role}: {content[:100]}...")
            print("=" * 40)

        # ChatSession을 통해 메시지 전송 (문맥 자동 유지)
        response = chat_session.send_message(request.prompt)

        # 응답 디버깅
        print("--- Gemini Response ---")
        print(response.text)
        print("-" * 25)

        # 응답 차단 확인
        if not response.parts:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown') if hasattr(response,
                                                                                                   'prompt_feedback') else 'Unknown'
            print(f"Response was blocked. Reason: {block_reason}")
            raise HTTPException(status_code=400, detail=f"Response blocked by safety filters: {block_reason}")

        return ChatResponse(response=response.text, sessionId=session_id)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")


# 추가 유틸리티 엔드포인트들

@app.get("/api/chatbot/history/{session_id}")
async def get_chat_history(session_id: str):
    """특정 세션의 대화 히스토리 조회"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_session = chat_sessions[session_id]
    history = []

    if hasattr(chat_session, 'history'):
        for message in chat_session.history:
            history.append({
                "role": message.role,
                "content": message.parts[0].text if message.parts else "",
                "timestamp": getattr(message, 'timestamp', None)
            })

    return {
        "sessionId": session_id,
        "history": history,
        "messageCount": len(history)
    }


@app.delete("/api/chatbot/session/{session_id}")
async def delete_chat_session(session_id: str):
    """특정 채팅 세션 삭제"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/chatbot/sessions")
async def list_active_sessions():
    """활성 세션 목록 및 각 세션의 메시지 수 조회"""
    sessions_info = {}
    for session_id, chat_session in chat_sessions.items():
        message_count = len(chat_session.history) if hasattr(chat_session, 'history') else 0
        sessions_info[session_id] = {
            "messageCount": message_count,
            "lastMessage": chat_session.history[-1].parts[0].text[
                               :50] + "..." if chat_session.history else "No messages"
        }

    return {
        "activeSessions": list(chat_sessions.keys()),
        "sessionsInfo": sessions_info,
        "totalSessions": len(chat_sessions)
    }