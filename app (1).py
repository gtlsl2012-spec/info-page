import streamlit as st
import time
import pandas as pd
import io
import json
from typing import List, Dict, Any

# Try multiple possible Google GenAI client imports to maximize compatibility in different environments.
GENAI_AVAILABLE = False
genai = None
try:
    import google.generativeai as genai  # older package import
    GENAI_AVAILABLE = True
except Exception:
    try:
        from google import genai  # newer package layout
        GENAI_AVAILABLE = True
    except Exception:
        GENAI_AVAILABLE = False

# ---------------------------
# System prompt (counseling-specific, concise)
# ---------------------------
DEFAULT_SYSTEM_PROMPT = (
    "1. 사용자의 감정과 고민을 진심으로 공감하며, 따뜻하고 존중하는 말투로 대화하세요.\n"
    "2. 사용자의 상황을 구체적으로 이해하기 위해 언제, 어디서, 어떤 일이 있었는지 자연스럽게 질문하세요.\n"
    "3. 단순한 위로에 그치지 말고, 현실적으로 도움이 될 수 있는 조언이나 방향을 제시하세요."
)

# Models list (user can choose; exclude -exp)
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.0",
    "gemini-1.5",
]

# ---------------------------
# API / Model call wrapper
# ---------------------------
def configure_genai(api_key: str):
    if not GENAI_AVAILABLE:
        raise RuntimeError("GenAI client 패키지가 설치되어 있지 않습니다. requirements를 확인하세요.")
    # Support both possible clients' configure functions
    if hasattr(genai, "configure"):
        genai.configure(api_key=api_key)
    elif hasattr(genai, "client"):
        # some clients use client.init or similar; attempt generic
        try:
            genai.client.configure(api_key=api_key)
        except Exception:
            pass

def call_gemini_chat(api_key: str, model: str, messages: List[Dict[str, str]], max_retries: int = 5) -> Dict[str, Any]:
    \"\"\"Call Gemini-like chat. Includes simple 429 retry logic with exponential backoff.\"\"\"
    if not GENAI_AVAILABLE:
        raise RuntimeError("GenAI client 패키지가 필요합니다. 설치 후 실행하세요.")

    configure_genai(api_key)
    backoff = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Try a few call patterns to support different client libraries
            if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
                return genai.chat.create(model=model, messages=messages)
            elif hasattr(genai, "create_chat_completion"):
                return genai.create_chat_completion(model=model, messages=messages)
            elif hasattr(genai, "client") and hasattr(genai.client, "chat"):
                return genai.client.chat.create(model=model, messages=messages)
            else:
                # Fallback: try genai.ChatCompletion if present
                if hasattr(genai, "ChatCompletion"):
                    return genai.ChatCompletion.create(model=model, messages=messages)
                raise RuntimeError("지원되지 않는 GenAI 클라이언트 인터페이스입니다.")
        except Exception as e:
            errstr = str(e).lower()
            last_exc = e
            if '429' in errstr or 'rate' in errstr or 'quota' in errstr:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    raise RuntimeError("API rate limit: retries exhausted.") from e
            else:
                # Non-retryable
                raise
    raise last_exc

# ---------------------------
# Optional realtime info fetcher (SerpAPI optional)
# ---------------------------
import requests
def fetch_realtime_info(query: str, serpapi_key: str = None) -> str:
    if not serpapi_key:
        return ""
    try:
        params = {"q": query, "api_key": serpapi_key}
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=6)
        data = resp.json()
        items = data.get('organic_results') or data.get('organic') or []
        lines = []
        for it in items[:3]:
            title = it.get('title') or ''
            link = it.get('link') or it.get('url') or ''
            snippet = it.get('snippet') or ''
            lines.append(f"- {title}: {snippet} ({link})")
        return "\\n".join(lines)
    except Exception:
        return ""

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Counseling Chatbot (Gemini)", layout="wide")
st.title("💬 상담 챗봇 — 감정·고민 이해 중심 (Gemini API)")

with st.sidebar:
    st.header("설정 / 세션 정보")
    api_key = st.secrets.get('GEMINI_API_KEY') if st.secrets.get('GEMINI_API_KEY') else st.text_input("GEMINI API Key", type="password")
    serpapi_key = st.secrets.get('SERPAPI_KEY') if st.secrets.get('SERPAPI_KEY') else st.text_input("(선택) SerpAPI Key", type="password")
    model = st.selectbox("모델 선택", AVAILABLE_MODELS, index=0)
    enable_csv = st.checkbox("대화 CSV 자동 기록", value=True)
    st.markdown("---")
    st.markdown("**세션 상태**")
    st.write(f"모델: {model}")
    st.write(f"세션 ID: {st.session_state.get('session_id','(new)')}")
    if not api_key:
        st.warning("GEMINI API 키가 필요합니다. 사이드바에서 입력하거나 st.secrets['GEMINI_API_KEY']에 설정하세요.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}]
if 'full_logs' not in st.session_state:
    st.session_state.full_logs = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(int(time.time()))

# Controls
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("대화 초기화 (시스템 프롬프트 유지)"):
        st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}]
        st.success("세션 초기화 완료")
with col2:
    if st.button("대화 전부 삭제 (새 세션)"):
        st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}]
        st.session_state.full_logs = []
        st.session_state.session_id = str(int(time.time()))
        st.success("새 세션이 생성되었습니다.")
with col3:
    if st.session_state.full_logs:
        df = pd.DataFrame(st.session_state.full_logs)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("로그 다운로드 (CSV)", csv_bytes, file_name=f"chat_logs_{st.session_state.session_id}.csv", mime='text/csv')

st.subheader("대화창 — 사용자의 감정과 고민을 공감하며 상담하세요")

def render_history():
    for turn in st.session_state.history[1:]:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            st.markdown(f"**사용자:** {content}")
        elif role == 'assistant':
            st.markdown(f"**상담봇:** {content}")

render_history()

user_input = st.text_area("메시지 입력", height=140, placeholder="감정이나 고민을 편하게 적어주세요. 상담봇이 공감하고 도와드릴게요.")
if st.button("전송") and user_input.strip():
    st.session_state.history.append({'role':'user', 'content': user_input.strip()})
    messages_for_api = [{'role': m['role'], 'content': m['content']} for m in st.session_state.history]
    realtime_summary = fetch_realtime_info(user_input[:160], serpapi_key)
    if realtime_summary:
        messages_for_api.append({'role':'system', 'content': f"[실시간 정보]\\n{realtime_summary}"})
    try:
        response = call_gemini_chat(api_key=api_key, model=model, messages=messages_for_api)
        resp_text = ""
        try:
            if isinstance(response, dict):
                choices = response.get('choices') or []
                if choices:
                    # common patterns
                    resp_text = choices[0].get('message', {}).get('content') or choices[0].get('text') or ''
                else:
                    resp_text = response.get('content') or str(response)
            else:
                resp_text = getattr(response, 'content', None) or getattr(response, 'response', None) or str(response)
        except Exception:
            resp_text = str(response)
        assistant_message = resp_text or "(응답을 받아오지 못했습니다.)"
        st.session_state.history.append({'role':'assistant', 'content': assistant_message})
        if enable_csv:
            st.session_state.full_logs.append({'timestamp': int(time.time()), 'role':'user', 'text': user_input.strip()})
            st.session_state.full_logs.append({'timestamp': int(time.time()), 'role':'assistant', 'text': assistant_message})
        st.experimental_rerun()
    except RuntimeError as e:
        st.warning(f"모델 호출 중 오류가 발생했습니다: {e}")
        preserved = st.session_state.history[-6:]
        st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}] + preserved
        st.error("요청이 과다하여 세션을 최근 6턴으로 축소한 뒤 재시작했습니다. 다시 시도해주세요.")

st.markdown("---")
st.caption("앱 노트: GenAI 클라이언트 패키지가 없으면 사이드바에서 API 키를 입력해도 모델 호출이 불가합니다. requirements를 확인하세요.")

with st.expander("세션/디버그 정보"):
    st.write({'session_id': st.session_state.session_id, 'history_len': len(st.session_state.history)})
    st.json(st.session_state.history)
