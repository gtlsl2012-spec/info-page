import streamlit as st
import time
import pandas as pd
import io
import json
from typing import List, Dict, Any

# Try to import Google's Generative AI client. If not available, we'll provide a helpful error at runtime.
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    GOOGLE_GENAI_AVAILABLE = False

# ---------------------------
# Minimal helper functions
# ---------------------------
DEFAULT_SYSTEM_PROMPT = \"\"\"당신은 사용자의 **개인적인 취향과 제약 조건(예산, 기간, 동행)**을 깊이 이해하고, 이를 바탕으로 최적의 맞춤 여행지 및 맞춤 일정을 추천하는 전문 여행 큐레이터 AI입니다.
대화 원칙: 응답은 친절하고, 호기심을 유발하며, 전문적인 여행 전문가의 말투를 사용합니다. 사용자에게 새로운 가능성을 제안하되, 항상 여행 경험에 대한 긍정적인 기대감을 심어주도록 노력합니다.
취향 파악 단계(최우선): 추천에 앞서, 사용자의 취향을 최소 3가지 이상의 구체적인 질문을 통해 심층적으로 파악해야 합니다.
필수 파악 항목: 여행 스타일 (휴식 vs 액티비티), 선호 분위기 (도시 vs 자연), 동행 (혼자 vs 그룹), 예산/기간.
질문 예시: \"최근 여행에서 가장 좋았던 경험은 무엇이었나요?\" 또는 \"숙소는 가격보다 청결이 중요한가요, 분위기가 중요한가요?\"
추천 출력 형식 (3단계 구조): 1) 핵심 추천지 및 이유 (예: \"사용자님의 [A한 취향]과 [B한 스타일]을 고려하여 이 곳을 선택했습니다.\") 2) 3박 4일 맞춤형 일정 초안 (간결 요약) 3) 다음 단계 안내 (이 추천지에 대해 더 궁금한 점이 있는지, 다른 후보지 비교 원하시는지 질문)
정보 활용: 답변 생성 시, 외부 검색된 실시간 정보(예: 날씨, 축제 등)를 반드시 활용하여 최신성을 확보하고, 출처가 명확한 정보만을 제공합니다. (실행 환경에서 실시간 검색 API 키가 없으면 해당 기능은 비활성화됩니다.)
\"\"\"

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
    if not GOOGLE_GENAI_AVAILABLE:
        raise RuntimeError(\"google.generativeai 패키지가 설치되어 있지 않습니다. requirements.txt를 확인하세요.\")
    genai.configure(api_key=api_key)


def call_gemini_chat(api_key: str, model: str, messages: List[Dict[str, str]], max_retries: int = 5) -> Dict[str, Any]:
    \"\"\"Call Gemini chat. Includes 429 retry logic with exponential backoff.
       If repeated 429s occur and retries exhausted, this function will raise a RuntimeError.
    \"\"\"
    if not GOOGLE_GENAI_AVAILABLE:
        raise RuntimeError(\"google.generativeai 패키지가 필요합니다. 설치 후 실행하세요.\")

    configure_genai(api_key)
    backoff = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # The google.generativeai chat API surface may vary by SDK version.
            # This code uses the common 'chat' entrypoint. If your environment uses a different call,
            # please adapt accordingly.
            response = genai.chat.create(model=model, messages=messages)
            return response
        except Exception as e:
            # Basic handling: if this looks like a rate-limit error, wait and retry.
            errstr = str(e).lower()
            last_exc = e
            if '429' in errstr or 'rate' in errstr or 'quota' in errstr:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    # Retries exhausted
                    raise RuntimeError(\"API rate limit: retries exhausted.\") from e
            else:
                # Non-retryable
                raise

    raise last_exc

# ---------------------------
# Optional realtime info fetcher (requires SERPAPI_KEY or other)
# ---------------------------
import requests
def fetch_realtime_info(query: str, serpapi_key: str = None) -> str:
    \"\"\"Attempt to fetch realtime info via SerpAPI if key provided.
       Returns a short summary with source links. If no key, returns an empty string.
    \"\"\"
    if not serpapi_key:
        return \"\"
    try:
        params = {\"q\": query, \"api_key\": serpapi_key}
        # SerpAPI endpoint example - user must provide real key and may need to adjust for their provider
        resp = requests.get(\"https://serpapi.com/search.json\", params=params, timeout=8)
        data = resp.json()
        # Build a short summary from organic results
        items = data.get('organic_results') or data.get('organic') or []
        lines = []
        for it in items[:3]:
            title = it.get('title') or it.get('position') or ''
            link = it.get('link') or it.get('url') or ''
            snippet = it.get('snippet') or it.get('snippet_highlighted') or ''
            lines.append(f\"- {title}: {snippet} ({link})\")
        return \"\\n\".join(lines)
    except Exception:
        return \"\"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=\"Gemini Travel Concierge\", layout=\"wide\")

st.title(\"✈️ Gemini 기반 AI 여행 큐레이터 — 고객 응대 챗봇 (Streamlit)\")

# Sidebar: settings
with st.sidebar:
    st.header(\"설정 / 세션 정보\")
    api_key = st.secrets.get('GEMINI_API_KEY') if st.secrets.get('GEMINI_API_KEY') else st.text_input(\"GEMINI API Key\", type=\"password\")
    serpapi_key = st.secrets.get('SERPAPI_KEY') if st.secrets.get('SERPAPI_KEY') else st.text_input(\"(선택) SerpAPI Key (실시간 검색용)\", type=\"password\")
    model = st.selectbox(\"모델 선택\", AVAILABLE_MODELS, index=0)
    enable_csv = st.checkbox(\"대화 CSV 자동 기록\", value=True)
    st.markdown(\"---\")
    st.markdown(\"**세션 상태**\")
    st.write(f\"모델: {model}\")
    st.write(f\"세션 ID: {st.session_state.get('session_id','(new)')}\")
    if not api_key:
        st.warning(\"GEMINI API 키가 필요합니다. 사이드바에서 입력하거나 st.secrets['GEMINI_API_KEY']에 설정하세요.\")


# Initialize session state
if 'history' not in st.session_state:
    # history list of dicts: {'role': 'user'/'assistant'/'system', 'content': '...'}
    st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}]
if 'full_logs' not in st.session_state:
    st.session_state.full_logs = []  # for CSV export
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(int(time.time()))

# Controls: reset, download logs
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button(\"대화 초기화 (초기 시스템 프롬프트 유지)\"):
        st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}]
        st.success(\"세션 초기화 완료 — 시스템 프롬프트는 유지됩니다.\")
with col2:
    if st.button(\"대화 전부 삭제 (새 세션)\"):
        st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}]
        st.session_state.full_logs = []
        st.session_state.session_id = str(int(time.time()))
        st.success(\"새 세션이 생성되었습니다.\")
with col3:
    if st.session_state.full_logs:
        df = pd.DataFrame(st.session_state.full_logs)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(\"로그 다운로드 (CSV)\", csv_bytes, file_name=f\"chat_logs_{st.session_state.session_id}.csv\", mime='text/csv')

# Main chat area
st.subheader(\"대화창 — 고객은 불편 사항이 있는 상황 (친절 응대 필수)\")


def render_history():
    for turn in st.session_state.history[1:]:  # skip system prompt display
        role = turn['role']
        content = turn['content']
        if role == 'user':
            st.markdown(f\"**사용자:** {content}\")
        elif role == 'assistant':
            st.markdown(f\"**Assistant:** {content}\")

render_history()

# User input
user_input = st.text_area(\"메시지 입력\", height=120, placeholder=\"고객의 불편사항을 친절하게 받아 적고, 먼저 상황 파악을 위한 구체적인 질문 3개 이상을 포함하여 응답하도록 유도하세요.\")
if st.button(\"전송\") and user_input.strip():
    # Append user message
    st.session_state.history.append({'role':'user', 'content': user_input.strip()})
    # Build messages for API: convert to chat format expected by SDK
    messages_for_api = [{'role': m['role'], 'content': m['content']} for m in st.session_state.history]
    # Optionally fetch realtime info and append as system note
    realtime_summary = fetch_realtime_info(user_input[:160], serpapi_key)
    if realtime_summary:
        messages_for_api.append({'role':'system', 'content': f\"[실시간 정보 참고]\\n{realtime_summary}\"})

    # Call model with retry logic. If rate-limited beyond retries, preserve last 6 turns and restart session as required.
    try:
        response = call_gemini_chat(api_key=api_key, model=model, messages=messages_for_api)
        # SDK response object handling - try common attributes
        resp_text = None
        try:
            # some SDKs return .content or .message
            if isinstance(response, dict):
                # try to pull common fields
                choices = response.get('choices') or []
                if choices:
                    resp_text = choices[0].get('message', {}).get('content') or choices[0].get('text') or ''
                else:
                    resp_text = response.get('content') or str(response)
            else:
                # attempt attribute access
                resp_text = getattr(response, 'content', None) or getattr(response, 'response', None) or str(response)
        except Exception:
            resp_text = str(response)

        assistant_message = resp_text or \"(응답을 받아오지 못했습니다.)\"
        st.session_state.history.append({'role':'assistant', 'content': assistant_message})

        # Logging
        if enable_csv:
            st.session_state.full_logs.append({'timestamp': int(time.time()), 'role':'user', 'text': user_input.strip()})
            st.session_state.full_logs.append({'timestamp': int(time.time()), 'role':'assistant', 'text': assistant_message})

        st.experimental_rerun()
    except RuntimeError as e:
        # Handle rate-limit exhaustion: preserve last 6 turns and restart session
        st.warning(f\"모델 호출 중 오류가 발생했습니다: {e}\")
        # preserve last 6 entries (excluding system prompt)
        preserved = st.session_state.history[-6:]
        st.session_state.history = [{'role':'system', 'content': DEFAULT_SYSTEM_PROMPT}] + preserved
        st.error(\"요청이 과다하여 세션을 최근 6턴으로 축소한 뒤 재시작했습니다. 잠시 후 다시 시도해주세요.\")


st.markdown(\"---\")
st.caption(\"앱 노트: 이 코드 예시는 google.generativeai SDK 사용을 전제로 합니다. 로컬 실행 전 requirements를 설치하고, 사이드바에 GEMINI API 키(또는 st.secrets['GEMINI_API_KEY'])를 설정하세요.\")

# Display raw session (for debugging)
with st.expander(\"세션/디버그 정보 (토큰 절약용으로 평소 닫아두세요)\"):
    st.write({ 'session_id': st.session_state.session_id, 'history_len': len(st.session_state.history) })
    st.json(st.session_state.history)
