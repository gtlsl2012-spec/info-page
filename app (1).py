import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
from datetime import datetime

# ---------------------------------------------
# 초기 설정
# ---------------------------------------------
st.set_page_config(page_title="AI 상담 챗봇", page_icon="💬", layout="wide")

# 비밀키 설정 (없을 경우 수동 입력)
if "GEMINI_API_KEY" not in st.secrets:
    api_key = st.text_input("🔑 Gemini API Key를 입력하세요:", type="password")
else:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.warning("Gemini API Key가 필요합니다.")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------
SYSTEM_PROMPT = (
    "1. 사용자의 감정과 고민을 진심으로 공감하며, 따뜻하고 존중하는 말투로 대화하세요.\n"
    "2. 사용자의 상황을 구체적으로 이해하기 위해 언제, 어디서, 어떤 일이 있었는지 자연스럽게 질문하세요.\n"
    "3. 단순한 위로에 그치지 말고, 현실적으로 도움이 될 수 있는 조언이나 방향을 제시하세요."
)

# ---------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_limit" not in st.session_state:
    st.session_state.history_limit = 6

# ---------------------------------------------
# Gemini 호출 함수 (429 재시도 포함)
# ---------------------------------------------
def call_gemini(prompt, history):
    model = genai.GenerativeModel("gemini-2.0-flash")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-st.session_state.history_limit:]
    messages.append({"role": "user", "content": prompt})
    
    for attempt in range(3):
        try:
            response = model.generate_content([m["content"] for m in messages])
            return response.text
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return f"⚠️ 오류 발생: {e}"
    return "⚠️ 재시도 후에도 응답을 받지 못했습니다."

# ---------------------------------------------
# CSV 저장 함수
# ---------------------------------------------
def save_history_csv():
    data = pd.DataFrame(st.session_state.messages)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_log_{timestamp}.csv"
    data.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

# ---------------------------------------------
# 사이드바 UI
# ---------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    st.write(f"모델: gemini-2.0-flash")
    st.write(f"세션 유지 턴 수: {st.session_state.history_limit}")
    if st.button("💾 대화 로그 다운로드"):
        file = save_history_csv()
        st.download_button("CSV 다운로드", data=open(file, "rb"), file_name=file, mime="text/csv")
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.experimental_rerun()

# ---------------------------------------------
# 메인 대화 영역
# ---------------------------------------------
st.title("💬 AI 상담 챗봇")
st.caption("상대의 감정과 고민을 이해하고 따뜻하게 응답합니다.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("지금 어떤 고민이 있으신가요?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = call_gemini(user_input, st.session_state.messages)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
