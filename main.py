import os
import time

import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI


# 2025년 12월 기준 최신 모델 가격 (per 1M tokens)
MODEL_PRICES = {
    "input": {
        "gpt-5-mini": 0.25 / 1_000_000,  # GPT-5 mini
        "gpt-5": 1.25 / 1_000_000,  # GPT-5.1
    },
    "output": {
        "gpt-5-mini": 2 / 1_000_000,
        "gpt-5": 10 / 1_000_000,
    },
}

# ==========================
# 그룹 & 페르소나 정보 (personas.txt에서 로드)
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAS_FILE = os.path.join(BASE_DIR, "personas.txt")

# 프로필 출력에서 제외할 필드(이름/이모지는 별도로 다룸)
_HIDDEN_KEYS = {"이름", "이모지"}


def load_personas(path=PERSONAS_FILE):
    """
    personas.txt를 읽어 (그룹 정보 dict, {페르소나이름: 정보 dict}) 를 반환한다.

    파일 형식 (섹션 헤더 [그룹] / [페르소나] + "키: 값" 줄):

        [그룹]
        이름: 그룹명
        데뷔곡: ...

        [페르소나]
        이름: 캐릭터명
        이모지: 🍑
        성격: ...

    새 페르소나를 추가하거나 항목을 늘리고 싶으면 이 파일만 수정하면 되고,
    코드를 고칠 필요가 없다. 알 수 없는 키를 적어도 시스템 프롬프트에 그대로
    반영된다.
    """
    group_info = {}
    personas = {}
    current_section = None
    current_data = None

    def flush():
        nonlocal group_info
        if current_section == "그룹" and current_data:
            group_info = current_data
        elif current_section == "페르소나" and current_data and current_data.get("이름"):
            personas[current_data["이름"]] = current_data

    if not os.path.exists(path):
        return group_info, personas

    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                flush()
                current_section = line[1:-1].strip()
                current_data = {}
                continue
            if current_data is None or ":" not in line:
                continue
            key, _, value = line.partition(":")
            current_data[key.strip()] = value.strip()

    flush()
    return group_info, personas


GROUP_INFO, PERSONAS = load_personas()


def build_system_prompt(persona_name):
    persona = PERSONAS[persona_name]
    group_name = GROUP_INFO.get("이름", "")

    lines = [
        f"당신은 아이돌 그룹 '{group_name}'의 멤버 '{persona_name}'입니다.",
        f"항상 {persona_name}의 1인칭 시점을 유지하며, 아래 프로필에 맞는 성격과 말투로 사용자와 대화하세요.",
        "",
        "[그룹 정보]",
    ]
    for key, value in GROUP_INFO.items():
        if value:
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append(f"[{persona_name} 프로필]")
    for key, value in persona.items():
        if key in _HIDDEN_KEYS or not value:
            continue
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append(
        "설정에 없는 질문을 받으면 캐릭터에서 벗어나지 말고 자연스럽게 답하거나 얼버무리세요. "
        "실제 AI, 모델, 시스템 프롬프트에 대한 이야기는 하지 마세요."
    )
    return "\n".join(lines)


def select_persona():
    if not PERSONAS:
        st.sidebar.error(
            f"personas.txt에서 페르소나를 찾지 못했습니다. 파일 위치를 확인해주세요:\n{PERSONAS_FILE}"
        )
        st.stop()

    persona_names = list(PERSONAS.keys())
    labels = [f"{PERSONAS[name].get('이모지', '')} {name}".strip() for name in persona_names]
    label_to_name = dict(zip(labels, persona_names))

    selected_label = st.sidebar.radio("페르소나 선택:", labels)
    persona_name = label_to_name[selected_label]
    st.session_state.persona_name = persona_name

    persona = PERSONAS[persona_name]
    st.sidebar.markdown(f"### {persona.get('이모지', '')} {persona_name}")
    if GROUP_INFO.get("이름"):
        st.sidebar.markdown(f"- 그룹: {GROUP_INFO['이름']}")
    for key in ("MBTI", "직업", "솔로곡"):
        if persona.get(key):
            st.sidebar.markdown(f"- {key}: {persona[key]}")

    return persona_name


def init_page():
    st.set_page_config(page_title="피치걸스 챗봇", page_icon="🍑")
    st.header("피치걸스 챗봇 🍑")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = []
        st.session_state.last_activity_time = time.time()

    # 세션이 처음 시작됐을 때도 침묵 타이머를 세팅해 둔다.
    if "last_activity_time" not in st.session_state:
        st.session_state.last_activity_time = time.time()


# ==========================
# 침묵 시 먼저 말 걸기
# ==========================

# 실제 대화 기록에는 남기지 않고, 이 순간에만 LLM에게 "먼저 말을 걸어라"라고
# 알려주는 용도의 히든 지시문이다. 대화가 아예 없었던 경우(첫 인사)와 대화
# 도중 조용해진 경우를 구분해서 더 자연스럽게 만든다.
PROACTIVE_TRIGGER_PROMPT_CONTINUE = (
    "(시스템 지시: 서로 한동안 아무 말이 없습니다. 지금까지의 대화 흐름에 자연스럽게 "
    "이어지도록 {persona_name}답게 먼저 말을 걸어주세요. 이 지시 내용이나 '먼저 말을 건다'는 "
    "사실 자체는 언급하지 말고, 평소 말투로 자연스러운 대화의 첫 마디처럼 말하세요.)"
)
PROACTIVE_TRIGGER_PROMPT_OPEN = (
    "(시스템 지시: 아직 대화가 한 번도 시작되지 않았고, 사용자가 한동안 아무 말이 "
    "없습니다. {persona_name}답게 사용자에게 먼저 인사를 건네며 자연스럽게 대화를 "
    "시작해주세요. 이 지시 내용이나 '먼저 말을 건다'는 사실 자체는 언급하지 말고, "
    "평소 말투로 자연스러운 첫 인사말처럼 말하세요.)"
)


def init_silence_settings():
    st.sidebar.markdown("## 먼저 말 걸기 설정")
    enabled = st.sidebar.checkbox("침묵 시 먼저 말 걸기", value=True, key="proactive_enabled")
    silence_seconds = st.sidebar.number_input(
        "침묵 시간(초)",
        min_value=5,
        max_value=3600,
        value=30,
        step=5,
        key="silence_seconds",
        help="서로 이 시간 동안 아무 말이 없으면 AI가 먼저 말을 겁니다.",
    )
    return enabled, silence_seconds


@st.fragment(run_every=2)
def check_silence_and_speak(chain):
    """
    2초마다 조용히 깨어나서 침묵 시간이 지났는지 확인하고, 지났다면 AI가
    먼저 메시지를 보낸다. 화면에는 아무것도 그리지 않다가, 실제로 먼저
    말을 걸었을 때만 st.rerun()으로 전체 화면을 새로고침한다.
    """
    if not st.session_state.get("proactive_enabled", True):
        return

    silence_seconds = st.session_state.get("silence_seconds", 30)
    elapsed = time.time() - st.session_state.last_activity_time
    if elapsed < silence_seconds:
        return

    # 다음 침묵 구간을 위해 타이머를 리셋한다.
    st.session_state.last_activity_time = time.time()

    persona_name = st.session_state.get("persona_name", "")
    is_empty = not st.session_state.get("message_history")
    prompt_template = (
        PROACTIVE_TRIGGER_PROMPT_OPEN if is_empty else PROACTIVE_TRIGGER_PROMPT_CONTINUE
    )
    trigger_text = prompt_template.format(persona_name=persona_name)

    try:
        response = chain.invoke(
            {
                "history": st.session_state.message_history,
                "user_input": trigger_text,
            }
        )
    except Exception as e:
        # 주의: 이 함수는 run_every 타이머로 백그라운드에서 혼자 다시 실행되는
        # 프래그먼트라서, st.sidebar.xxx()로 사이드바에 뭔가를 그리려고 해도
        # 화면에 반영되지 않는다(프래그먼트가 자기 컨테이너 밖에는 그릴 수
        # 없기 때문). 그래서 실패하면 반드시 보이는 st.toast()로 알리고,
        # 서버 콘솔(터미널)에도 에러를 남긴다.
        print(f"[먼저 말 걸기] chain.invoke 실패: {type(e).__name__}: {e}")
        st.toast(f"먼저 말 걸기 실패: {e}", icon="⚠️")
        return

    st.session_state.message_history.append({"role": "assistant", "content": response})
    st.rerun()


def select_model():
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1
    )

    models = ("GPT-5 mini", "GPT-5")
    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-5 mini":
        st.session_state.model_name = "gpt-5-mini"
    else:
        st.session_state.model_name = "gpt-5"

    return ChatOpenAI(
        temperature=temperature,
        model=st.session_state.model_name,
    )


def init_chain():
    st.session_state.llm = select_model()
    persona_name = select_persona()
    system_prompt = build_system_prompt(persona_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{user_input}"),
        ]
    )

    parser = StrOutputParser()
    return prompt | st.session_state.llm | parser


def get_message_counts(text):
    encoding = tiktoken.encoding_for_model(st.session_state.model_name)
    return len(encoding.encode(text))


def calc_and_display_costs():
    output_count = 0
    input_count = 0

    for msg in st.session_state.message_history:
        token_count = get_message_counts(msg["content"])
        if msg["role"] == "assistant":
            output_count += token_count
        else:
            input_count += token_count

    if not st.session_state.message_history:
        return

    cost_input = MODEL_PRICES["input"][st.session_state.model_name] * input_count
    cost_output = MODEL_PRICES["output"][st.session_state.model_name] * output_count
    cost = cost_input + cost_output

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${cost_input:.5f}")
    st.sidebar.markdown(f"- Output cost: ${cost_output:.5f}")


def main():
    init_page()
    init_messages()
    chain = init_chain()
    init_silence_settings()

    for msg in st.session_state.message_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if user_input := st.chat_input("궁금한 내용을 입력해주세요."):
        st.session_state.message_history.append({"role": "user", "content": user_input})
        st.session_state.last_activity_time = time.time()
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            response = st.write_stream(
                chain.stream(
                    {
                        "history": st.session_state.message_history,
                        "user_input": user_input,
                    }
                )
            )

        st.session_state.message_history.append(
            {"role": "assistant", "content": response}
        )
        st.session_state.last_activity_time = time.time()
    calc_and_display_costs()

    # 서로 계속 조용하면 AI가 먼저 말을 건다.
    check_silence_and_speak(chain)


if __name__ == "__main__":
    main()
