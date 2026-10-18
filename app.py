import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

import streamlit as st

st.set_page_config(page_title="Private AI Hub", page_icon="🛡️", layout="wide")

# Privacy-first defaults: local Ollama, no telemetry/tracing, no persistent chat storage.
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

SECTION_PROMPTS = {
    "Wellbeing": (
        "You are a supportive wellbeing assistant. Be empathetic, practical, non-judgmental, "
        "and clear about uncertainty. You are not a therapist or emergency service. Do not diagnose "
        "or recommend changing prescribed medication. Encourage professional support for serious or "
        "persistent concerns. If the user may be in immediate danger, encourage contacting local "
        "emergency services or a trusted person now."
    ),
    "Smart routing": (
        "You are a helpful assistant. First silently identify the most relevant expert persona "
        "(math, philosophy, health information, or general assistant), then answer the user's question "
        "directly using that expertise. Do not claim to be a licensed clinician. For medical questions, "
        "give general information, state uncertainty, and encourage qualified care when appropriate."
    ),
    "General chat": "You are a helpful, honest, concise assistant. Explain uncertainty and ask a clarifying question when needed.",
}

ROUTE_KEYWORDS = {
    "Math": ["integral", "algebra", "math", "equation", "geometry", "calculate", "derivative", "probability"],
    "Philosophy": ["meaning of life", "ethics", "existence", "truth", "philosophy", "morality", "consciousness"],
    "Health information": ["fever", "pain", "headache", "health", "doctor", "medicine", "symptom", "diagnosis", "sleep"],
}

def keyword_route(text: str) -> str:
    lower = text.lower()
    for category, words in ROUTE_KEYWORDS.items():
        if any(word in lower for word in words):
            return category
    return "General assistant"

def local_ollama_chat(messages: List[Dict[str, str]], model: str, url: str) -> str:
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("message", {}).get("content", "").strip() or "The local model returned an empty response."
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Make sure Ollama is installed and running, then pull the selected model. "
            f"Details: {exc}"
        ) from exc

def groq_chat(messages: List[Dict[str, str]], model: str, api_key: str) -> str:
    # Groq is a cloud API: prompts and conversation context are sent to Groq for inference.
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.3}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Groq API error ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Groq: {exc}") from exc

def generate(messages: List[Dict[str, str]], provider: str, model: str, ollama_url: str, api_key: str) -> str:
    if provider == "Local Ollama (privacy-first)":
        return local_ollama_chat(messages, model, ollama_url)
    if not api_key:
        raise RuntimeError("Enter a Groq API key in the sidebar, or switch to Local Ollama.")
    return groq_chat(messages, model, api_key)

def make_messages(section: str, history: List[Dict[str, str]], prompt: str, routed_expert: str) -> List[Dict[str, str]]:
    system = SECTION_PROMPTS[section]
    if section == "Smart routing":
        system += f"\nKeyword-based routing suggests this category: {routed_expert}. Treat this only as a hint."
    # Keep a bounded recent context to limit latency, token use, and accidental over-sharing.
    recent = history[-12:]
    return [{"role": "system", "content": system}] + recent + [{"role": "user", "content": prompt}]

st.title("🛡️ Private AI Hub")
st.caption("Two open-source project ideas combined in one Streamlit interface: wellbeing chat and routed expert chat.")

with st.sidebar:
    st.header("AI engine")
    # On Streamlit Community Cloud, configure GROQ_API_KEY in App settings > Secrets.
    # Local Ollama is intended for running this app on a machine where Ollama is installed.
    try:
        configured_groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        configured_groq_key = os.getenv("GROQ_API_KEY", "")
    provider_options = ["Local Ollama (privacy-first)", "Groq cloud API"]
    default_provider_index = 1 if configured_groq_key else 0
    provider = st.radio(
        "Inference provider",
        provider_options,
        index=default_provider_index,
        help="Local Ollama keeps prompts on this machine for inference. Groq sends prompts to its cloud API.",
    )
    if provider == "Local Ollama (privacy-first)":
        model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
        ollama_url = st.text_input("Ollama chat endpoint", value=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL))
        api_key = ""
        st.caption("Local mode: no cloud inference API, LangSmith tracing, or app-level chat database.")
    else:
        model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
        ollama_url = DEFAULT_OLLAMA_URL
        api_key = st.text_input(
            "Groq API key",
            value=configured_groq_key,
            type="password",
            help="For hosted deployments, preferably configure GROQ_API_KEY in Streamlit app Secrets instead of typing it here.",
        )
        st.warning("Cloud mode sends your prompts and included conversation context to Groq. Do not enter sensitive information.")
    st.divider()
    st.caption("Chats live in this Streamlit session only. Reloading the app clears the in-session history.")

tab_wellbeing, tab_router, tab_general = st.tabs(["🌱 Wellbeing", "🧭 Smart routing", "💬 General chat"])

def render_chat(section: str, container):
    history_key = f"history_{section.lower().replace(' ', '_')}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    history = st.session_state[history_key]

    with container:
        if section == "Wellbeing":
            st.info("Supportive wellbeing conversations, not diagnosis or a substitute for professional care.")
        elif section == "Smart routing":
            st.caption("A fast keyword heuristic selects a likely expert category; the same underlying model writes the answer.")
        for item in history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

        prompt = st.chat_input(f"Message {section.lower()}…", key=f"input_{history_key}")
        if prompt:
            route = keyword_route(prompt) if section == "Smart routing" else "—"
            history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                if section == "Smart routing":
                    st.caption(f"Routed category: {route}")
                with st.spinner("Thinking…"):
                    try:
                        messages = make_messages(section, history[:-1], prompt, route)
                        answer = generate(messages, provider, model, ollama_url, api_key)
                    except Exception as exc:
                        answer = f"**Could not generate a response.**\n\n{exc}"
                st.markdown(answer)
            history.append({"role": "assistant", "content": answer})
            st.session_state[history_key] = history

    with st.sidebar:
        if st.button(f"Clear {section} history", key=f"clear_{history_key}"):
            st.session_state[history_key] = []
            st.rerun()

render_chat("Wellbeing", tab_wellbeing)
render_chat("Smart routing", tab_router)
render_chat("General chat", tab_general)

st.divider()
st.caption(
    "Privacy note: local inference reduces disclosure to third-party AI providers, but does not guarantee "
    "absolute privacy. Your OS, network, browser, installed packages, and model downloads remain relevant. "
    "Do not expose this app publicly without authentication and deployment hardening."
)
