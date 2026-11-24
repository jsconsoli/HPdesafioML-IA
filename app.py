"""Interface conversacional mínima em Streamlit."""

import streamlit as st

from gpu_assistant.agent import GpuAgent
from gpu_assistant.config import get_model_name
from gpu_assistant.database import GpuDatabase
from gpu_assistant.tools import GpuQueryTools


def main() -> None:
    st.set_page_config(page_title="GPU Assistant", page_icon="🖥️", layout="centered")
    model = get_model_name()
    with st.sidebar:
        st.title("GPU Assistant", anchor=False)
        st.caption("Assistente de consultas à disponibilidade simulada de GPUs")
        st.html(
            """
            <style>
                .st-key-sidebar-info [data-testid="stCaptionContainer"] {
                    color: inherit;
                    color: light-dark(currentColor, #fff);
                    opacity: 1;
                }
            </style>
            """
        )
        sidebar_info = st.container(key="sidebar-info")
        sidebar_info.caption(f"Modelo em uso (Ollama): {model}")
        if st.button("Limpar conversa"):
            st.session_state.pop("chat", None)
            st.session_state.pop("agent_history", None)
            st.rerun()

    if st.session_state.get("last_model") != model:
        st.session_state["last_model"] = model
        st.session_state["chat"] = []
        st.session_state["agent_history"] = []

    database = GpuDatabase()
    if not database.db_path.is_file():
        st.error("Banco não inicializado. No terminal, execute `python init_db.py`.")
        st.stop()

    try:
        summary = database.get_summary()
    except Exception as exc:
        st.error(f"Não foi possível consultar a base de GPUs: {exc}")
        st.stop()

    sidebar_info.caption(f"Modelos cadastrados: {summary['modelos']}")
    sidebar_info.caption(f"GPUs disponíveis: {summary['total_disponivel']}")

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("agent_history", [])
    for message in st.session_state["chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    composer = st
    # No primeiro envio, o valor recebido já permite renderizar o campo no rodapé.
    if not st.session_state["chat"] and not st.session_state.get("chat_prompt"):
        composer = st.container(key="chat-welcome")
        composer.html(
            """
            <style>
                .stMainBlockContainer:has(.st-key-chat-welcome) {
                    margin-block: auto;
                    padding-block: 4rem;
                }
                .st-key-chat-welcome h2 {
                    text-align: center;
                    padding: 0;
                }
            </style>
            <h2>O que gostaria de saber sobre GPUs hoje?</h2>
            """
        )

    prompt = composer.chat_input(
        "Pergunte sobre modelos, memória ou disponibilidade...", key="chat_prompt"
    )
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                try:
                    agent = GpuAgent(model_name=model, query_tools=GpuQueryTools(database))
                    response, new_history = agent.answer(
                        prompt, st.session_state["agent_history"]
                    )
                    st.session_state["agent_history"] = new_history
                except Exception:
                    response = (
                        "Não foi possível conectar ao modelo. Verifique se o Ollama "
                        "está em execução e se o modelo informado foi baixado."
                    )
            st.markdown(response)
        st.session_state["chat"].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
