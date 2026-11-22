"""Interface conversacional mínima em Streamlit."""

import streamlit as st

from gpu_assistant.agent import GpuAgent
from gpu_assistant.config import get_model_name
from gpu_assistant.database import GpuDatabase
from gpu_assistant.tools import GpuQueryTools


def main() -> None:
    st.set_page_config(page_title="GPU Assistant", page_icon="🖥️", layout="centered")
    st.title("GPU Assistant")
    st.caption("Assistente de consultas à disponibilidade simulada de GPUs")

    with st.sidebar:
        st.header("Configurações")
        model = st.text_input("Modelo do Ollama", value=get_model_name())
        st.caption("Altere o nome para comparar modelos instalados no Ollama.")
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

    left, right = st.columns(2)
    left.metric("Modelos cadastrados", summary["modelos"])
    right.metric("GPUs disponíveis", summary["total_disponivel"])

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("agent_history", [])
    for message in st.session_state["chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Pergunte sobre modelos, memória ou disponibilidade...")
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
