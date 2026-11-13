"""Minimal Streamlit chat UI."""
import uuid
import streamlit as st
from graph import get_model, make_graph

st.set_page_config(page_title="Weather Advisory Bot", page_icon="🌦️")
st.title("Weather Advisory Bot")
st.caption("Weather guidance from written SOPs. Not a substitute for official alerts.")
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []
    st.session_state.graph = make_graph(get_model())

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        if item.get("details"):
            with st.expander("Why this answer?"):
                for sop in item["details"]["sops"]:
                    st.write(f"{sop['id']} · severity {sop['severity']} · {sop['title']}")
                st.json(item["details"]["facts"])

if prompt := st.chat_input("Ask about an outdoor plan and city"):
    st.session_state.chat.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        result = st.session_state.graph.invoke(
            {"user_text": prompt, "messages": st.session_state.chat[-6:-1], "session_context": st.session_state.get("context", {})},
            {"configurable": {"thread_id": st.session_state.thread_id}},
        )
        reply = result["reply"]
        st.markdown(reply)
        sops = ([result["chosen_sop"]] + result.get("other_sops", [])) if result.get("chosen_sop") else []
        if sops:
            with st.expander("Why this answer?"):
                for sop in sops:
                    st.write(f"{sop['id']} · severity {sop['severity']} · {sop['title']}")
                st.json(result.get("weather_facts", {}))
        st.session_state.chat.append({"role": "assistant", "text": reply, "details": {"sops": sops, "facts": result.get("weather_facts", {})} if sops else None})
        st.session_state.context = result.get("session_context", st.session_state.get("context", {}))
