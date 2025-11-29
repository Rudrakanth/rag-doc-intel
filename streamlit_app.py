import streamlit as st

from src.query.search_query import answer_with_search
from src.services.document_manager import (
    get_document_status,
    reindex_document,
    remove_document,
    upload_and_index,
)


st.set_page_config(page_title="Document RAG Console", layout="wide")
st.title("Azure Document Intelligence RAG")
st.caption("Manage contract files, monitor indexing, and chat with the documents.")

st.markdown(
    """
    <style>
    .chat-box {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def reset_chat():
    st.session_state["messages"] = []


def load_documents():
    try:
        return get_document_status()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load documents: {exc}")
        return []


def render_with_collapsible_sources(text: str) -> str:
    """
    Render only the answer body and drop the SOURCES block entirely.
    """
    if not text or "SOURCES:" not in text:
        return text

    body, _ = text.split("SOURCES:", 1)
    return body.strip()


left_col, right_col = st.columns([1.1, 1], gap="large")


with left_col:
    st.subheader("File management")
    st.caption("Upload, index, reindex, or delete documents.")

    with st.form("upload_form", clear_on_submit=True):
        upload = st.file_uploader("Add a PDF", type=["pdf"])
        submitted = st.form_submit_button("Upload and index", use_container_width=True)
        if submitted:
            if not upload:
                st.warning("Please select a PDF to upload.")
            else:
                with st.spinner("Uploading to blob storage, running analysis, and indexing..."):
                    try:
                        upload_and_index(upload, upload.name)
                        st.success(f"Uploaded and indexed {upload.name}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to ingest file: {exc}")

    st.markdown("---")
    st.subheader("Managed documents")
    docs = load_documents()

    if not docs:
        st.info("No documents found yet. Upload a PDF to get started.")
    else:
        header_cols = st.columns([3, 2, 2, 2, 1.5, 1.5])
        header_cols[0].markdown("**Filename**")
        header_cols[1].markdown("**Status**")
        header_cols[2].markdown("**Indexed chunks**")
        header_cols[3].markdown("**Last modified**")
        header_cols[4].markdown("**Reindex**")
        header_cols[5].markdown("**Delete**")

        for doc in docs:
            cols = st.columns([3, 2, 2, 2, 1.5, 1.5])
            cols[0].write(doc.get("filename"))
            cols[1].write(doc.get("status"))
            cols[2].write(doc.get("indexed_chunks"))
            cols[3].write(doc.get("last_modified"))

            if cols[4].button("Reindex", key=f"reindex-{doc.get('blob_name')}"):
                with st.spinner("Reindexing document..."):
                    try:
                        count = reindex_document(
                            doc.get("blob_name"),
                            {"doc_id": doc.get("doc_id"), "original_name": doc.get("filename")},
                            doc.get("filename"),
                        )
                        st.success(f"Reindexed {count} chunks for {doc.get('filename')}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Reindex failed: {exc}")

            if cols[5].button(
                "Delete",
                key=f"delete-{doc.get('blob_name')}",
                type="secondary",
            ):
                with st.spinner("Deleting document..."):
                    try:
                        remove_document(doc.get("blob_name"), doc.get("doc_id"))
                        st.success(f"Deleted {doc.get('filename')}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Delete failed: {exc}")


with right_col:
    st.subheader("Chat with documents")

    control_cols = st.columns([1, 1, 1.5, 1.5])
    with control_cols[-2]:
        if st.button("Clear chat", use_container_width=True):
            reset_chat()
    with control_cols[-1]:
        if st.button("New chat", use_container_width=True):
            reset_chat()

    st.caption("Chat history stays in this session; newest messages appear at the bottom.")

    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for msg in st.session_state["messages"]:
            if msg["role"] == "assistant":
                st.chat_message("assistant").markdown(
                    render_with_collapsible_sources(msg["content"]),
                    unsafe_allow_html=True,
                )
            else:
                st.chat_message(msg["role"]).markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask a question about the indexed contracts")
    if prompt:
        history = list(st.session_state["messages"])
        history.append({"role": "user", "content": prompt})

        with chat_container:
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching and drafting answer..."):
                    try:
                        result = answer_with_search(prompt, chat_history=history)
                        answer_text = result["answer"]
                    except Exception as exc:  # noqa: BLE001
                        answer_text = f"Error while answering: {exc}"

                    st.markdown(
                        render_with_collapsible_sources(answer_text),
                        unsafe_allow_html=True,
                    )

        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["messages"].append({"role": "assistant", "content": answer_text})
