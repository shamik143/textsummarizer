
import streamlit as st
from summarizer import clean_text, extractive_summary, abstractive_summary
from utils import read_pdf, scrape_url

# Page config
st.set_page_config(page_title="🧠 Text Summarizer", layout="wide")
st.title("🧠 Intelligent Text Summarizer")
st.markdown("Upload a **PDF**, enter a **URL**, or paste **text**. Choose a summarization mode below:")

# --- Setup session state ---
if "final_input" not in st.session_state:
    st.session_state.final_input = ""
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "last_type" not in st.session_state:
    st.session_state.last_type = ""

# --- UI Controls ---
mode = st.selectbox("Select Summarization Type", ["Abstractive", "Extractive"])
max_len = st.slider("Max Summary Length (Abstractive only)", 50, 250, 130)

text_input = st.text_area("✍️ Enter Text to Summarize", height=200)
pdf_file = st.file_uploader("📄 Or Upload a PDF", type=["pdf"])
url_input = st.text_input("🌐 Or Enter a Web Article URL")

# --- On Summarize ---
if st.button("Summarize"):
    input_text = ""

    # Priority: PDF > URL > Text
    if pdf_file is not None:
        try:
            input_text = read_pdf(pdf_file)
        except Exception as e:
            st.error(f"PDF error: {e}")
            st.stop()
    elif url_input.strip():
        try:
            input_text = scrape_url(url_input.strip())
        except Exception as e:
            st.error(f"URL error: {e}")
            st.stop()
    elif text_input.strip():
        input_text = text_input.strip()
    else:
        st.warning("❗ Please provide some input.")
        st.stop()

    if not input_text.strip():
        st.warning("⚠️ No readable content found.")
        st.stop()

    st.session_state.final_input = input_text
    st.session_state.last_type = mode

    cleaned = clean_text(input_text)
    with st.spinner("🧠 Generating summary..."):
        try:
            if mode == "Abstractive":
                summary = abstractive_summary(cleaned, max_len)
            else:
                summary = extractive_summary(cleaned)

            st.session_state.last_summary = summary
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            st.stop()

# --- Show Summary if Available ---
if st.session_state.last_summary:
    wc = len(st.session_state.final_input.split())
    st.info(f"📖 Original Text Length: {wc} words ({st.session_state.last_type} Summary)")
    st.subheader("📋 Summary:")
    st.success(st.session_state.last_summary)

    with st.expander("📄 Show Original Text"):
        st.write(st.session_state.final_input)

    if st.button("🔁 Clear All"):
        st.session_state.final_input = ""
        st.session_state.last_summary = ""
        st.session_state.last_type = ""
        st.success("✅ Cleared. You can start fresh.")
