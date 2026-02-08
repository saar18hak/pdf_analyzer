import os
import streamlit as st
import hashlib
from pathlib import Path
import re
import json
from collections import Counter

import pytesseract
from pdf2image import convert_from_path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

os.environ["GROQ_API_KEY"] = "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"

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------------------
# SESSION STATE
# ---------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "docs" not in st.session_state:
    st.session_state.docs = None

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# ---------------------------
# HELPERS
# ---------------------------
def hash_pdf(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_with_ocr(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    docs = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs

# ---------------------------
# FALLBACK SKILL EXTRACTION
# ---------------------------
DEFAULT_SKILL_GROUPS = {
    "Frontend": ["html", "css", "javascript", "react"],
    "Backend": ["node", "express", "python", "flask", "django"],
    "Database": ["mysql", "mongodb", "sql"],
    "AI/ML": ["machine learning", "deep learning", "ai"],
    "Cloud & DevOps": ["aws", "azure", "docker", "git", "github"],
    "Soft Skills": ["communication", "teamwork", "leadership", "problem solving"]
}

def fallback_extract_skills(docs):
    text = " ".join(d.page_content.lower() for d in docs)
    found = {group: [] for group in DEFAULT_SKILL_GROUPS}
    for group, skills in DEFAULT_SKILL_GROUPS.items():
        for skill in skills:
            if re.search(rf"\b{re.escape(skill)}\b", text):
                found[group].append(skill)
    return found

def flatten_grouped_skills(grouped_skills):
    flat = []
    if isinstance(grouped_skills, dict):
        for skills in grouped_skills.values():
            if isinstance(skills, list):
                flat.extend(skills)
            elif isinstance(skills, str):
                flat.append(skills)
    elif isinstance(grouped_skills, list):
        flat.extend(grouped_skills)
    elif isinstance(grouped_skills, str):
        flat.append(grouped_skills)
    return flat

# ---------------------------
# AI SKILL EXTRACTION
# ---------------------------
def ai_extract_skills(docs):
    text = " ".join(d.page_content.strip() for d in docs)
    if not text or len(text) < 20:
        return fallback_extract_skills(docs)

    prompt = f"""
Extract all technical and non-technical skills from the resume below.
Return a valid JSON dictionary ONLY, with keys as skill groups (like "Frontend", "Backend", "Cloud", "AI/ML", "Soft Skills") 
and values as lists of skills. Do NOT include any text outside JSON.

Resume Text:
{text}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Attempt to extract JSON from any text
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]

        skills_grouped = json.loads(json_text)
        if not isinstance(skills_grouped, dict):
            return fallback_extract_skills(docs)
        return skills_grouped

    except Exception as e:
        st.warning(f"AI skill extraction failed, using fallback: {e}")
        return fallback_extract_skills(docs)

# ---------------------------
# ATS SCORE
# ---------------------------
def calculate_ats_score(skills_grouped):
    total_skills = sum(len(v) for v in DEFAULT_SKILL_GROUPS.values())
    matched = len(flatten_grouped_skills(skills_grouped))
    return min(100, int((matched / total_skills) * 100))

# ---------------------------
# FAISS + RAG
# ---------------------------
def create_or_load_faiss(pdf_bytes):
    embeddings = load_embeddings()
    temp_pdf = Path("temp.pdf")
    temp_pdf.write_bytes(pdf_bytes)

    loader = PyPDFLoader(str(temp_pdf))
    docs = loader.load()
    if sum(len(d.page_content.strip()) for d in docs) < 500:
        docs = load_pdf_with_ocr(str(temp_pdf))

    st.session_state.docs = docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    temp_pdf.unlink()
    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("""
Answer strictly using the provided context.

Context:
{context}

Question:
{input}

Answer:
""")
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📄 Upload Resume PDF")
    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Indexing resume..."):
            vectorstore = create_or_load_faiss(uploaded_file.getvalue())
            st.session_state.rag_chain = build_rag_chain(vectorstore)

    if st.button("📊 Resume Skill Extraction Dashboard"):
        st.session_state.show_dashboard = not st.session_state.show_dashboard

# ---------------------------
# DASHBOARD
# ---------------------------
if st.session_state.show_dashboard and st.session_state.docs:
    st.subheader("📊 Resume Skill Extraction Dashboard")

    grouped_skills = ai_extract_skills(st.session_state.docs)
    flat_skills = flatten_grouped_skills(grouped_skills)
    ats_score = calculate_ats_score(grouped_skills)

    st.metric("ATS Resume Score", f"{ats_score} / 100")

    if flat_skills:
        st.subheader("Grouped Skills")
        for group, skills in grouped_skills.items():
            if skills:
                st.write(f"**{group}:** {', '.join(skills)}")

        st.subheader("All Detected Skills")
        st.dataframe([{"Skill": s} for s in flat_skills])
        st.bar_chart(flat_skills)

        # ---------------------------
        # Suggest Job Role dynamically
        # ---------------------------
        role = "software developer"
        if any(s in flat_skills for s in ["python", "machine learning", "deep learning", "ai"]):
            role = "data scientist"
        elif any(s in flat_skills for s in ["react", "javascript", "html", "css", "node"]):
            role = "full stack developer"
        elif any(s in flat_skills for s in ["aws", "azure", "docker"]):
            role = "cloud engineer"

        st.subheader("🔗 Job Opportunities")
        st.markdown(f"""
- [LinkedIn Jobs for {role}](https://www.linkedin.com/jobs/search/?keywords={role.replace(' ', '%20')})
- [Naukri Jobs for {role}](https://www.naukri.com/{role.replace(' ', '-')}-jobs)
        """)
    else:
        st.info("No skills detected.")

# ---------------------------
# CHAT UI
# ---------------------------
# ---------------------------
# CHAT UI
# ---------------------------
if uploaded_file and "rag_chain" in st.session_state:
    rag_chain = st.session_state.rag_chain

    # Display chat history
    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)

    def submit_query():
        q = st.session_state.input_text.strip()
        if q:
            st.session_state.pending_query = q
        st.session_state.input_text = ""

    st.text_input(
        "Ask something about your resume",
        key="input_text",
        on_change=submit_query
    )

    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None

        # Save user message
        st.session_state.chat.append(("user", q))

        # Build conversation context
        history_text = ""
        for role, msg in st.session_state.chat:
            history_text += f"{role}: {msg}\n"

        # ✅ CORRECT RAG INVOCATION
        try:
            response = rag_chain.invoke({"input": history_text})
            answer = response["answer"]
        except Exception as e:
            answer = f"AI failed to answer: {e}"

        st.session_state.chat.append(("assistant", answer))
        st.rerun()

