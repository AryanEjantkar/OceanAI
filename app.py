import os
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import google.generativeai as genai
from google.generativeai import types

# ------------------------------------------------
# Load .env API key
# ------------------------------------------------
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Email Agent — Gemini", layout="wide")
st.title("📧 Prompt-Driven Email Agent — Gemini 2.0/2.5 (SDK v0.8.5)")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env")
    st.stop()

# Configure the SDK
genai.configure(api_key=API_KEY)

# ------------------------------------------------
# LLM Model (high-level)
# ------------------------------------------------
MODEL = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL)

# ------------------------------------------------
# File paths
# ------------------------------------------------
DATA_DIR = "data"
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")
os.makedirs(DATA_DIR, exist_ok=True)

VALID_CATEGORIES = ["Important", "Newsletter", "Spam", "To-Do"]

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def load_data(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                txt = f.read().strip()
                if txt:
                    return json.loads(txt)
        except:
            st.warning("⚠ Corrupt JSON — restoring defaults.")
    return default or {}

def save_data(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    st.success(f"Saved {os.path.basename(path)} ✔")


def extract_text(resp):
    """Unified safe text extractor for SDK 0.8.5"""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    try:
        return resp.candidates[0].content.parts[0].text
    except:
        return str(resp)


# ------------------------------------------------
# Pydantic schemas
# ------------------------------------------------
class ActionItem(BaseModel):
    task: str
    deadline: Optional[str]

class ActionItemList(BaseModel):
    actions: List[ActionItem]


# ------------------------------------------------
# Default prompts
# ------------------------------------------------
def default_prompts():
    return {
        "categorization_prompt":
        "Categorize the email into one category: Important, Newsletter, Spam, To-Do. "
        "Respond with ONLY the category word.",

        "action_item_prompt":
        "Extract action items as JSON matching schema {actions: [{task:'', deadline:''}]}. "
        "If no tasks, return {actions:[]}.",

        "auto_reply_draft_prompt":
        "Draft a short polite reply email. Return ONLY the reply body.",

        "chat_agent_system_prompt":
        "You are an email assistant. Be concise and prioritize email context.",

        "draft_subject_template": "RE: {subject} (DRAFT)"
    }


# ------------------------------------------------
# Mock inbox
# ------------------------------------------------
def mock_inbox():
    now = datetime.now().replace(microsecond=0)
    return [
        {
            "id": str(uuid.uuid4()),
            "sender": "Alice Johnson <alice@corp.com>",
            "subject": "Urgent: Review Q4 Budget Slides",
            "timestamp": now.isoformat(),
            "body": "Please review the Q4 slides and send feedback by EOD tomorrow.",
            "category": None, "actions": [], "drafts": []
        },
        {
            "id": str(uuid.uuid4()),
            "sender": "Team Newsletter <news@corp.com>",
            "subject": "Weekly Product Updates",
            "timestamp": (now - timedelta(days=2)).isoformat(),
            "body": "This week's updates include new features…",
            "category": None, "actions": [], "drafts": []
        },
        {
            "id": str(uuid.uuid4()),
            "sender": "Bob Smith <bob@corp.com>",
            "subject": "Quick call?",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "body": "Can we have a 15-minute sync about Project X?",
            "category": None, "actions": [], "drafts": []
        },
    ]


# ------------------------------------------------
# LLM — Categorization
# ------------------------------------------------
def llm_categorize(body, prompt):
    try:
        resp = model.generate_content(
            f"{prompt}\n\nEMAIL:\n{body}"
        )
        out = extract_text(resp).strip()
        out = out.split("\n")[0].replace(".", "").strip()

        for c in VALID_CATEGORIES:
            if out.lower() == c.lower():
                return c

        return "ERROR_UNCATEGORIZED"
    except Exception as e:
        st.warning(f"Categorization error: {e}")
        return "ERROR_UNCATEGORIZED"


# ------------------------------------------------
# LLM — Action Item Extraction
# ------------------------------------------------
def llm_extract_actions(body, prompt):
    try:
        resp = model.generate_content(
            contents=f"{prompt}\n\nEMAIL:\n{body}",
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ActionItemList.model_json_schema()
            )
        )

        out = extract_text(resp)
        parsed = ActionItemList.model_validate_json(out)
        return [a.model_dump() for a in parsed.actions]

    except Exception as e:
        st.warning(f"Action extraction error: {e}")
        return []


# ------------------------------------------------
# LLM — Chat Agent / Drafts
# ------------------------------------------------
def llm_chat(system_prompt, body, user_query):
    try:
        full = f"SYSTEM:\n{system_prompt}\n\nEMAIL:\n{body}\n\nUSER REQUEST:\n{user_query}"
        resp = model.generate_content(full)
        return extract_text(resp)
    except Exception as e:
        st.error(f"Chat agent error: {e}")
        return "Sorry, something went wrong."


# ------------------------------------------------
# Session state
# ------------------------------------------------
if "emails" not in st.session_state:
    st.session_state["emails"] = []

if "prompts" not in st.session_state:
    st.session_state["prompts"] = load_data(PROMPTS_PATH, default_prompts())


# ------------------------------------------------
# Prompt Config UI
# ------------------------------------------------
st.markdown("---")
with st.expander("🧠 Agent Prompt Configuration", True):
    p = st.session_state["prompts"]

    col1, col2, col3 = st.columns(3)

    with col1:
        p_cat = st.text_area("Categorization Prompt", p["categorization_prompt"], height=140)
    with col2:
        p_act = st.text_area("Action Extraction Prompt", p["action_item_prompt"], height=140)
    with col3:
        p_rep = st.text_area("Draft Reply Prompt", p["auto_reply_draft_prompt"], height=140)

    p_chat = st.text_area("Chat Agent System Prompt", p["chat_agent_system_prompt"])
    p_temp = st.text_input("Draft Subject Template", p["draft_subject_template"])

    if st.button("💾 Save Prompts"):
        p["categorization_prompt"] = p_cat
        p["action_item_prompt"] = p_act
        p["auto_reply_draft_prompt"] = p_rep
        p["chat_agent_system_prompt"] = p_chat
        p["draft_subject_template"] = p_temp
        save_data(PROMPTS_PATH, p)


# ------------------------------------------------
# Inbox Controls
# ------------------------------------------------
colL, _ = st.columns([1, 5])
with colL:
    if st.button("📥 Load Mock Emails"):
        st.session_state["emails"] = mock_inbox()
        st.success("Loaded mock inbox ✔")

    if st.button("⚙️ Run Ingestion Pipeline"):
        emails = st.session_state["emails"]
        total = len(emails)
        prog = st.progress(0)

        for i, email in enumerate(emails):
            body = email["body"]

            email["category"] = llm_categorize(body, p["categorization_prompt"])

            if email["category"] in ["Important", "To-Do"]:
                email["actions"] = llm_extract_actions(body, p["action_item_prompt"])
            else:
                email["actions"] = []

            prog.progress((i + 1) / total)

        st.success("LLM processing complete ✔")


# ------------------------------------------------
# Inbox
# ------------------------------------------------
st.header("📨 Inbox")

for email in sorted(st.session_state["emails"], key=lambda x: x["timestamp"], reverse=True):
    ts = datetime.fromisoformat(email["timestamp"])
    category = email["category"] or "Unprocessed"

    icon = {
        "To-Do": "🚨",
        "Newsletter": "📰",
        "Spam": "🗑️",
        "Important": "⚡",
    }.get(category, "📌")

    st.markdown(f"### {icon} {email['subject']}")

    with st.expander("Details"):
        st.write(f"**From:** {email['sender']}  \n**Date:** {ts}")
        st.write("---")

        st.write(email["body"])
        st.write("---")

        if email["actions"]:
            st.subheader("📌 Action Items")
            for a in email["actions"]:
                st.success(f"**Task:** {a['task']} — **Deadline:** {a.get('deadline','N/A')}")
        else:
            st.info("No action items extracted.")

        st.subheader("💬 Ask the Email Agent")
        query = st.text_area("Ask something about this email:", key=f"q_{email['id']}")

        if st.button("Ask", key=f"ask_{email['id']}"):
            is_draft = "draft" in query.lower() or "reply" in query.lower()
            sys_prompt = p["auto_reply_draft_prompt"] if is_draft else p["chat_agent_system_prompt"]

            reply = llm_chat(sys_prompt, email["body"], query)
            st.code(reply)

            if is_draft:
                email["drafts"].append({
                    "subject": p["draft_subject_template"].format(subject=email["subject"]),
                    "body": reply,
                    "timestamp": datetime.now().isoformat()
                })
                st.success("Draft saved ✔")

        if email["drafts"]:
            st.subheader("📝 Drafts")
            for d in email["drafts"]:
                with st.expander(d["subject"]):
                    st.code(d["body"])
                    st.caption(f"Generated: {d['timestamp']}")
