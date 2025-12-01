import os
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from google import genai
from google.genai import types

# ------------------------------------------------
# Load environment
# ------------------------------------------------
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Prompt-Driven Email Agent", layout="wide")
st.title("📧 Prompt-Driven Email Productivity Agent — Gemini v0.8.5")

if API_KEY:
    st.success("API key loaded ✔")
else:
    st.error("❌ GEMINI_API_KEY not found in .env")
    st.stop()

# ------------------------------------------------
# Initialize Gemini Client
# ------------------------------------------------
try:
    client = genai.Client(api_key=API_KEY)
    st.success("Gemini client initialized ✔")
except Exception as e:
    st.error(f"❌ Failed to initialize Gemini client:\n{e}")
    st.stop()

# ------------------------------------------------
# Data & Config
# ------------------------------------------------
DATA_DIR = "data"
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"
VALID_CATEGORIES = ["Important", "Newsletter", "Spam", "To-Do"]

# ------------------------------------------------
# Utility functions
# ------------------------------------------------
def load_data(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                txt = f.read().strip()
                if txt:
                    return json.loads(txt)
        except:
            st.warning("⚠ JSON corrupted — restoring defaults")
    return default or {}

def save_data(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    st.success(f"{os.path.basename(path)} saved ✔")


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
            "Categorize the email into exactly one of: Important, Newsletter, Spam, To-Do."
            " Respond with ONLY the category word.",

        "action_item_prompt":
            "Extract action items as JSON: {\"actions\": [{\"task\": \"\", \"deadline\": \"\"}]}."
            " If none, return {\"actions\": []}.",

        "auto_reply_draft_prompt":
            "Draft a short, polite professional reply. Return ONLY the reply body.",

        "chat_agent_system_prompt":
            "You are an AI email assistant. Be concise and prioritize the email content.",

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
            "body": "Please review Q4 budget slides and send feedback by EOD tomorrow.",
            "category": None, "actions": [], "drafts": []
        },
        {
            "id": str(uuid.uuid4()),
            "sender": "Marketing Team <newsletter@promo.net>",
            "subject": "Weekly Update - New Features!",
            "timestamp": (now - timedelta(days=2)).isoformat(),
            "body": "Checkout our new features. Unsubscribe anytime.",
            "category": None, "actions": [], "drafts": []
        },
        {
            "id": str(uuid.uuid4()),
            "sender": "Bob Smith <bob@corp.com>",
            "subject": "Meeting request - Project X",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "body": "Are you available for 15 minutes to discuss Project X this week?",
            "category": None, "actions": [], "drafts": []
        },
    ]


# ------------------------------------------------
# Extract text safely (for v0.8.5)
# ------------------------------------------------
def extract_text(resp):
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # fallback
    try:
        return resp.candidates[0].content.parts[0].text
    except:
        return str(resp)


# ------------------------------------------------
# LLM Calls
# ------------------------------------------------

# ---- Categorization ----
def llm_categorize(email_body, prompt):
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(parts=[types.Part(text=f"{prompt}\n\n{email_body}")])
            ]
        )

        out = extract_text(resp).strip()
        first = out.split("\n")[0].strip().replace(".", "")

        for v in VALID_CATEGORIES:
            if first.lower() == v.lower():
                return v

        st.warning(f"⚠ Unexpected category output:\n{out}")
        return "ERROR_UNCATEGORIZED"

    except Exception as e:
        st.warning(f"Gemini Categorization Error: {e}")
        return "ERROR_UNCATEGORIZED"


# ---- Action Extraction ----
def llm_extract_actions(email_body, prompt):
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(parts=[types.Part(text=f"{prompt}\n\n{email_body}")])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ActionItemList.model_json_schema()
            )
        )

        out = extract_text(resp)
        parsed = ActionItemList.model_validate_json(out)
        return [a.model_dump() for a in parsed.actions]

    except Exception as e:
        st.warning(f"Gemini Action Extraction Error:\n{e}")
        return []


# ---- Chat Agent ----
def llm_chat(system_prompt, email_body, query):
    try:
        full = f"EMAIL:\n{email_body}\n\nUSER REQUEST:\n{query}"

        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(parts=[types.Part(text=full)])],
            config=types.GenerateContentConfig(system_instruction=system_prompt)
        )

        return extract_text(resp)

    except Exception as e:
        st.error(f"Gemini Chat Error:\n{e}")
        return "Sorry, something went wrong."


# ------------------------------------------------
# Session state init
# ------------------------------------------------
if "emails" not in st.session_state:
    st.session_state["emails"] = []

if "prompts" not in st.session_state:
    st.session_state["prompts"] = load_data(PROMPTS_PATH, default_prompts())


# ------------------------------------------------
# Prompt Editing UI
# ------------------------------------------------
st.markdown("---")
with st.expander("🧠 Agent Prompts", True):
    p = st.session_state["prompts"]

    c1, c2, c3 = st.columns(3)
    with c1:
        p_cat = st.text_area("Categorization Prompt", p["categorization_prompt"], height=140)
    with c2:
        p_act = st.text_area("Action Extraction Prompt", p["action_item_prompt"], height=140)
    with c3:
        p_draft = st.text_area("Auto-Reply Prompt", p["auto_reply_draft_prompt"], height=140)

    p_chat = st.text_area("Chat Agent System Prompt", p["chat_agent_system_prompt"])
    p_template = st.text_input("Draft Subject Template", p["draft_subject_template"])

    if st.button("💾 Save Prompts"):
        p["categorization_prompt"] = p_cat
        p["action_item_prompt"] = p_act
        p["auto_reply_draft_prompt"] = p_draft
        p["chat_agent_system_prompt"] = p_chat
        p["draft_subject_template"] = p_template
        save_data(PROMPTS_PATH, p)


# ------------------------------------------------
# Inbox Controls
# ------------------------------------------------
colL, _ = st.columns([1, 5])
with colL:
    if st.button("📥 Load Mock Emails"):
        st.session_state["emails"] = mock_inbox()
        st.success("Loaded mock inbox ✔")

    if st.button("⚙️ Run LLM Pipeline"):
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

            prog.progress((i+1)/total)

        st.success("LLM processing complete ✔")


# ------------------------------------------------
# Inbox List
# ------------------------------------------------
st.header("Inbox")

for email in sorted(st.session_state["emails"], key=lambda x: x["timestamp"], reverse=True):
    ts = datetime.fromisoformat(email["timestamp"])
    category = email["category"] or "Unprocessed"

    icon = "📌"
    if category == "To-Do": icon = "🚨"
    elif category == "Newsletter": icon = "📰"
    elif category == "Spam": icon = "🗑️"
    elif category == "Important": icon = "⚡"

    st.markdown(f"### {icon} {email['subject']}")

    with st.expander("Details"):
        st.write(f"**From:** {email['sender']}  \n**Date:** {ts}")

        st.write("---")
        st.write(email["body"])
        st.write("---")

        if email["actions"]:
            st.subheader("Extracted Action Items")
            for a in email["actions"]:
                st.success(f"📌 Task: {a['task']}  —  Deadline: {a.get('deadline','N/A')}")
        else:
            st.info("No action items.")

        st.subheader("💬 Email Chat Agent")
        query = st.text_area("Ask something about this email:", key=f"q_{email['id']}")

        if st.button("Ask", key=f"ask_{email['id']}"):
            is_draft = "draft" in query.lower()
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
            st.subheader("Saved Drafts")
            for d in email["drafts"]:
                with st.expander(d["subject"]):
                    st.code(d["body"])
