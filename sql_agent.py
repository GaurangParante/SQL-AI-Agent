from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# =========================
# DATABASE
# =========================

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (
        status IN ('pending','in_progress','completed')
    ) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# =========================
# LLM
# =========================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)

tools = toolkit.get_tools()

# =========================
# SYSTEM PROMPT
# =========================

system_prompt = """
You are a task management assistant thet interacts with a sql database containing a 'tasks' table.

TASK RULES:
1. LIMIT SELECT queries to 10 result max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, Confirm with SELECT query.
3. If the user request a list of task, present the output in a structured table format to ensure a clean and organized display in the browser.

CRUD OPERATION:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks SET status=? WHERE id=? or title=?
    UPDATE: UPDATE tasks SET status=? WHERE id=? or title=? 
    DELETE: DELETE FROM tasks WHERE id=? or title=?

Table schema: id, title, description, status(pending,in_progress,completed), created_at.
"""

# =========================
# AGENT
# =========================

@st.cache_resource
def get_agent():
    return create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )

agent = get_agent()

# =========================
# UI
# =========================

st.set_page_config(
    page_title="TaskBot",
    page_icon="✅"
)

st.title("✅ TaskBot - Manage Your Tasks")

# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================

prompt = st.chat_input("Ask me to manage your tasks...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):

            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "1"
                        }
                    }
                )

                result = response["messages"][-1].content

                st.markdown(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })

            except Exception as e:
                st.error(f"Error: {e}")
                print(e)