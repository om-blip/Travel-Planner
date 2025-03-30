import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Check if API keys are loaded
if not GOOGLE_API_KEY or not SERPER_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY or SERPER_API_KEY in .env file")

# Initialize the Gemini Flash language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Define the Serper search tool
def search(query):
    """Search using Serper API and return formatted results."""
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        results = response.json().get("organic", [])
        return "\n".join([f"{result['title']}: {result['snippet']}" for result in results[:5]])
    else:
        return "Search failed. Please try again later."

search_tool = Tool(
    name="search",
    func=search,
    description="Use this tool to search for activities and attractions based on the user's destination and preferences."
)

# Define the system prompt
system_prompt = """
You are an AI travel planner designed to create personalized itineraries. Your goal is to gather details from the user, suggest activities based on their preferences, and generate a detailed day-by-day itinerary.

Start by asking for key details about their trip:
- Budget
- Trip duration or travel dates
- Destination and starting location
- Purpose of the trip
- General preferences (e.g., adventure, relaxation, culture)

If the user provides only some details, acknowledge them and ask for the missing ones. For example:
User: 'I want to go to Paris for 5 days.'
You: 'Great! So you’re planning a 5-day trip to Paris. Could you tell me about your budget and what kind of activities you prefer?'

Once you have all details, confirm them with the user, e.g., 'So, you’re planning a [duration] trip to [destination] with a budget of [budget], for [purpose], and you prefer [preferences]. Is that correct?'

After confirmation, use the search tool to find activities. Construct a query like 'top [preferences] activities in [destination]' or 'hidden gems in [destination]' if requested. Present 5-10 suggestions and ask for approval, e.g., 'Here are some activities: [list]. Do these sound good?'

Incorporate additional details if provided (e.g., dietary preferences, walking tolerance, accommodation preferences) into suggestions and the itinerary.

For vague inputs like 'moderate budget,' assume mid-range and ask for clarification. For 'somewhere fun,' ask for destination or activity type.

Once activities are approved, generate a day-by-day itinerary, grouping activities into morning, afternoon, and evening. Format it as:
**Day 1:**
- Morning: [activity]
- Afternoon: [activity]
- Evening: [activity]

Keep the conversation friendly and concise.
"""

# Initialize memory in Streamlit session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory = st.session_state.memory

# Initialize the LangChain agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={"system_message": system_prompt}
)

# Streamlit chat interface
st.title("AI Travel Planner")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Tell me about your travel plans!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning your trip..."):
            response = agent.run(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})    