# app/services/google_models.py

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain # <-- Add ConversationChain
from langchain.memory import ConversationBufferMemory   # <-- Add Memory
from langchain_google_genai import ChatGoogleGenerativeAI
from .llm_service import IdeaRequest, IdeaResponse

# We need a new Pydantic model for our chat endpoint
from pydantic import BaseModel, Field

load_dotenv()


class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="The user's message to the chatbot.")


class ChatResponse(BaseModel):
    ai_response: str


try:
    google_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", #"gemini-2.5-flash-lite",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )

    idea_template = """
    You are an expert content strategist.
    Your goal is to generate 5 engaging blog post titles for the following topic.
    Topic: {topic}
    Provide the titles as a numbered list.
    """

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=idea_template
    )

    idea_chain = LLMChain(llm=google_llm, prompt=prompt)

except Exception as e:
    print(f"Warning: Could not initialize Google legacy chain. Error: {e}")
    idea_chain = None


def generate_content_ideas(request: IdeaRequest) -> IdeaResponse:
    if not idea_chain:
        raise ConnectionError("Google Legacy Chain is not initialized. Check your API key.")

    try:
        result = idea_chain.invoke({"topic": request.topic})
        return IdeaResponse(ideas=result['text'])
    except Exception as e:
        raise ConnectionError(f"Failed to get response from Google legacy chain: {e}")


# --- NEW: LangChain Conversational Chain with Memory ---

try:
    # We create a single, shared memory object for this simple example.
    # In a real multi-user app, you'd manage one memory object per user session.
    chat_memory = ConversationBufferMemory()

    # The ConversationChain is simpler than LLMChain; it has a default prompt.
    conversation_chain = ConversationChain(
        llm=google_llm,
        memory=chat_memory,
        verbose=True  # Set to True to see the full prompt being sent to the LLM in your terminal
    )

except Exception as e:
    print(f"Warning: Could not initialize Google conversation chain. Error: {e}")
    conversation_chain = None


def get_chat_response(request: ChatRequest) -> ChatResponse:
    """Generates a conversational response using a chain with memory."""
    if not conversation_chain:
        raise ConnectionError("Google Conversation Chain is not initialized.")

    try:
        # The chain's 'predict' method takes the user input and returns the AI's response.
        # It automatically handles loading history, formatting the prompt, and storing the new turn.
        ai_message = conversation_chain.predict(input=request.user_input)
        return ChatResponse(ai_response=ai_message)

    except Exception as e:
        raise ConnectionError(f"Failed to get response from Google conversation chain: {e}")