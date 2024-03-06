from flask_app.app.llm.llm_config import llm
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Perhaps we would like to limit our conversation history with a hard token limit

memory_5_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately about the famous person. Keep your answer short and to the point.
Conversation History:
{chat_history}
Current Conversation:
<|user|>: {query}
<|model|>: """


memory = ConversationTokenBufferMemory(memory_key="chat_history", llm=llm, max_token_limit=60)

memory_5_example_prompt = PromptTemplate(
    input_variables=["history", "query"],
    template=memory_5_template
)

memory_5 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=memory_5_example_prompt,
    memory=memory
)

"""
Message 1:
{
    "message": {
        "chat_history": "",
        "query": "Who is Kim Kardashian?",
        "text": " Kim Kardashian West is an American media personality, socialite, model, businesswoman, producer, and
         actress. She gained fame through her family's reality television series "Keeping Up with the Kardashians" 
         (2007-present)."
    }
}
Message 2:
{
    "message": {
        "chat_history": "Human: Who is Kim Kardashian?
        AI:  Kim Kardashian West is an American media personality, socialite, model, businesswoman, producer, and 
        actress. She gained fame through her family's reality television series "Keeping Up with the Kardashians" 
        (2007-present).",
        "query": "Is she married?",
        "text": " Yes, Kim Kardashian is currently married to rapper and producer Kanye West."
    }
}
Message 3:
{
    "message": {
        "chat_history": "Human: Is she married?
        AI:  Yes, Kim Kardashian is currently married to rapper and producer Kanye West.",
        "query": "are they still married?",
        "text": " No, Kim Kardashian and Kanye West divorced in February 2021 after seven years of marriage."
    }
}
"""

# As you can see utilizing Token Buffer Memory we can command control over our token limitations, when the chat history
# exceeds our token limit, it shaves off the oldest interactions.