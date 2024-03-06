from flask_app.app.llm.llm_config import llm
from langchain.memory import ConversationKGMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

memory_4_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately about the famous person. Keep your answer short and to the point.
Relevant Information:
{history}
Conversation:
<|user|>: {query}
<|model|>: """

# Entity Memory requires an LLM input so that it can iterate over the conversation to find relevant information
memory = ConversationKGMemory(memory_key="history", llm=llm)

memory_4_example_prompt = PromptTemplate(
    input_variables=["history", "query"],
    template=memory_4_template
)

memory_4 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=memory_4_example_prompt,
    memory=memory
)

"""
Message 1:
{
    "message": {
        "history": "",
        "query": "Who is Kim Kardashian?",
        "text": " Kim Kardashian West is an American media personality, socialite, businesswoman, model, and actress. She
         gained fame through her family's reality television series "Keeping Up with the Kardashians" (2007-present)."
    }
}
Message 2:
{
    "message": {
        "history": "On Kim Kardashian: Kim Kardashian is a media personality. Kim Kardashian is an American socialite. 
        Kim Kardashian is a businesswoman. Kim Kardashian is a model. Kim Kardashian is an actress)
        END OF EXAMPL.",
        "query": "Is she related to anyone famous?",
        "text": " Yes, Kim Kardashian is related to the famous personality OJ Simpson."
    }
}
"""

# SIDE NOTE**** Did anyone else laugh at this? -> Yes, Kim Kardashian is related to the famous personality OJ Simpson.