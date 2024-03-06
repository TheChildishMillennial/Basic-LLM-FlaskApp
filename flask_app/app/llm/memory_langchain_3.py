from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.prompts import PromptTemplate
from flask_app.app.llm.llm_config import llm, config

# Beware of token limits! Entity memory is heavy in context tokens, so we need to format our prompt properly for short
# answers. We will also adjust our max new tokens so the LLM knows we mean business.

config['max_new_tokens'] = 500

memory_3_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately about the famous person. Keep your answer short and to the point.
Entity Context:
{entities}
<|user|>: {query}
<|model|>: """

# Entity Memory requires an LLM input so that it can iterate over the conversation to find relevant information
memory = ConversationEntityMemory(memory_key="entities", llm=llm)

memory_3_example_prompt = PromptTemplate(
    input_variables=["entities", "query"],
    template=memory_3_template
)

memory_3 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=memory_3_example_prompt,
    memory=memory
)

# memory_3.invoke({"query": "Who is Kim Kardashian?"})

"""
{
    "message": {
        "entities": {
            "Keeping Up with the Kardashians": "",
            "Kim Kardashian": "Kim Kardashian is an American media personality, socialite, model, businesswoman,
            producer, and actress. She gained fame through her family's reality television series "Keeping Up with the 
            Kardashians" (2007-2021)."
        },
        "history": "Human: Who is Kim Kardashian?
        AI:  Kim Kardashian is an American media personality, socialite, model, businesswoman, producer, and actress. 
        She gained fame through her family's reality television series "Keeping Up with the Kardashians" (2007-2021).",
        "query": "is Kim the star of the series?",
        "text": " Yes, Kim Kardashian is one of the stars of the series."
    }
}
"""