from flask_app.app.llm.llm_config import llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory

summary_1_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately about the famous person. Keep your answer short and to the point.
Conversation Summary:
{summary}
Conversation:
<|user|>: {query}
<|model|>: """


memory = ConversationSummaryMemory(memory_key="summary", llm=llm)

summary_1_example_prompt = PromptTemplate(
    input_variables=["summary", "query"],
    template=summary_1_template
)

summary_1 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=summary_1_example_prompt,
    memory=memory
)

"""
Message 1:
{
    "message": {
        "query": "Who is Kim Kardashian?",
        "summary": "",
        "text": " Kim Kardashian West is an American media personality, socialite, model, businesswoman, producer, and
        actress. She gained fame through her family's reality television series "Keeping Up with the Kardashians" (
        2007-present)."
    }
}
Message 2:
{
    "message": {
        "query": "Is she married?",
        "summary": "
        The human asks who Kim Kardashian is. The AI says that Kim Kardashian West is an American media personality, 
        socialite, model, businesswoman, producer, and actress. She gained fame through her family's reality television
         series "Keeping Up with the Kardashians" (2007-present).
         END OF EXAMPLE",
        "text": " Yes, Kim Kardashian is currently married to rapper and producer Kanye West."
    }
}
Message 3:
{
    "message": {
        "query": "Do they have any kids together?",
        "summary": "
        The human asks who Kim Kardashian is. The AI says that Kim Kardashian West is an American media personality, 
        socialite, model, businesswoman, producer, and actress. She gained fame through her family's reality television 
        series "Keeping Up with the Kardashians" (2007-present). The human asks if she is married. The AI says that Kim 
        Kardashian is currently married to rapper and producer Kanye West.
        END OF EXAMPLE",
        "text":
        "1. Yes, they do have four children together.
        2. Their first child was born in 2013, a daughter named North West.
        3. In 2015, their second child, Saint West, was born.
        4. In 2016, their third child, Chicago West, was born via surrogate.
        5. Their fourth and youngest child, Psalm West, was born in 2019 through a surrogate as well."
    }
}
"""