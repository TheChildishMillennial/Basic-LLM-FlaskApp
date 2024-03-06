from flask_app.app.llm.llm_config import llm
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

memory_1_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately about the famous person.
Conversation History:
{chat_history}
<|user|>: {query}
<|model|>: """

memory = ConversationBufferMemory(memory_key="chat_history")

memory_1_example_prompt = PromptTemplate(
    input_variables=["chat_history", "query"],
    template=memory_1_template
)

memory_1 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=memory_1_example_prompt,
    memory=memory
)

# memory_1.invoke({"query": "Who is Kim Kardashian?"})

"""
{
    "message": {
        "chat_history": "Human: Who is Kim Kardashian?
        AI:  Kim Kardashian West, born on October 21, 1980, is an American media personality, socialite, model, 
        businesswoman, and lawyer. She gained widespread fame through her family's reality television series 
        "Keeping Up with the Kardashians," which premiered in 2007.
        
        Kim has been married three times: first to music producer Damon Thomas (2000-2004), then to NBA player Kris 
        Humphries (2011) for just 72 days, and finally to rapper Kanye West (2014-present). She has four children with 
        her current husband: North, Saint, Chicago, and Psalm.
        
        Kim is known for her successful business ventures, including the cosmetics company KKW Beauty, which she 
        launched in 2017. She also owns a shapewear brand called SKIMS and has appeared on various television shows, 
        such as "Dancing with the Stars" and "Family Feud."
        
        In addition to her business pursuits, Kim is an advocate for criminal justice reform and has worked closely 
        with President Donald Trump's administration on prison reform initiatives. She also supports various charitable 
        causes, including children's health organizations and animal welfare groups.",
        "query": "Who are we talking about?",
        "text": " We are discussing Kim Kardashian West, an American media personality, socialite, model, businesswoman, 
        and lawyer."
    }
}
"""