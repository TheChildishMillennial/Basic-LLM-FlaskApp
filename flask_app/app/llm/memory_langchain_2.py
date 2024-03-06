from flask_app.app.llm.llm_config import llm
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

memory_2_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately about the famous person.
Conversation History:
{chat_history}
<|user|>: {query}
<|model|>: """

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)

memory_2_example_prompt = PromptTemplate(
    input_variables=["chat_history", "query"],
    template=memory_2_template
)

memory_2 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=memory_2_example_prompt,
    memory=memory
)

# memory_2.invoke({"query": "Who is Kim Kardashian?"})

"""
Message 1:
{
    "message": {
        "chat_history": "",
        "query": "Who is Kim Kardashian?",
        "text": " Kim Kardashian West, born on October 21, 1980, is an American media personality, socialite, model, 
        businesswoman, and lawyer. She gained widespread fame through her family's reality television series "Keeping Up
         with the Kardashians," which premiered in 2007.
         
         Kim has been married three times: first to music producer Damon Thomas (2000-2004), then to NBA player Kris 
         Humphries (2011) for just 72 days, and finally to rapper Kanye West (2014-present). She has four children with 
         her current husband: North, Saint, Chicago, and Psalm.
         
         Kim is known for her successful business ventures, including the cosmetics company KKW Beauty, which she 
         launched in 2017. She also owns a shapewear brand called SKIMS and has appeared on various television shows, 
         such as "Dancing with the Stars" and "Family Feud."
         
         In addition to her business pursuits, Kim is an advocate for criminal justice reform and has worked closely 
         with President Donald Trump's administration on prison reform initiatives. She also supports various charitable 
         causes, including children's health organizations and animal welfare groups."
    }
}

Message 2:
{
    "message": {
        "chat_history":
        "Human: Who is Kim Kardashian?
        AI:  Kim Kardashian West, born on October 21, 1980, is an American media personality, socialite, model, 
        businesswoman, and lawyer. She gained widespread fame through her family's reality television series "Keeping Up
        with the Kardashians," which premiered in 2007.
         
        Kim has been married three times: first to music producer Damon Thomas (2000-2004), then to NBA player Kris 
        Humphries (2011) for just 72 days, and finally to rapper Kanye West (2014-present). She has four children with 
        her current husband: North, Saint, Chicago, and Psalm.
         
        Kim is known for her successful business ventures, including the cosmetics company KKW Beauty, which she 
        launched in 2017. She also owns a shapewear brand called SKIMS and has appeared on various television shows, 
        such as "Dancing with the Stars" and "Family Feud."
         
        In addition to her business pursuits, Kim is an advocate for criminal justice reform and has worked closely
        with President Donald Trump's administration on prison reform initiatives. She also supports various charitable 
        causes, including children's health organizations and animal welfare groups.",
        
        "query": "how many times has she been married?",
        "text": "3 times"
    }
}

Message 3:
{
    "message": {
        "chat_history": "Human: Who is Kim Kardashian?
        AI:  Kim Kardashian West, born on October 21, 1980, is an American media personality, socialite, model, 
        businesswoman, and lawyer. She gained widespread fame through her family's reality television series "Keeping Up
        with the Kardashians," which premiered in 2007.
         
        Kim has been married three times: first to music producer Damon Thomas (2000-2004), then to NBA player Kris 
        Humphries (2011) for just 72 days, and finally to rapper Kanye West (2014-present). She has four children with 
        her current husband: North, Saint, Chicago, and Psalm.
         
        Kim is known for her successful business ventures, including the cosmetics company KKW Beauty, which she 
        launched in 2017. She also owns a shapewear brand called SKIMS and has appeared on various television shows, 
        such as "Dancing with the Stars" and "Family Feud."
         
        In addition to her business pursuits, Kim is an advocate for criminal justice reform and has worked closely 
        with President Donald Trump's administration on prison reform initiatives. She also supports various charitable
        causes, including children's health organizations and animal welfare groups.
        
        Human: how many times has she been married?        
        AI: 3 times",
        
        "query": "Who is Kanye West?",
        "text": " Kanye Omari West was born on June 8, 1977, in Atlanta, Georgia. He is an American rapper, singer, 
        songwriter, record producer, and fashion designer. West gained mainstream recognition as a producer for 
        Roc-A-Fella Records before releasing his debut album "The College Dropout" in 2004.
        
        nKanye has been married three times: first to designer Alexis Phifer (2002-2013), then to reality television 
        personality Kim Kardashian West (2014-present), and most recently to model Bianca Censori in February 2023. He 
        has seven children with his current wife, Kim: North, Saint, Chicago, Psalm, and three more whose names have not 
        been publicly disclosed.
        
        Kanye is known for his controversial statements and outspoken personality, which often attracts media attention.
        In addition to his music career, he has ventured into fashion design with the clothing line Yeezy and has 
        collaborated with various brands such as Adidas and Gap. He also runs a creative content company called Donda, 
        named after his late mother, which focuses on art, culture, and technology.
        
    }
}

# PAY CLOSE ATTENTION TO THE FIRST ENTRY IN OUR CHAT_HISTORY, IT HAS NOW CHANGED

Message 4:
{
    "message": {
        "chat_history":
        "Human: how many times has she been married?
        AI: 3 times
        Human: Who is Kanye West?
        AI:  Kanye Omari West was born on June 8, 1977, in Atlanta, Georgia. He is an American rapper, singer, 
        songwriter, record producer, and fashion designer. West gained mainstream recognition as a producer for 
        Roc-A-Fella Records before releasing his debut album "The College Dropout" in 2004.
        
        Kanye has been married three times: first to designer Alexis Phifer (2002-2013), then to reality television 
        personality Kim Kardashian West (2014-present), and most recently to model Bianca Censori in February 2023. 
        He has seven children with his current wife, Kim: North, Saint, Chicago, Psalm, and three more whose names have 
        not been publicly disclosed.
        
        Kanye is known for his controversial statements and outspoken personality, which often attracts media attention. 
        In addition to his music career, he has ventured into fashion design with the clothing line Yeezy and has 
        collaborated with various brands such as Adidas and Gap. He also runs a creative content company called Donda, 
        named after his late mother, which focuses on art, culture, and technology.
        
        "query": "are they still married?",
        "text": " Kanye West is currently married to model Bianca Censori. They got married in February 2023, and it's 
        their first marriage together."
    }
}
"""
# Because K is set to 3, our memory will only retain the last 3 interactions. Therefor, on the 4th message, our LLM
# forgets the first interaction we had, in favor of the newest information as context.