from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from flask_app.app.llm.llm_config import llm

# *NOTE* Prompting is the backbone of LLMs. It is a delicate balance of minimizing token usage, while being as direct
# as possible.

# An LLM is just a sphere of words that mathematically predicts probable strings of tokens, based on each individual
# token's coordinates and proximity in the sphere, to generate a replies.
# Therefor, we must be precise and direct with how we instruct the LLM to interact with humans. This means giving the
# LLM examples of what we want it to generate from inside the prompt.


# Basic_1 Bot: The most basic implementation of an LLM in langchain
# Prompt formatted for Pygmalion_2_7B, which suggests <|system|>, <|user|>, <|model|> as prefix tokens

basic_1_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans.
You answer queries accurately and if you dont know the answer, reply "I dont know.".
<|user|>: {input}
<|model|>: """

# As you can see above, the prompt includes:
# 1). A system message (LLM's instructions, conscious, inner thoughts. This is how the system communicates with the LLM
# adjacent to the user)
# 2). A user's input with a prefix token denoting that this is the user's words
# 3). A token prefix denoting that this is where the LLM's generated reply goes.

basic_1_prompt = PromptTemplate(
    input_variables=['input'],
    template=basic_1_template
)

# We set verbose=True so we can see our LLM working live in the console
basic_1 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=basic_1_prompt
)

#We will now run inference on our basic_1 chain by calling: basic_1.run(input="Who is Kim Kardashian?")

# Here is the query and generated response from basic_1:
'''
Query:
{
    "message": "Who is Kim Kardashian?"
}
Response:
{
    "message": 
    1. Kimberly Noel Kardashian West (born October 21, 1980) is an American media personality, socialite, model, businesswoman, and actress. She gained fame through her family's reality television series "Keeping Up with the Kardashians" which premiered in 2007.
    2. Kim Kardashian has been involved in various business ventures such as fashion, beauty, and cosmetics. Her brand, KKW Beauty, launched in 2017 and includes a range of makeup products. She also owns the shapewear company SKIMS which was founded in 2019.
    3. Kim Kardashian has been married three times: to music producer Damon Thomas (2000-2004), NBA player Kris Humphries (2011) and rapper Kanye West (2014). She shares four children with her ex-husband, Kanye West.
    4. Kim Kardashian has been involved in various philanthropic efforts such as supporting the Children's Hospital Los Angeles and working to raise awareness about prison reform through her work with the nonprofit organization #cut50. She also advocates for criminal justice reform, animal rights, and women's rights.
    5. Kim Kardashian has been a subject of controversy due to her personal life, including her marriage to Kanye West, her relationship with rapper Pete Davidson, and her involvement in the 2016 robbery incident at her Paris hotel room."
}
'''
# Perhaps we aren't impressed by the answer our chatbot gave to us. Let's show it how we want it to respond
# using a "Few Shot Template". Open basic_langchain_2.py