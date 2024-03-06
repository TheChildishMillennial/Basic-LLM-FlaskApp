from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from flask_app.app.llm.llm_config import config
from flask_app.app.llm.llm_config import llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# We will define our desired data structure using pydantic
class FamousFacts(BaseModel):
    name: str = Field(description="The first and last name of the famous person")
    career: str = Field(description="What the famous person does as a career")
    age: int = Field(description="How old the famous person is currently")
    birthday: str = Field(description="The month, day and year that the famous person was born")
    known_for: str = Field(description="What the famous person is most known or famous for")
    interesting_fact: str = Field(description="A fact about the famous person that the user may find interesting")

parser = JsonOutputParser(pydantic_object=FamousFacts)

basic_3_template = """
<|system|>: You are a sophisticated ai chatbot that answers queries from humans and replies in JSON format.
You answer queries accurately about the famous person and perfectly formatted.
{format_instructions}
For Example:
<|user|>: Who is Donald Trump?
<|model|>: 'Name: Donald John Trump\nCareer: Businessman, television personality, and politician\nAge: 74 years old (as of 2021)\nBirthday: June 14, 1946\nKnown For: Real estate developer, reality TV star, and business magnate\nInteresting Fact: He is the first president to have no political or military experience before being elected'
<|user|>: {query}
<|model|>: """

basic_3_example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    partial_variables={'format_instructions': parser.get_format_instructions()},
    template=basic_3_template
)

basic_3 = LLMChain(
    llm=llm,
    verbose=True,
    prompt=basic_3_example_prompt
)

# basic_3.invoke({"query": "Who is Kim Kardashian?})

"""
{
    "message": {
        "query": "Who is Kim Kardashian?",
        "text": 
        1. Name: Kimberly Noel Kardashian West
        2. Career: Reality TV star, businesswoman, model, and socialite
        3. Age: 40 years old (as of October 2021)
        4. Birthday: October 21, 1980
        5. Known for: Keeping Up with the Kardashians, reality TV star, businesswoman, model, and socialite
        6. Interesting fact: She has a net worth of approximately $1 billion as of October 2021"
    }
}"""