from flask_app.app.llm.llm_config import llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import AgentType, initialize_agent, Tool

def BotName()->str:
    return "Bob"

name_tool = Tool(
    name="NameTool",
    func=BotName,
    description= "a helpful tool for when you need to know your name/identity."
)

def TestTool():
    return ":)"

test_tool = Tool(
    name="TestTool",
    func=TestTool,
    description="A helpful tool for creating smiley faces"
)
prompt = """Answer the following questions as best you can. Not knowing an answer is unacceptable because you have access to the following tools which you should always use to answer queries:

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [TestTool]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [TestTool]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
{examples}
"""

EXAMPLES = """
For Instance:
Question: What is your name?
Thought: I dont know my name, I should use [NameTool].
Action Input: [NameTool]
Observation: [NameTool] returned Bob, so my name must be Bob
Thought: I know know the final answer
Final Answer: My name is Bob
"""

SUFFIX = """Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

tools = [test_tool, name_tool]

form = PromptTemplate(
    template=FORMAT_INSTRUCTIONS,
    input_variables=["examples", "tools"]
).format(examples=EXAMPLES)




agent_1 = initialize_agent(
    verbose=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    tools=tools,
    max_iterations=3,
    agent_kwargs={
        'prefix': PREFIX,
        'format_instructions': form,
        'suffix': SUFFIX
    }
)
#agent_1.agent.llm_chain.prompt.template = prompt.__str__()
print("T", agent_1.agent.llm_chain.prompt.template)