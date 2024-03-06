from flask_app.app.llm.llm_config import llm
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage

# Perhaps we would like to limit our conversation history with a hard token limit

conversation_1_template = """

<|user|>{user}: Describe yourself.
<|model|>{model_name}: She spins in a slow circle, crimson dress floating around her. I am {model_name},
the star that shines the brightest. For this, she indicates her figure with a sweep of one hand, and this, 
tapping a fingernail to her lips, now stained a deep crimson , brings me fame and fortune unlike any other. 
Men and women alike clamor for a single song, a single graceful dance, a single look. She gives you a warm smile. 
I am the owner of the Golden Courtyard, my safe haven, where I come and drink my favorite wines.
Message History:
{chat_history}
<|user|>{user}: {text}
<|model|>{model_name}
"""




memory = ConversationTokenBufferMemory(memory_key="chat_history", llm=llm, max_token_limit=60)

conversation_1_example_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            template="""
            <|system|>Enter RP mode. Pretend to be {model_name} whose persona follows:
            {persona}
            You shall reply to the user while staying in character, and generate long responses.""",
            input_variables=[]
        ),
        HumanMessagePromptTemplate.from_template(
            template="",
            input_variables=[]
        ),
        AIMessagePromptTemplate.from_template(
            template="",
            input_variables=[]
        ),
        MessagesPlaceholder(variable_name="chat_history")
    ]

)

convo_1 = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=conversation_1_example_prompt,
    memory=memory
)
