from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.3
)


question = "糖尿病有哪些常见症状？"


answer = llm.invoke(question)


print(answer.content)