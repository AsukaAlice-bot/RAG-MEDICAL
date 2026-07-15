from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from query_rewrite import rewrite_query


load_dotenv()


llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2
)


test_questions = [
    "糖尿病人脚麻怎么办？",
    "血糖高眼睛看不清是什么问题？",
    "糖尿病患者多久查一次神经病变？",
]


for index, question in enumerate(test_questions, start=1):
    rewritten = rewrite_query(llm, question)

    print("\n" + "=" * 60)
    print(f"测试 {index}")
    print(f"原始问题：{question}")
    print(f"改写查询：{rewritten}")
