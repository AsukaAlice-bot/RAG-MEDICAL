import re


def clean_rewritten_query(text):
    """
    清理大模型返回的查询文本，只保留一条可直接用于检索的查询。
    """
    if not text:
        return ""

    text = str(text).strip()

    # 移除 Markdown 代码块和反引号
    text = text.replace("```text", "")
    text = text.replace("```", "")
    text = text.replace("`", "")

    # 移除常见前缀
    prefixes = [
        "改写结果：",
        "改写结果:",
        "检索查询：",
        "检索查询:",
        "查询：",
        "查询:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # 合并多余空白和换行
    text = re.sub(r"\s+", " ", text).strip()

    # 移除首尾引号
    text = text.strip("\"'“”‘’")

    return text


def rewrite_query(llm, question):
    """
    将用户的口语化医疗问题改写成适合医疗知识库检索的专业查询。

    Args:
        llm: 已初始化的 LangChain 大模型对象。
        question: 用户原始问题。

    Returns:
        改写后的查询。若问题为空、模型异常或结果为空，则返回原问题。
    """
    original_question = str(question or "").strip()

    if not original_question:
        return ""

    prompt = f"""
你是医疗知识库的查询改写器。

请把用户问题改写为一条适合医学指南、临床教材和诊疗文档检索的中文查询。

要求：
1. 保留用户原始意图，不增加原问题中没有的限定条件。
2. 将口语化表达转换为常见医学术语。
3. 可以补充常见同义词，但不要回答问题。
4. 只输出一条改写后的查询，不要解释，不要列点。
5. 输出尽量简洁，建议不超过80个汉字。
6. 如果原问题已经清晰专业，可以直接原样输出。

用户问题：
{original_question}
""".strip()

    try:
        response = llm.invoke(prompt)
        rewritten = clean_rewritten_query(response.content)

        if not rewritten:
            return original_question

        return rewritten

    except Exception as exc:
        print(f"查询改写失败，已使用原始问题：{exc}")
        return original_question
