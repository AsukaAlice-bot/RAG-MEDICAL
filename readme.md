完成 $py$ 虚拟环境创建

```python
.\venv\Scripts\activate // 激活虚拟环境

.\backend\venv\Scripts\activate
```

创建data文件，来源中国糖尿病指南

接入通义千问大模型

```
OPENAI_API_KEY=sk-ws-H.EDDXEYD.fxRq.MEQCIH7WSfu3XK7Vlf7KyfCi9N3OASLCYerDa0t9UDCCJIExAiAyk100RDs1WpgOKYGohD9sSLgYTuZZVcW23HcpOIplqw
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

创建 **build_knowledge.py** ，读取PDF文档，完成文档切割



创建向量库代码 **create_vectorstore.py **，生成 **vectorstore**

创建  **rag_chat** ,构建rag问答链

创建 **api.py** ，初步建立前端页面

```
uvicorn api:app --reload

http://127.0.0.1:8000/docs
```
成功接入 $reranker$ ,根据回答排序

实现查询改写

