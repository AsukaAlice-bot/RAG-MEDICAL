import streamlit as st
import requests


st.set_page_config(
    page_title="医疗智能问答系统",
    page_icon="🏥"
)


st.title("🏥 医疗智能问答系统")


question = st.text_input(
    "请输入您的医学问题："
)


if st.button("查询"):

    if question:

        with st.spinner("AI正在分析医学资料..."):

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={
                    "question":question
                }
            )


            result=response.json()


        st.subheader("回答：")

        st.write(
            result["answer"]
        )


        st.subheader("参考来源：")

        for s in result["sources"]:
            st.write(
                "- "+s
            )


    else:

        st.warning(
            "请输入问题"
        )