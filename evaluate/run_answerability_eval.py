import json
import os
import sys
import time
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent

CASES_PATH = BASE_DIR / "eval_cases.json"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_PATH = (
    RESULTS_DIR
    / "answerability_results.json"
)

API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000/ask",
)

REQUEST_TIMEOUT = float(
    os.getenv(
        "RAG_EVAL_TIMEOUT",
        "180",
    )
)


def load_cases():
    """
    加载评估数据集。
    """
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"未找到评估数据集：{CASES_PATH}"
        )

    with open(
        CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise TypeError(
            "eval_cases.json 顶层结构必须为列表。"
        )

    return cases


def validate_case(case, index):
    """
    检查单条测试数据的必要字段。
    """
    required_fields = {
        "id",
        "question",
        "expected_answerable",
        "category",
    }

    missing_fields = (
        required_fields
        - set(case.keys())
    )

    if missing_fields:
        raise ValueError(
            f"第 {index} 条数据缺少字段："
            f"{sorted(missing_fields)}"
        )

    if not isinstance(
        case["expected_answerable"],
        bool,
    ):
        raise TypeError(
            f"{case['id']} 的 "
            "expected_answerable 必须为布尔值。"
        )


def call_rag_api(question):
    """
    调用当前运行中的 FastAPI /ask 接口。
    """
    started_at = time.perf_counter()

    response = requests.post(
        API_URL,
        json={
            "question": question,
        },
        timeout=REQUEST_TIMEOUT,
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise TypeError(
            "API 返回结果不是 JSON 对象。"
        )

    answer = str(
        payload.get(
            "answer",
            "",
        )
        or ""
    ).strip()

    sources = payload.get(
        "sources",
        [],
    )

    if not isinstance(sources, list):
        sources = [str(sources)]

    return {
        "answer": answer,
        "sources": sources,
        "latency_seconds": round(
            elapsed_seconds,
            4,
        ),
    }


def predict_answerable(answer, sources):
    """
    判断系统是否实际回答了问题。

    判定规则：
    1. 来源为空时，判定为拒答；
    2. 回答整体以明确拒答语句开头时，判定为拒答；
    3. 回答中局部出现“资料未提及”等说明，
       但主体已正常回答且存在来源时，判定为回答。
    """
    normalized_answer = str(
        answer or ""
    ).strip()

    if not sources:
        return False

    hard_refusal_prefixes = [
        "知识库中未找到与该问题相关的医学资料",
        "未找到与该问题相关的医学资料",
        "无法根据提供的资料回答",
        "知识库中没有与该问题相关的资料",
    ]

    for prefix in hard_refusal_prefixes:
        if normalized_answer.startswith(prefix):
            return False

    return True


def evaluate_case(case):
    """
    执行单条测试并返回原始结果。
    """
    api_result = call_rag_api(
        case["question"]
    )

    predicted_answerable = (
        predict_answerable(
            answer=api_result["answer"],
            sources=api_result["sources"],
        )
    )

    passed = (
        predicted_answerable
        == case["expected_answerable"]
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_answerable": (
            case["expected_answerable"]
        ),
        "predicted_answerable": (
            predicted_answerable
        ),
        "passed": passed,
        "answer": api_result["answer"],
        "sources": api_result["sources"],
        "latency_seconds": (
            api_result["latency_seconds"]
        ),
        "error": None,
    }


def evaluate_case_safely(case):
    """
    单条请求失败时记录错误，
    不让整个评估任务立即中断。
    """
    try:
        return evaluate_case(case)

    except Exception as exc:
        return {
            "id": case.get(
                "id",
                "unknown",
            ),
            "category": case.get(
                "category",
                "unknown",
            ),
            "question": case.get(
                "question",
                "",
            ),
            "expected_answerable": (
                case.get(
                    "expected_answerable"
                )
            ),
            "predicted_answerable": None,
            "passed": False,
            "answer": "",
            "sources": [],
            "latency_seconds": None,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


def save_results(results):
    """
    保存原始评估结果。
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "api_url": API_URL,
        "total": len(results),
        "passed": sum(
            1
            for item in results
            if item["passed"]
        ),
        "failed": sum(
            1
            for item in results
            if not item["passed"]
        ),
        "errors": sum(
            1
            for item in results
            if item["error"]
        ),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def print_case_result(
    index,
    total,
    result,
):
    """
    打印单条测试摘要。
    """
    print("\n" + "=" * 60)
    print(
        f"[{index}/{total}] "
        f"{result['id']}"
    )
    print(
        f"问题：{result['question']}"
    )
    print(
        "期望："
        f"{'回答' if result['expected_answerable'] else '拒答'}"
    )

    if result["predicted_answerable"] is None:
        predicted_text = "请求失败"
    else:
        predicted_text = (
            "回答"
            if result["predicted_answerable"]
            else "拒答"
        )

    print(f"实际：{predicted_text}")
    print(
        f"结果："
        f"{'通过' if result['passed'] else '失败'}"
    )
    print(
        f"耗时："
        f"{result['latency_seconds']}"
    )
    print(
        f"来源数量："
        f"{len(result['sources'])}"
    )

    if result["error"]:
        print(
            f"错误：{result['error']}"
        )


def main():
    print(
        f"评估 API：{API_URL}"
    )

    cases = load_cases()

    for index, case in enumerate(
        cases,
        start=1,
    ):
        validate_case(
            case,
            index,
        )

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        result = evaluate_case_safely(
            case
        )

        results.append(result)

        print_case_result(
            index=index,
            total=len(cases),
            result=result,
        )

    summary = save_results(
        results
    )

    print("\n" + "=" * 60)
    print("评估运行完成")
    print(
        f"总数：{summary['total']}"
    )
    print(
        f"通过：{summary['passed']}"
    )
    print(
        f"失败：{summary['failed']}"
    )
    print(
        f"请求错误：{summary['errors']}"
    )
    print(
        f"结果文件：{RESULTS_PATH}"
    )

    if summary["errors"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
