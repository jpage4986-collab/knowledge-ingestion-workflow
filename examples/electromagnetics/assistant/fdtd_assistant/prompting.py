"""Optional LangChain prompt seam; deterministic generation remains the default."""

from __future__ import annotations


def answer_prompt(question: str, evidence_text: str) -> object:
    """Return a LangChain prompt when installed, otherwise a plain prompt string."""
    template = (
        "你是电磁/FDTD 助手。仅依据证据回答问题；保留单位和公式；"
        "用户要求仿真代码时，优先使用同软件、同任务的检索代码样例，注明来源、版本和许可证；"
        "区分样例原文与为当前问题做的修改，不臆造 API、材料参数或边界条件；"
        "不得声称代码已运行或验证，除非有明确的执行结果；"
        "每个关键结论附 [n] 引用；证据不足时明确说明。\n"
        "问题：{question}\n证据：{evidence}"
    )
    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore

        return ChatPromptTemplate.from_template(template).format_messages(question=question, evidence=evidence_text)
    except ImportError:
        return template.format(question=question, evidence=evidence_text)
