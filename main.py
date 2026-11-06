# =========================
# day13.py — LangGraph 高级 RAG：并行检索 + 多路结果融合 + Gradio 前端
# 适配 RTX 5060 8G + deepseek-r1:7b
# =========================

import os
import re
import asyncio
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

import gradio as gr
from dotenv import load_dotenv  # 负责读取 .env 文件里的键值对

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 加载 .env 文件中的环境变量（必须在 os.getenv 读取之前调用）
load_dotenv()

# 镜像 + 关闭 HuggingFace 软链接警告
# HF_ENDPOINT 优先从 .env 读取，没配置就用国内镜像兜底
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# asyncio：parallel_retrieve 用 await asyncio.gather 并行执行多个检索任务，主入口用 ainvoke。
# Annotated + add_messages：让 messages 字段按"追加"方式合并，而不是被覆盖。
# TypedDict：定义图状态结构。ChatOllama：本地调用 Ollama 模型。
# Chroma + HuggingFaceEmbeddings：本地向量库和嵌入模型。Document：检索返回的文档对象。
# ChatPromptTemplate：构造每个节点的提示词。HumanMessage / AIMessage：对话消息对象。
# StateGraph / START / END：构建 LangGraph 工作流。gradio：本地聊天前端界面。


# =========================
# 0. 配置：从 .env 集中读取（未配置时用默认值兜底）
# =========================
# 这里把易变参数集中成变量，下游 ChatOllama / Chroma 直接引用，
# 不要把这些变量名当参数塞进 ChatOllama()。
LLM_MODEL = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CTX_SIZE = int(os.getenv("LLM_CTX_SIZE", "4096"))
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_day13")


# =========================
# 1. 工具函数：清洗 DeepSeek-R1 的  思考块
# =========================
# deepseek-r1 是推理模型，输出里会先带一大段  思考，
# 路由/评估/重写这类"只要标签"的节点必须先把思考块去掉再判断。
def strip_think(text: str) -> str:
    if not text:
        return ""
    # 去掉成对的  块（含跨行内容）
    text = re.sub(r"<think.*?>.*?</think>", "", text, flags=re.DOTALL)
    # 兜底：去掉残留的单个  /  标签
    text = re.sub(r"</?think>", "", text)
    return text.strip()


# =========================
# 2. 状态定义
# =========================
class State(TypedDict):
    messages: Annotated[List, add_messages]
    route: str
    context: str
    grade: str
    rewrite_count: int
    sub_queries: List[str]
# messages：对话历史，自动追加。route：路由结果 knowledge/reasoning/chat。
# context：检索后拼好的上下文。grade：评估结果 yes/no。
# rewrite_count：重写次数，防止无限循环。sub_queries：扩展出的多个子查询。


# =========================
# 3. 知识库
# =========================
docs = [
    Document(
        page_content="RTX 3050 显存 4GB，适合本地运行 1.5B ~ 3B 参数的小模型，例如 deepseek-r1:1.5b。",
        metadata={"source": "显卡配置"},
    ),
    Document(
        page_content="RTX 5060 显存 8GB，可以流畅运行 7B 级别的量化模型，例如 deepseek-r1:7b。",
        metadata={"source": "显卡配置"},
    ),
    Document(
        page_content="LangGraph 适合用来构建有状态、多轮对话的 RAG 应用，可以自动管理对话历史。",
        metadata={"source": "框架说明"},
    ),
    Document(
        page_content="Ollama 是本地运行大模型的工具，支持 deepseek-r1、qwen、llama 等模型。",
        metadata={"source": "工具介绍"},
    ),
]

persist_dir = PERSIST_DIR  # 持久化目录改为从 .env 读取（默认 ./chroma_db_day13）

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 目录不存在就从零建库（首次运行自动入库，不依赖第 12 天）；已存在则直接加载，避免重复建库
if not os.path.exists(persist_dir):
    vector_store = Chroma.from_documents(
        docs, embedding, persist_directory=persist_dir
    )
else:
    vector_store = Chroma(
        persist_directory=persist_dir, embedding_function=embedding
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# 这里先用模拟文档建库。retriever 每次检索返回最相关的 2 条，后面 parallel_retrieve 会并行调用多次。


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)
# 把检索到的 Document 列表拼成一段文本，作为 context 喂给回答节点。


# =========================
# 4. 模型
# =========================
# 注意：ChatOllama 的合法参数是 model / base_url / num_ctx / num_predict，
# 上面定义的 LLM_MODEL / OLLAMA_BASE_URL / CTX_SIZE 在这里被正确"喂"进去，
# 而不是把变量名当关键字参数。
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_ctx=CTX_SIZE,       # R1 推理输出较长；8G 显存下可跑，若 OOM 可在 .env 把 LLM_CTX_SIZE 调成 2048
    num_predict=2048,
)

MAX_REWRITE = 2
NUM_SUB_QUERIES = 3


# =========================
# 5. 查询路由节点
# =========================
ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个查询路由器。
请判断用户问题属于哪一类，只返回一个标签：

- knowledge：需要查询知识库才能回答的问题
- chat：日常聊天、问候、情绪表达
- reasoning：需要推理、计算或分步分析的问题

只返回标签，不要解释。""",
        ),
        ("human", "{question}"),
    ]
)


def route_query(state: State) -> dict:
    last_msg = state["messages"][-1]
    question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    chain = ROUTE_PROMPT | llm
    output = chain.invoke({"question": question})

    route_label = strip_think(output.content).strip().lower()

    if "knowledge" in route_label:
        return {"route": "knowledge"}
    if "reasoning" in route_label:
        return {"route": "reasoning"}
    return {"route": "chat"}  # 默认兜底为 chat


def route_after_query(state: State) -> Literal["generate_queries", "reason", "chat"]:
    if state["route"] == "knowledge":
        return "generate_queries"
    if state["route"] == "reasoning":
        return "reason"
    return "chat"


# =========================
# 6. 生成子查询节点
# =========================
QUERY_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个查询扩展助手。
根据用户问题，生成多个不同角度的检索子查询。
请返回一个 Python 列表，例如：["查询1", "查询2", "查询3"]
只返回列表，不要解释。""",
        ),
        ("human", "原始问题：{question}"),
    ]
)


def generate_queries(state: State) -> dict:
    last_msg = state["messages"][-1]
    question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    chain = QUERY_GEN_PROMPT | llm
    output = chain.invoke({"question": question})

    text = strip_think(output.content).strip()

    # 容错：模型可能没按"列表"格式返回，解析失败一律兜底回原问题，防止直接崩
    try:
        sub_queries = eval(text)
        if not isinstance(sub_queries, list):
            sub_queries = [question]
    except Exception:
        sub_queries = [question]

    # 逐个清洗：只保留非空字符串元素；清洗后为空再兜底回原问题
    sub_queries = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]
    if not sub_queries:
        sub_queries = [question]

    return {"sub_queries": sub_queries}


# =========================
# 7. 并行检索节点
# =========================
async def parallel_retrieve(state: State) -> dict:
    sub_queries = state.get("sub_queries") or []

    # 子查询为空时用原问题兜底，保证至少检索一次
    if not sub_queries:
        last_msg = state["messages"][-1]
        question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)
        sub_queries = [question]

    tasks = []
    for q in sub_queries:
        tasks.append(retriever.ainvoke(q))

    results = await asyncio.gather(*tasks)
    # 这是今天最关键的一处：每个子查询生成一个 retriever.ainvoke(q) 协程，
    # asyncio.gather(*tasks) 让它们并发执行。因为节点是 async def，
    # 主入口必须用 app.ainvoke，否则报 TypeError: No synchronous function provided。

    all_docs = []
    seen_contents = set()

    # 多路结果去重融合：用 seen_contents 按整段内容去重，避免同一段重复进入上下文
    for docs in results:
        for d in docs:
            content = d.page_content.strip()
            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(d)
    context = format_docs(all_docs)
    return {"context": context}


# =========================
# 8. 评估节点
# =========================
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你只负责判断检索到的上下文是否能帮助回答用户问题。
如果上下文里有明显相关信息，返回：yes
如果上下文完全不相关，返回：no
只返回 yes 或 no，不要解释。""",
        ),
        ("human", "问题：{question}\n\n上下文：{context}"),
    ]
)


def grade(state: State) -> dict:
    last_msg = state["messages"][-1]
    question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    result = GRADE_PROMPT | llm
    output = result.invoke(
        {
            "question": question,
            "context": state["context"],
        }
    )

    grade_text = strip_think(output.content).strip().lower()
    grade_text = grade_text.replace(" ", "")  # 去掉可能混入的空格，提高 in 判定鲁棒性

    if "yes" in grade_text:
        return {"grade": "yes"}
    return {"grade": "no"}


def route_after_grade(state: State) -> Literal["answer", "rewrite"]:
    if state.get("grade") == "yes":
        return "answer"
    return "rewrite"


# =========================
# 9. 重写节点
# =========================
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个查询重写助手。
根据用户原始问题，生成一个更适合知识库检索的查询语句。
只返回重写后的查询，不要解释。""",
        ),
        ("human", "原始问题：{question}"),
    ]
)


def rewrite(state: State) -> dict:
    last_msg = state["messages"][-1]
    question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    chain = REWRITE_PROMPT | llm
    rewritten = chain.invoke({"question": question})

    new_messages = [HumanMessage(content=strip_think(rewritten.content).strip())]
    return {
        "messages": new_messages,
        "rewrite_count": state.get("rewrite_count", 0) + 1,  # 计数 +1，配合建图处的上限防死循环
    }


# =========================
# 10. 推理节点
# =========================
REASON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个推理助手。请逐步分析用户问题，给出清晰、有条理的回答。
不需要检索知识库，但可以结合常识和逻辑推理。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def reason(state: State) -> dict:
    chain = REASON_PROMPT | llm
    response = chain.invoke({"messages": state["messages"]})
    response.content = strip_think(response.content)
    return {"messages": [response]}


# =========================
# 11. 聊天节点
# =========================
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个友好的聊天助手。请用自然、简洁的方式回应用户。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def chat(state: State) -> dict:
    chain = CHAT_PROMPT | llm
    response = chain.invoke({"messages": state["messages"]})
    response.content = strip_think(response.content)
    return {"messages": [response]}


# =========================
# 12. 回答节点
# =========================
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个 AI 助手。请严格根据参考上下文回答问题。
如果上下文里没有相关信息，就说"知识库中没有相关内容"，不要编造。
回答要自然完整，不要只罗列关键词。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "参考上下文：\n{context}"),
    ]
)


def answer(state: State) -> dict:
    chain = ANSWER_PROMPT | llm
    response = chain.invoke(
        {
            "messages": state["messages"],
            "context": state["context"],
        }
    )
    response.content = strip_think(response.content)
    return {"messages": [response]}


# =========================
# 13. 构建图
# =========================
workflow = StateGraph(State)

workflow.add_node("route_query", route_query)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("parallel_retrieve", parallel_retrieve)
workflow.add_node("grade", grade)
workflow.add_node("rewrite", rewrite)
workflow.add_node("reason", reason)
workflow.add_node("chat", chat)
workflow.add_node("answer", answer)

workflow.add_edge(START, "route_query")

workflow.add_conditional_edges(
    "route_query",
    route_after_query,
    {
        "generate_queries": "generate_queries",
        "reason": "reason",
        "chat": "chat",
    },
)

workflow.add_edge("generate_queries", "parallel_retrieve")
workflow.add_edge("parallel_retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "answer": "answer",
        "rewrite": "rewrite",
    },
)

workflow.add_conditional_edges(
    "rewrite",
    lambda state: "generate_queries" if state.get("rewrite_count", 0) < MAX_REWRITE else "answer",
    {
        "generate_queries": "generate_queries",
        "answer": "answer",
    },
)

workflow.add_edge("reason", END)
workflow.add_edge("chat", END)
workflow.add_edge("answer", END)

app = workflow.compile()

# 整条 knowledge 路径的执行顺序：
# 用户问题 -> route_query -> generate_queries -> parallel_retrieve 并行检索多路
# -> grade 评估上下文是否相关 -> 相关: answer / 不相关: rewrite 后回到 generate_queries
# 相比第 12 天，最大工程意义：检索从"单查询单路"变成"多查询并行 + 去重融合"。


# =========================
# 14. Gradio 前端界面
# =========================
# 把 Gradio 历史转成 LangChain 消息列表
def to_lc_messages(history: List[dict]) -> List:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def build_ui():
    with gr.Blocks(title="RAG 第13天 · 并行检索 + 多路融合") as demo:
        gr.Markdown("## 第 13 天：并行检索 + 多路结果融合")
        gr.Markdown(f"模型：`{LLM_MODEL}`  |  支持 知识库问答 / 推理 / 闲聊")

        # 关键修正：本文件历史用的是字典格式 {"role","content"}，
        # 在 Gradio 4/5/6 下必须配 type="messages" 才能正确显示；
        # 若哪天退回 Gradio 3.x，则需把本行改回不带 type，并把历史改回 [[user, bot], ...] 元组格式。
        chatbot = gr.Chatbot(label="对话", height=520, type="messages")
        msg = gr.Textbox(
            label="输入问题",
            placeholder="输入你的问题，按 Enter 发送...",
        )
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话")

        # history 用 gr.State 维护一份完整消息列表
        state = gr.State([])

        async def respond(text, history):
            text = (text or "").strip()
            if not text:
                return history, history
            history = history + [{"role": "user", "content": text}]

            graph_state: State = {
                "messages": to_lc_messages(history),
                "route": "",
                "context": "",
                "grade": "",
                "rewrite_count": 0,
                "sub_queries": [],
            }

            result = await app.ainvoke(graph_state)
            ai_msg = result["messages"][-1]
            history = history + [{"role": "assistant", "content": ai_msg.content}]
            return history, history

        def clear():
            return [], []

        msg.submit(respond, [msg, state], [chatbot, state])
        send_btn.click(respond, [msg, state], [chatbot, state])
        clear_btn.click(clear, outputs=[chatbot, state])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, inbrowser=True)