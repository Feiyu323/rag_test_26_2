# 🚀 RAG 聊天机器人（第13天：并行检索 + 多路融合）

基于 DeepSeek-R1 本地大模型和 LangGraph 构建的 RAG 智能问答系统。

## 🌟 项目特性

- **并行检索**：自动将用户问题改写为多个子查询，并发执行检索。
- **多路融合**：对不同子查询检索到的文档进行智能去重和融合。
- **动态路由**：自动判断用户意图，区分“知识库问答”、“逻辑推理”和“日常闲聊”。
- **自校正循环**：具备相关性评估与查询重写机制，检索失败时自动优化重试。
- **流式交互**：基于 Gradio 的现代化聊天前端。

## 🏗️ 系统流程图

```text
用户输入
  │
  ▼
查询路由 (route_query)
  │
  ├── chat ───────────────────────▶ 纯聊天响应 ──────────▶ 结束
  │
  ├── knowledge ───▶ 生成子查询 (generate_queries)
  │                         │
  │                  并行检索 (parallel_retrieve)
  │                         │
  │                  评估相关性 (grade_docs)
  │                         │
  │            ┌─── yes ─────┼──── no ────┐
  │            ▼             │            ▼
  │      生成答案 (answer)   │     重写查询 (rewrite)
  │            │             │            │
  │            ▼             │            └─────▶ (回到 generate_queries)
  │          结束             │
  │                           │
  └──────── reasoning ────────▶ 纯推理响应 ──────────▶ 结束
  
🛠️ 快速开始

1. 环境准备
安装 Ollama 并启动服务。
拉取 DeepSeek-R1 模型：
ollama pull deepseek-r1:7b

2. 安装依赖
pip install -r requirements.txt

3. 配置环境变量
复制 .env.example 为 .env：
cp .env.example .env
在 .env 中确认模型名称和向量库路径。

4. 运行项目
python main.py
浏览器会自动打开聊天界面。

📁 目录结构
rag_test_26_2/
├── main.py              # 项目主入口与 RAG 核心逻辑
├── requirements.txt     # Python 依赖列表
├── .env                 # 本地配置文件（不上传）
├── .env.example         # 配置文件模板
├── .gitignore           # Git 忽略文件列表
├── chroma_db_day13/     # 向量数据库存储目录（不上传）
└── README.md            # 项目说明文档