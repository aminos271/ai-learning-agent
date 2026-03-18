
---

# 学习型智能 Agent（RAG知识检索 + 笔记记忆） 

一个基于 LangGraph 构建的智能学习助手，支持知识检索与个人笔记记忆的结合。


---

## 🚀 特色

* 🔍 **RAG 知识检索**（Multi-Query 提升召回）
* 🧠 **用户笔记存储与召回**
* 🔄 **问题改写（Rewrite）**
* 🚦 **意图路由（Router）**
* 🧩 **LangGraph 工作流编排**

---

## ⚙️ 亮点

相比简单的 RAG Demo，本项目进行了如下工程优化：

- 🔀 Multi-Query 检索：提升召回率
- 📊 简单 Rerank：基于相似度与结构信息排序
- 🧩 Metadata-aware 检索：结合文档结构信息
- 🧠 独立 Memory Store：用户笔记与知识库分离
- 🔄 状态驱动工作流：基于 LangGraph 实现可控流程

## 📁 项目结构

```bash
ai_learning_agent/
├── core/          # 配置与运行时
├── rag/           # 文档处理与检索
├── memory/        # 笔记存储与检索
├── graph/         # Agent 工作流
│   ├── nodes/     # 各功能节点
│   ├── state.py
│   ├── workflow.py
│   └── prompts.py
├── main.py
```

---

# ⚙️ 环境准备


## 🧱 系统依赖

本项目依赖以下组件：

- **LLM**：DeepSeek API
- **Embedding**：Ollama（本地 bge-m3）
- **向量数据库**：Qdrant（Docker 部署）

请确保以下服务运行：

- Qdrant: http://localhost:6333
- Ollama: http://localhost:11434


### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2️⃣ 启动 Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

### 3️⃣ 启动 Ollama 并拉取 embedding 模型

```bash
ollama pull bge-m3
```

> 默认使用本地 Ollama 作为 embedding 服务（见 `Config`）

---

### 4️⃣ 配置环境变量

```bash
cp .env.example .env
```

填写以下内容：

```env
LLM_MODEL_ID=your_model
LLM_BASE_URL=your_base_url
LLM_API_KEY=your_api_key
```

---

## ▶️ 运行

### 交互模式

```bash
python main.py -i
```

### 单轮问答

```bash
python main.py -q "什么是LLM"
```

---

## 💡 示例

```text
Q: 什么是 LLM
→ 走 RAG，返回知识库内容
> 示例
------------------------------------------------------------
✍️ [重写节点] 正在分析上下文并重写问题...
⏩ [重写节点] 意图明确，未做修改。
🔄 路由决策: rag (置信度: 1.0)
📝 决策原因: 用户在询问外部知识、概念解释
🔍 正在检索知识库...
🌀 启动原生 Multi-Query 引擎，正在裂变问题: '什么是LLM'
改写后的问题：
['LLM的定义是什么？', '请解释大语言模型（LLM）的含义', 'LLM（大语言模型）的基本概念介绍']
✅ 裂变检索完成，共合并去重得到 4 个独立文档块。
🧠 正在呼叫大模型思考中...
✅ RAG回答生成完成
🚦 [路由系统]: 走 rag 通道
🤖 [AI 助手]:
根据提供的文档，LLM（Large Language Model）即......

Q: 帮我记录一下：Transformer 很重要
→ 走 Note Store，保存笔记

Q: 我刚刚记录了什么
→ 走 Note Recall，返回历史笔记
```

---

## 🧠 工作流

```text
User Input
   ↓
Rewrite（问题改写）
   ↓
Router（意图判断）
   ├── RAG（知识检索）
   ├── Note Recall（笔记召回）
   ├── Note Store (记录笔记知识点) 
   └── Chat（闲聊）
```

> 基于 LangGraph 构建状态驱动流程 

---

## 📌 TODO

* [ ] RAG + Memory 融合（当前是分离的）
* [ ] 前端界面（Web UI）
* [ ] 多轮记忆优化（长期记忆 vs 会话记忆）
* [ ] Metadata-aware 检索增强

---

## ⚠️ 注意

* 需要提前启动：

  * Qdrant（向量数据库）
  * Ollama（embedding 服务）
* 默认配置：

  * Qdrant: `http://localhost:6333`
  * Ollama: `http://localhost:11434` 

---

## 🧩 设计说明

这个项目的核心目标不是实现一个简单的 RAG，
而是探索如何将：

- 外部知识（RAG）
- 用户记忆（Memory）
- 意图判断（Router）

整合成一个可控的 Agent 系统。



## 📎 补充说明

本项目是使用 LangGraph 结合 RAG 与记忆系统的一个早期探索，本项目目前是一个实验性质的个人项目。

代码主要在作者的本地环境中经过了测试。由于项目依赖于 Ollama 和 Qdrant 等外部服务，在不同的部署环境下可能会遇到一些特定的兼容或连接问题。

如果您在运行中遇到困难，请优先检查相关的配置选项以及底层服务的运行状态。 非常欢迎大家提交 Issue 或提出宝贵的建议。

目前仍有许多需要改进的地方，特别是在以下方面：

- RAG 与记忆系统的深度整合
- 长期记忆的设计
- 评估与基准测试

欢迎大家提出反馈与建议。

