# RAG 智能知识库 (python 版)

一个基于 RAG (检索增强生成) 架构的智能问答系统. 用户上传 PDF 或 Word 文档后, 系统自动提取内容并建立轻量级索引, 用户基于文档内容进行语义检索, 结合 DeepSeek API 生成有依据的回答.

## 核心功能

- 文档解析: 支持 PDF 和 Word(.docx) 格式提取
- 文本切片: 长文本自动切片, 保留上下文重叠
- 轻量检索: 采用 BM25 算法 + 'jieba' 中文分词, 无需庞大的本地向量模型, 极低内存占用
- 智能生成: 调用 DeepSeek Chat API, 严格基于检索到的文档片段生成回答, 避免幻觉
- 前端交互: React + Vite 实现的聊天界面, 支持文件上传和上下文对话

## 技术栈

- 前端: React + Vite
- 后端: Python 3.11 + FastAPI
- 检索算法: BM25 (`rank_bm25`) + Jieba 中文分词
- AI 大模型: DeepSeek API (`openai` SDK)
- 部署: Render(后端), Netlify(前端)

## 工程亮点

1. 资源受限下的工程妥协: 针对 Render 免费版 512mb 内存的限制, 放弃了占用内存极大的`sentence-transformers` 本地向量模型, 改用 BM25 关键词检索. 在保证基础问答精准的同时, 彻底解决了服务器 OOM (内存溢出) 崩溃的问题.
2. 防止 AI 幻觉: 在 Prompt 中严格约束大模型: "只根据文档片段回答, 不知道就说不知道", 并引入了中文分词, 提升了检索的准确度.

## 项目结构

```text
ai-knowledge-base-python/
├── backend/
│   ├── main.py            # FastAPI 核心代码（解析、检索、AI 调用）
│   ├── requirements.txt   # Python 依赖清单
│   └── .env               # 本地环境变量（DEEPSEEK_API_KEY）
├── frontend/              # React 前端代码
│   ├── src/
│   └── package.json
└── .gitignore
```

## 本地运行

### 1. 启动后端

```bash
cd backend
pip install -r requirements.txt
# 创建 .env 文件, 填入 DEEPSEEK_API_KEY
uvicorn main:app --reload --port 5001
```

### 2. 启动前端
```bash
cd frontend
npm install
npm run dev
```

## 线上地址

- GitHub 仓库: https://github.com/mysky-mysky/ai-knowledge-base-python
- 前端: https://fanciful-gumdrop-8adeac.netlify.app/
- 后端: https://ai-knowledge-base-python.onrender.com
- 接口文档: https://ai-knowledge-base-python.onrender.com/docs

## 作者

mysky-mysky
