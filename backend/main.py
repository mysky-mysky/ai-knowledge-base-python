import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import jieba
import pypdf
import docx
import io

load_dotenv()
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

client = OpenAI( #用openai框架初始化deep seek客户端对象,用来调用deep seek模型
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

app.add_middleware( #给app添加中间件
    CORSMiddleware,  #中间件类型是cors
    allow_origins=["*"], #允许所有域名(访问)
    allow_methods=["*"], #允许所有方法(访问),如GET POST PUT等
    allow_headers=["*"], #允许所有请求头
)

#全局变量
document_chunks = []
bm25_model = None #存放 BM25 模型

@app.get("/") #先检查接口
def read_root():
    return {"message": "服务器跑起来了"}

#切片函数
def split_text(text, chunk_size=300, overlap=50):
    chunks = [] #存放切后的文本
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks #返回切片后的文本

@app.post("/api/upload") #前端将文件提交到后端接口
async def upload_file(file:UploadFile = File(...)): #定义异步函数,接收来自上传的文件
    global document_chunks, bm25_model #要改变全局变量了
    content = await file.read() #等待上传的文件读完后存入content(一堆字节)
    text = "" #定义空字符串(准备存解析后的文字)

    if file.filename.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content)) #将读到的文件包装成内存里的文件
        for page in reader.pages: #遍历PDF的每一页
            text += page.extract_text() or "" #提取每页文字,追加到text
    elif file.filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        for para in doc.paragraphs: #遍历每个段落
            text += para.text + "\n"
    elif file.filename.endswith(".doc"):
        return {"message": "请将.doc文件另存为.docx后上传"}
    else:
        return {"message": "不支持的文件格式, 请上传PDF或DOCX"}

    #切片
    document_chunks = split_text(text)
    if not document_chunks:
        return {"message": "文档中没有可提取的文字"}
    print("切片数量:", len(document_chunks))
    
    # 开始使用 BM25
    # 1.用jieba对每个切片进行中文分词
    tokenized = [list(jieba.cut(chunk)) for chunk in document_chunks]
    # 2.构建 BM25 索引
    bm25_model = BM25Okapi(tokenized)
    
    print("BM25索引构建完成")
    return {"message": "上传成功", "length": len(text), "chunks": len(document_chunks)}

@app.post("/api/chat")
async def chat(req: ChatRequest): #定义一个名字叫聊天的异步函数
    global bm25_model
    message = req.message #取参数里键为message的值,叫message,如无,返回空
    if not message:
        return {"error": "消息不能为空"}
    if bm25_model is None:
        return {"error": "请先上传文档"}

    # 使用 BM25 检索
    # 1. 对用户问题分词
    tokenized_query = list(jieba.cut(message))
    # 2. 从 BM25 模型中获取打分最高的5个片段
    top_chunks = bm25_model.get_top_n(tokenized_query, document_chunks, n=5)

    print("匹配到的片段:", top_chunks)

    #拼提示词
    context = "\n\n".join(top_chunks) #将几个片段拼成一段文字
    prompt = f"""请根据以下文档片段回答问题,如果文档中没有相关信息,就说不知道:

    {context}

    用户问题:{message}
    """ #得到提示词
    try:
        #调用 DeepSeek
        response = client.chat.completions.create( #让客户端去聊天窗口的补全部门发个请求
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        ) #得到一个响应对象

        reply = response.choices[0].message.content #取出响应对象里的choices里的message里的content
        return {"reply": reply} #返回给前端
    except Exception as e:
        print("调用 DeepSeek 失败:", e)
        return {'error': 'AI 接口调用失败'}
