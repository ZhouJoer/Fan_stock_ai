"""Research Agent 知识库管理模块。

说明：
- 参照附件中 knowledge.py 的设计理念，使用 LangChain + FAISS + HuggingFace Embeddings 实现。
- 提供向量数据库的初始化、文档添加与语义检索功能。
- 支持从 PDF/Markdown 文件提取内容并自动插入知识库。
- 数据目前存储在内存中（session bound），如需持久化可扩展 `save_local/load_local`。
"""

import os
import io
import re
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# 使用 langchain-community 和 langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)

# 全局单例
_vector_store: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型单例。
    
    使用 'all-MiniLM-L6-v2' (384维)，速度快且效果在接受范围内。
    """
    global _embeddings
    if _embeddings is None:
        try:
            logger.info("Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
            _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise e
    return _embeddings


def get_vector_store() -> FAISS:
    """获取向量数据库单例（懒加载）。"""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        # 初始化一个空的 FAISS 索引（需要至少一个文本来初始化，或者使用特定 API）
        # 这里为了防止报错，我们插入一条系统预设的元知识
        initial_doc = Document(
            page_content="欢迎使用 ValueCell Research Agent 知识库。本知识库用于存储和检索研报、公告及其他分析材料。",
            metadata={"source": "system_init", "type": "meta"}
        )
        _vector_store = FAISS.from_documents([initial_doc], embeddings)
        logger.info("Vector Store initialized.")
    return _vector_store


def add_text_to_knowledge(text: str, metadata: dict = None) -> str:
    """向知识库添加纯文本片段。"""
    try:
        vs = get_vector_store()
        doc = Document(page_content=text, metadata=metadata or {})
        vs.add_documents([doc])
        return "Success"
    except Exception as e:
        logger.error(f"Error adding text to knowledge: {e}")
        return f"Error: {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for relevant information.
    Use this tool to find details from previously saved articles, reports, or filings.
    
    Args:
        query: The semantic search query string.
    """
    try:
        vs = get_vector_store()
        # 检索 Top 4
        results = vs.similarity_search(query, k=4)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted_results = []
        for i, res in enumerate(results):
            source = res.metadata.get("source", "Unknown")
            title = res.metadata.get("title", "")
            content_snippet = res.page_content
            formatted_results.append(
                f"--- Result {i+1} ---\nSource: {source} | Title: {title}\nContent: {content_snippet}\n"
            )
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def save_important_info(content: str, title: str = "Note", source: str = "Agent Research") -> str:
    """
    Save important research findings, summaries, or data snippets to the knowledge base for future reference.
    
    Args:
        content: The text content to save.
        title: A brief title for the content.
        source: The original source of the information (e.g., 'CNINFO', 'Market News').
    """
    try:
        add_text_to_knowledge(content, metadata={"title": title, "source": source})
        return f"Successfully saved info '{title}' to knowledge base."
    except Exception as e:
        return f"Failed to save info: {str(e)}"


@tool
def analyze_pdf_document(pdf_url: str, focus_keywords: str = "") -> str:
    """
    Download and analyze a PDF document (e.g., annual report, research report).
    Automatically extracts content and saves to knowledge base for future searches.
    
    Args:
        pdf_url: The URL of the PDF document to analyze.
        focus_keywords: Optional keywords to focus on (e.g., "revenue", "risk factors").
    
    Returns:
        A summary of the PDF content or key sections related to focus keywords.
    """
    try:
        logger.info(f"Analyzing PDF document: {pdf_url}")
        
        # 下载并提取内容
        content = download_and_extract_pdf(pdf_url)
        
        if content.startswith("["):
            return content  # 返回错误信息
        
        # 构建元数据
        metadata = {
            "source": "PDF Document",
            "pdf_url": pdf_url,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 如果有关键词，尝试提取相关段落
        if focus_keywords:
            keywords = [kw.strip() for kw in focus_keywords.split(",")]
            relevant_sections = []
            
            for section in content.split("\n\n"):
                if any(kw.lower() in section.lower() for kw in keywords):
                    relevant_sections.append(section)
            
            if relevant_sections:
                focused_content = "\n\n".join(relevant_sections[:10])  # 最多10个相关段落
                summary = f"Found {len(relevant_sections)} sections related to keywords: {focus_keywords}\n\n{focused_content}"
            else:
                summary = f"No sections found for keywords: {focus_keywords}\n\nShowing first 1000 chars:\n{content[:1000]}"
        else:
            summary = f"PDF Content Preview (first 2000 chars):\n{content[:2000]}"
        
        # 分块保存到知识库
        if len(content) > 2000:
            chunks = split_text_into_chunks(content, chunk_size=1500, chunk_overlap=200)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk"] = f"{i+1}/{len(chunks)}"
                add_text_to_knowledge(chunk, metadata=chunk_metadata)
            logger.info(f"Saved PDF content in {len(chunks)} chunks to knowledge base")
        else:
            add_text_to_knowledge(content, metadata=metadata)
        
        return summary + f"\n\n✅ Full document saved to knowledge base for future searches."
    
    except Exception as e:
        return f"Failed to analyze PDF: {str(e)}"


@tool
def analyze_markdown_document(file_path_or_url: str, section_filter: str = "") -> str:
    """
    Read and analyze a Markdown document (e.g., research notes, documentation).
    Automatically saves to knowledge base for future searches.
    
    Args:
        file_path_or_url: Local file path or URL of the Markdown document.
        section_filter: Optional section heading to focus on (e.g., "## Financial Analysis").
    
    Returns:
        The Markdown content or filtered sections.
    """
    try:
        logger.info(f"Analyzing Markdown document: {file_path_or_url}")
        
        # 读取内容
        content = read_markdown_file(file_path_or_url)
        
        if content.startswith("["):
            return content  # 返回错误信息
        
        # 构建元数据
        metadata = {
            "source": "Markdown Document",
            "file_path": file_path_or_url,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 如果有section筛选，提取相关部分
        if section_filter:
            # 简单的 Markdown section 提取（基于标题层级）
            sections = re.split(r'\n(?=#{1,6}\s)', content)
            relevant_sections = [s for s in sections if section_filter.lower() in s.lower()]
            
            if relevant_sections:
                filtered_content = "\n\n".join(relevant_sections)
                summary = f"Found {len(relevant_sections)} sections matching '{section_filter}':\n\n{filtered_content}"
            else:
                summary = f"No sections found for '{section_filter}'\n\nShowing first 1000 chars:\n{content[:1000]}"
        else:
            summary = f"Markdown Content Preview (first 2000 chars):\n{content[:2000]}"
        
        # 分块保存到知识库
        if len(content) > 2000:
            chunks = split_text_into_chunks(content, chunk_size=1500, chunk_overlap=200)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk"] = f"{i+1}/{len(chunks)}"
                add_text_to_knowledge(chunk, metadata=chunk_metadata)
            logger.info(f"Saved Markdown content in {len(chunks)} chunks to knowledge base")
        else:
            add_text_to_knowledge(content, metadata=metadata)
        
        return summary + f"\n\n✅ Full document saved to knowledge base for future searches."
    
    except Exception as e:
        return f"Failed to analyze Markdown: {str(e)}"


def download_and_extract_pdf(url: str, metadata: dict = None) -> str:
    """从 URL 下载 PDF 并提取文本内容（同步版本）。
    
    Args:
        url: PDF 文件的 URL
        metadata: 元数据字典
    
    Returns:
        提取的文本内容（前 20 页，约 10000 字符）
    """
    if PdfReader is None:
        logger.warning("pypdf not installed, cannot extract PDF content")
        return "[PDF extraction unavailable: pypdf not installed]"
    
    try:
        import requests
        
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return f"[Failed to download PDF: HTTP {response.status_code}]"
        
        pdf_bytes = response.content
        
        # 提取文本
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        # 只读前 20 页避免过大
        for i, page in enumerate(reader.pages[:20]):
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- Page {i+1} ---\n{text}")
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1}: {e}")
        
        full_text = "\n\n".join(text_parts)
        
        # 限制长度
        if len(full_text) > 15000:
            full_text = full_text[:15000] + "\n\n[... content truncated ...]"
        
        return full_text
    
    except Exception as e:
        logger.error(f"Error extracting PDF from {url}: {e}")
        return f"[PDF extraction failed: {str(e)}]"


def read_markdown_file(file_path: str) -> str:
    """读取 Markdown 文件内容（同步版本）。
    
    Args:
        file_path: Markdown 文件路径（本地路径或 URL）
    
    Returns:
        Markdown 文本内容
    """
    try:
        import requests
        
        # 如果是 URL，下载内容
        if file_path.startswith(("http://", "https://")):
            response = requests.get(file_path, timeout=15)
            if response.status_code == 200:
                return response.text
            else:
                return f"[Failed to download Markdown: HTTP {response.status_code}]"
        else:
            # 本地文件
            path = Path(file_path)
            if path.exists() and path.suffix.lower() in ['.md', '.markdown']:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"[File not found or not a Markdown file: {file_path}]"
    
    except Exception as e:
        logger.error(f"Error reading Markdown file {file_path}: {e}")
        return f"[Markdown reading failed: {str(e)}]"


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """将长文本分割成块以便更好地存储和检索。
    
    Args:
        text: 要分割的文本
        chunk_size: 每块的最大字符数
        chunk_overlap: 块之间的重叠字符数
    
    Returns:
        文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def add_cninfo_filing_to_knowledge(filing_info: dict) -> None:
    """将 CNINFO 公告信息添加到知识库（同步版本）。
    
    Args:
        filing_info: CNINFO 返回的公告字典，包含 title/date/pdf_url/company/code 等
    """
    try:
        # 构建元数据
        metadata = {
            "source": "CNINFO",
            "company": filing_info.get("company", ""),
            "code": filing_info.get("code", ""),
            "title": filing_info.get("title", ""),
            "date": filing_info.get("date", ""),
            "market": filing_info.get("market", ""),
            "pdf_url": filing_info.get("pdf_url", ""),
        }
        
        # 尝试下载并提取 PDF 内容
        pdf_url = filing_info.get("pdf_url")
        if pdf_url:
            logger.info(f"Downloading and parsing PDF for knowledge base: {pdf_url}")
            pdf_content = download_and_extract_pdf(pdf_url, metadata)
            
            # 组合标题和内容
            full_content = f"""【公告标题】{filing_info.get('title', 'N/A')}
【公司】{filing_info.get('company', 'N/A')} ({filing_info.get('code', 'N/A')})
【日期】{filing_info.get('date', 'N/A')}
【市场】{filing_info.get('market', 'N/A')}
【PDF链接】{pdf_url}

【内容摘要】
{pdf_content}
"""
            
            # 如果内容太长，分块存储
            if len(full_content) > 2000:
                chunks = split_text_into_chunks(full_content, chunk_size=1500, chunk_overlap=200)
                logger.info(f"Splitting PDF content into {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk"] = f"{i+1}/{len(chunks)}"
                    add_text_to_knowledge(chunk, metadata=chunk_metadata)
            else:
                add_text_to_knowledge(full_content, metadata=metadata)
            
            logger.info(f"Added CNINFO filing to knowledge base: {filing_info.get('title', 'Unknown')}")
        else:
            # 如果没有 PDF，至少保存元数据
            summary = f"""【公告标题】{filing_info.get('title', 'N/A')}
【公司】{filing_info.get('company', 'N/A')} ({filing_info.get('code', 'N/A')})
【日期】{filing_info.get('date', 'N/A')}
【市场】{filing_info.get('market', 'N/A')}
"""
            add_text_to_knowledge(summary, metadata=metadata)
    
    except Exception as e:
        logger.error(f"Failed to add CNINFO filing to knowledge: {e}")

