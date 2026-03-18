from markitdown import MarkItDown
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import Config
from typing import List, Optional, Tuple
from langchain_core.documents import Document

import re
import os


class MarkdownIngestor:
    """
    智能文档处理流水线：
    PDF -> MarkItDown -> Markdown 清洗 -> 智能标题切分 -> 递归细粒度切分 -> Document
    """

    def __init__(self, 
                 chunk_size: int = Config.CHUNK_SIZE, 
                 chunk_overlap: int = Config.CHUNK_OVERLAP,
                 headers_to_split_on: Optional[List[Tuple[str, str]]] = None):
        """
        在初始化时加载转换器和切分器，避免每次处理文件时重复实例化
        """
        # 1. 初始化文件转换器
        self.md_converter = MarkItDown()
        
        # 2. 配置切分参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 3. 配置 Markdown 标题层级
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "h1"),
            ("##", "h2"), 
            ("###", "h3"),
        ]
        
        # 4. 初始化 LangChain 分割器
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # 保留标题在内容中，利于大模型理解上下文
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # 加入中文标点支持，防止把一句话劈成两半
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

    def convert_to_md(self, file_path: str) -> str:
        """步骤一：使用 MarkItDown 将文件转换为 Markdown 格式"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        result = self.md_converter.convert(file_path)
        return result.text_content
    

    def clean_md(self, text: str) -> str:
        """步骤二：清洗 Markdown 文本中的冗余换行和空格"""

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        invisible_chars = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u200e\u200f\ufeff]+'
        text = re.sub(invisible_chars, '', text)

        text = text.replace('\xa0', ' ')

        # 去掉单独页码
        text = re.sub(r"^\s*[-_]*\s*\d+\s*[-_]*\s*$", "", text, flags=re.MULTILINE)

        # Prompt 模板标题降级
        prompt_pattern = r"^(#{1,6})\s+(Instruction|Response|Input|Output|Question|Answer|User|Assistant):"
        text = re.sub(prompt_pattern, r"**\2:**", text, flags=re.MULTILINE | re.IGNORECASE)

        # 数字章节标题提升为 markdown 标题
        def promote_numeric_headers(match):
            number = match.group(1)
            title = match.group(2).strip()

            # 太长就别当标题，避免误伤正文
            if len(title) > 60:
                return match.group(0)

            dot_count = number.count(".")
            if dot_count == 0:
                return f"# {number} {title}"
            elif dot_count == 1:
                return f"## {number} {title}"
            else:
                return f"### {number} {title}"

        text = re.sub(
            r"^(\d+(?:\.\d+)*)\s+(.{1,60})$",
            promote_numeric_headers,
            text,
            flags=re.MULTILINE
        )

        # 只降级“特别像代码注释/句子”的超长标题
        def demote_long_headers(match):
            level = match.group(1)
            content = match.group(2).strip()
            if len(content) > 80 and any(p in content for p in ["。", "：", "，", "def ", "class ", "return ", "="]):
                return f"**{content}**"
            return match.group(0)

        text = re.sub(r"^(#{1,6})\s+(.+)$", demote_long_headers, text, flags=re.MULTILINE)

        text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def split_md_by_headers(self, text: str, source: str, max_direct_keep: int = 1000) -> List[Document]:
        """步骤三：按标题层级与字符长度进行双重切分"""

        header_docs = self.header_splitter.split_text(text)

        final_docs = []
        for idx, doc in enumerate(header_docs):
            content = doc.page_content.strip()
            if not content:
                continue

            metadata = dict(doc.metadata)
            metadata["source"] = source
            metadata["chunk_id"] = str(idx)

            section_parts = [
                metadata.get("h1"),
                metadata.get("h2"),
                metadata.get("h3"),
            ]
            section_path = " > ".join([p for p in section_parts if p])
            if section_path:
                metadata["section_path"] = section_path

            if len(content) <= max_direct_keep:
                final_docs.append(Document(page_content=content, metadata=metadata))
            else:
                sub_docs = self.recursive_splitter.create_documents([content], [metadata])
                for sub_idx, sub_doc in enumerate(sub_docs):
                    sub_doc.metadata["chunk_id"] = f"{idx}-{sub_idx}"
                    final_docs.append(sub_doc)

        return final_docs

    def process_file(self, file_path: str, max_direct_keep: int = 1000) -> List[Document]:
        """
        转换 -> 清洗 -> 切分
        """
        print(f"📄 正在解析文件并转换为 Markdown: {file_path}")
        source_name = os.path.basename(file_path)
        
        # 1. 转换
        raw_md = self.convert_to_md(file_path)
        
        # 2. 清洗
        cleaned_md = self.clean_md(raw_md)
        
        # 3. 切分
        print("✂️ 正在进行基于 Markdown 语义的智能分块...")
        docs = self.split_md_by_headers(
            cleaned_md, 
            source=source_name, 
            max_direct_keep=max_direct_keep
        )
        
        print(f"✅ 处理完成，共生成 {len(docs)} 个带层级 Metadata 的文档块。")
        return docs

