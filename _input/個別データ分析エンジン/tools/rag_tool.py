"""
RAG (Retrieval-Augmented Generation) Tool

Provides vector search capabilities for specifications and code documents.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import config
from ..utils.logger import get_logger
from ..utils.file_utils import read_markdown, split_into_chunks

logger = get_logger(__name__)


class RAGTool:
    """
    RAG tool for document retrieval and search
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.openai.api_key
        )
        self.vector_stores: Dict[str, FAISS] = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.vectorstore.chunk_size,
            chunk_overlap=config.vectorstore.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def initialize_vector_stores(self, documents: Dict[str, List[str]]) -> bool:
        """
        Initialize vector stores for different document segments

        Args:
            documents: Dictionary mapping segment names to document paths

        Returns:
            True if initialization successful
        """
        try:
            config.ensure_directories()

            for segment_name, doc_paths in documents.items():
                logger.info(f"Initializing vector store for segment: {segment_name}")

                # Check if vector store already exists
                vectorstore_path = config.get_vectorstore_path(segment_name)
                if self._vectorstore_exists(vectorstore_path):
                    logger.info(f"Loading existing vector store for {segment_name}")
                    self.vector_stores[segment_name] = FAISS.load_local(
                        vectorstore_path, self.embeddings, allow_dangerous_deserialization=True
                    )
                    continue

                # Create new vector store
                docs = self._load_and_split_documents(doc_paths, segment_name)
                if docs:
                    vectorstore = FAISS.from_documents(docs, self.embeddings)
                    vectorstore.save_local(vectorstore_path)
                    self.vector_stores[segment_name] = vectorstore
                    logger.info(f"Created vector store for {segment_name} with {len(docs)} chunks")
                else:
                    logger.warning(f"No documents found for segment {segment_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize vector stores: {e}")
            return False

    def search(self, query: str, segment: str = "all", k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents

        Args:
            query: Search query
            segment: Segment to search in ("all" for all segments)
            k: Number of results to return

        Returns:
            List of search results with metadata
        """
        try:
            results = []

            if segment == "all":
                # Search all segments
                for seg_name, vectorstore in self.vector_stores.items():
                    seg_results = self._search_single_segment(query, seg_name, k // len(self.vector_stores) + 1)
                    results.extend(seg_results)
            else:
                # Search specific segment
                if segment in self.vector_stores:
                    results = self._search_single_segment(query, segment, k)
                else:
                    logger.warning(f"Segment {segment} not found")

            # Sort by relevance score and limit results
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results[:k]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _search_single_segment(self, query: str, segment: str, k: int) -> List[Dict[str, Any]]:
        """Search within a single segment"""
        if segment not in self.vector_stores:
            return []

        try:
            vectorstore = self.vector_stores[segment]
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in docs_with_scores:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'segment': segment,
                    'source': doc.metadata.get('source', 'unknown')
                })

            return results

        except Exception as e:
            logger.error(f"Search in segment {segment} failed: {e}")
            return []

    def _load_and_split_documents(self, doc_paths: List[str], segment_name: str) -> List[Document]:
        """Load and split documents for a segment"""
        documents = []

        for doc_path in doc_paths:
            try:
                if not Path(doc_path).exists():
                    logger.warning(f"Document not found: {doc_path}")
                    continue

                # Read document based on file type
                if doc_path.endswith('.md'):
                    content = read_markdown(doc_path)
                else:
                    # Try to read as text file
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                if not content.strip():
                    logger.warning(f"Empty document: {doc_path}")
                    continue

                # Split into chunks
                chunks = self.text_splitter.split_text(content)

                # Create Document objects
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': doc_path,
                            'segment': segment_name,
                            'chunk_id': i,
                            'file_hash': self._get_file_hash(doc_path)
                        }
                    )
                    documents.append(doc)

                logger.info(f"Loaded {len(chunks)} chunks from {doc_path}")

            except Exception as e:
                logger.error(f"Failed to load document {doc_path}: {e}")

        return documents

    def _vectorstore_exists(self, path: str) -> bool:
        """Check if vector store exists and is valid"""
        try:
            if not Path(path).exists():
                return False

            # Try to load to verify it's valid
            FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            return True

        except Exception:
            return False

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def get_segment_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all segments"""
        info = {}
        for segment_name, vectorstore in self.vector_stores.items():
            try:
                # This is a simplified way to get basic info
                info[segment_name] = {
                    'document_count': len(vectorstore.docstore._dict) if hasattr(vectorstore, 'docstore') else 0,
                    'path': config.get_vectorstore_path(segment_name)
                }
            except Exception as e:
                logger.warning(f"Failed to get info for segment {segment_name}: {e}")
                info[segment_name] = {'error': str(e)}

        return info

    def update_segment(self, segment_name: str, doc_paths: List[str]) -> bool:
        """Update a specific segment with new documents"""
        try:
            logger.info(f"Updating segment: {segment_name}")
            docs = self._load_and_split_documents(doc_paths, segment_name)

            if docs:
                vectorstore = FAISS.from_documents(docs, self.embeddings)
                vectorstore_path = config.get_vectorstore_path(segment_name)
                vectorstore.save_local(vectorstore_path)
                self.vector_stores[segment_name] = vectorstore
                return True
            else:
                logger.warning(f"No documents to update for segment {segment_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to update segment {segment_name}: {e}")
            return False
