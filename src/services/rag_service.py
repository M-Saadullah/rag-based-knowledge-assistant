from typing import List, Dict
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .embeddings import NomicEmbeddingsService
from ..utils.file_loader import FileLoader

class RAGService:
    def __init__(self):
        self.embeddings = NomicEmbeddingsService()
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    async def process_document(self, content: bytes, filename: str):
        """Process and store a document in the vector store."""
        # Validate file type
        if not FileLoader.validate_file_type(filename):
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Convert bytes to text using the file loader
        text = FileLoader.load_text_file(content, filename)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents
        documents = [
            Document(page_content=chunk, metadata={"source": filename})
            for chunk in chunks
        ]
        
        # Add documents to vector store
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()

    async def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant document chunks for a query."""
        docs_with_scores = await self.retrieve_relevant_chunks_with_scores(query, k)
        return [doc for doc, score in docs_with_scores]
    
    async def retrieve_relevant_chunks_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve relevant document chunks with their similarity scores."""
        try:
            # Use similarity search with scores to filter low-relevance results
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)
            
            # Filter out results with very low similarity scores
            # Chroma uses distance (lower is better), but we need stricter filtering
            relevant_docs_with_scores = []
            for doc, score in docs_with_scores:
                # Balanced threshold - include relevant content
                if score < 1.2:  # More inclusive threshold to capture PDF content
                    relevant_docs_with_scores.append((doc, score))
                    print(f"Vector search: score={score:.3f}, preview={doc.page_content[:100]}...")
                    
            return relevant_docs_with_scores[:k]  # Return top k relevant documents with scores
        except Exception as e:
            print(f"Error in vector search: {e}")
            # Fallback to regular similarity search
            fallback_docs = self.vectorstore.similarity_search(query, k=k)
            return [(doc, None) for doc in fallback_docs]  # Return with None scores

    async def keyword_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform keyword-based search as a fallback."""
        try:
            # Get all documents from the vector store
            all_docs = self.vectorstore.get()
            
            if not all_docs or not all_docs.get('documents'):
                return []
            
            # Extract query keywords (better preprocessing)
            import re
            clean_query = re.sub(r'[^\w\s]', '', query.lower())  # Remove punctuation
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
            query_words = set(word for word in clean_query.split() if word not in stop_words and len(word) > 2)
            
            if len(query_words) == 0:
                return []  # No meaningful keywords
            
            # Score documents based on keyword matches
            scored_docs = []
            documents = all_docs['documents']
            metadatas = all_docs.get('metadatas', [{}] * len(documents))
            
            for i, doc_text in enumerate(documents):
                if doc_text:
                    doc_words = set(word.lower() for word in doc_text.split())
                    # Calculate keyword overlap score
                    overlap = len(query_words.intersection(doc_words))
                    # More flexible matching: 1 match for short queries, 2 for longer ones
                    min_matches = 1 if len(query_words) <= 2 else min(2, len(query_words))
                    if overlap >= min_matches:
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        scored_docs.append((overlap, Document(page_content=doc_text, metadata=metadata)))
            
            # Sort by score (descending) and return top k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:k]]
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []