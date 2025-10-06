from typing import Dict, List, Tuple, Any, TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain.schema import Document
from .rag_service import RAGService
import google.generativeai as genai
import os
import time

# Define the state type as a TypedDict
class State(TypedDict):
    question: str
    context: Optional[List[Document]]
    answer: Optional[str]
    error: Optional[str]
    execution_log: List[Dict[str, Any]]
    workflow_path: List[str]
    start_time: float
    sources_used: List[Dict[str, Any]]

class KnowledgeAssistant:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=api_key
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create a state graph with the defined schema
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("fallback", self._fallback)

        # Add conditional edges (no regular edges needed when using conditional edges)
        workflow.add_conditional_edges(
            "retrieve_context",
            self._should_fallback,
            {
                True: "fallback",
                False: "generate_answer"
            }
        )
        
        # Add terminal edges
        workflow.add_edge("generate_answer", "__end__")
        workflow.add_edge("fallback", "__end__")

        # Set entry point
        workflow.set_entry_point("retrieve_context")

        # Compile the graph
        return workflow.compile()

    async def _retrieve_context(self, state: State) -> State:
        """Retrieve context using parallel retrievers."""
        import asyncio
        
        step_start = time.time()
        question = state["question"]
        execution_log = state.get("execution_log", [])
        workflow_path = state.get("workflow_path", [])
        sources_used = state.get("sources_used", [])
        
        workflow_path.append("retrieve_context")
        
        try:
            # Log start of parallel retrieval
            execution_log.append({
                "step": "retrieve_context_start",
                "status": "running",
                "details": {"message": "Starting parallel retrieval (vector + keyword search)"},
                "timestamp": time.time() - state.get("start_time", time.time())
            })
            
            # Run retrievers in parallel using asyncio.gather
            vector_start = time.time()
            vector_results, keyword_results = await asyncio.gather(
                self.rag_service.retrieve_relevant_chunks_with_scores(question),
                self.rag_service.keyword_search(question),
                return_exceptions=True
            )
            retrieval_time = time.time() - vector_start
            
            # Handle exceptions from parallel execution
            vector_count = 0
            keyword_count = 0
            vector_docs = []
            vector_scores = {}
            
            if isinstance(vector_results, Exception):
                execution_log.append({
                    "step": "vector_search",
                    "status": "error",
                    "details": {"error": str(vector_results)},
                    "timestamp": time.time() - state.get("start_time", time.time())
                })
                vector_results = []
            else:
                vector_count = len(vector_results)
                # Extract documents and scores from vector results
                for doc, score in vector_results:
                    vector_docs.append(doc)
                    vector_scores[doc.page_content] = score
                    
                execution_log.append({
                    "step": "vector_search",
                    "status": "completed",
                    "details": {"results_found": vector_count, "time_taken": f"{retrieval_time:.2f}s"},
                    "timestamp": time.time() - state.get("start_time", time.time())
                })
            
            if isinstance(keyword_results, Exception):
                execution_log.append({
                    "step": "keyword_search", 
                    "status": "error",
                    "details": {"error": str(keyword_results)},
                    "timestamp": time.time() - state.get("start_time", time.time())
                })
                keyword_results = []
            else:
                keyword_count = len(keyword_results)
                execution_log.append({
                    "step": "keyword_search",
                    "status": "completed", 
                    "details": {"results_found": keyword_count, "time_taken": f"{retrieval_time:.2f}s"},
                    "timestamp": time.time() - state.get("start_time", time.time())
                })
            
            # Merge and deduplicate results (prioritize vector results for scores)
            all_results = list({doc.page_content: doc for doc in vector_docs + keyword_results}.values())
            
            # Log sources used with actual relevance scores
            for i, doc in enumerate(all_results):
                # Get the relevance score from vector search, or None if from keyword search
                relevance_score = vector_scores.get(doc.page_content)
                # Convert distance to similarity (lower distance = higher similarity)
                similarity_score = None
                if relevance_score is not None:
                    # Convert Chroma distance to similarity percentage (approximate)
                    similarity_score = max(0, (1.0 - relevance_score) * 100)
                
                source_info = {
                    "source": doc.metadata.get("source", f"unknown_source_{i}"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "relevance_score": round(similarity_score, 2) if similarity_score is not None else None
                }
                sources_used.append(source_info)
            
            # Log context retrieval completion
            execution_log.append({
                "step": "context_merge",
                "status": "completed",
                "details": {
                    "total_chunks_found": len(all_results),
                    "vector_chunks": vector_count,
                    "keyword_chunks": keyword_count,
                    "duplicates_removed": len(vector_results) + len(keyword_results) - len(all_results)
                },
                "timestamp": time.time() - state.get("start_time", time.time())
            })
            
            return {
                "question": question, 
                "context": all_results, 
                "answer": None, 
                "error": None,
                "execution_log": execution_log,
                "workflow_path": workflow_path,
                "start_time": state.get("start_time", time.time()),
                "sources_used": sources_used
            }
            
        except Exception as e:
            execution_log.append({
                "step": "retrieve_context",
                "status": "error",
                "details": {"error": str(e)},
                "timestamp": time.time() - state.get("start_time", time.time())
            })
            return {
                "question": question, 
                "context": [], 
                "answer": None, 
                "error": str(e),
                "execution_log": execution_log,
                "workflow_path": workflow_path,
                "start_time": state.get("start_time", time.time()),
                "sources_used": sources_used
            }

    def _should_fallback(self, state: State) -> bool:
        """Determine if we should fall back based on context quality."""
        execution_log = state.get("execution_log", [])
        context = state.get("context", [])
        question = state.get("question", "").lower()
        
        # Log the decision process
        if len(context) == 0:
            execution_log.append({
                "step": "fallback_decision",
                "status": "triggered",
                "details": {"reason": "No context found", "context_count": 0},
                "timestamp": time.time() - state.get("start_time", time.time())
            })
            return True
        
        # Additional check: if context is very short
        total_content = " ".join(doc.page_content for doc in context)
        if len(total_content.strip()) < 100:  # Increased threshold
            execution_log.append({
                "step": "fallback_decision",
                "status": "triggered",
                "details": {
                    "reason": "Context too short", 
                    "context_count": len(context),
                    "total_content_length": len(total_content.strip())
                },
                "timestamp": time.time() - state.get("start_time", time.time())
            })
            return True
        
        # Smart relevance check: look for key question words in context
        # Better preprocessing: remove punctuation, handle short words
        import re
        clean_question = re.sub(r'[^\w\s]', '', question)  # Remove punctuation
        question_words = set(word for word in clean_question.split() if len(word) > 2)  # Allow 3+ char words
        content_words = set(word.lower() for word in total_content.split())
        
        # Calculate overlap ratio
        if question_words:
            overlap = len(question_words.intersection(content_words))
            overlap_ratio = overlap / len(question_words)
            
            # More lenient threshold for names and specific terms
            threshold = 0.1 if len(question_words) <= 2 else 0.2  # Lower threshold for short questions
            
            # If very low overlap, trigger fallback
            if overlap_ratio < threshold:
                execution_log.append({
                    "step": "fallback_decision",
                    "status": "triggered",
                    "details": {
                        "reason": "Low semantic relevance", 
                        "context_count": len(context),
                        "question_words": list(question_words),
                        "overlap_ratio": f"{overlap_ratio:.2f}",
                        "matching_words": list(question_words.intersection(content_words))
                    },
                    "timestamp": time.time() - state.get("start_time", time.time())
                })
                return True
        
        # Context is sufficient, proceed to answer generation
        execution_log.append({
            "step": "fallback_decision",
            "status": "not_triggered",
            "details": {
                "reason": "Sufficient relevant context found", 
                "context_count": len(context),
                "total_content_length": len(total_content.strip()),
                "semantic_relevance": f"{overlap_ratio:.2f}" if question_words else "N/A"
            },
            "timestamp": time.time() - state.get("start_time", time.time())
        })
        return False

    async def _generate_answer(self, state: State) -> State:
        """Generate an answer using the retrieved context."""
        execution_log = state.get("execution_log", [])
        workflow_path = state.get("workflow_path", [])
        workflow_path.append("generate_answer")
        
        execution_log.append({
            "step": "generate_answer_start",
            "status": "running",
            "details": {"message": "Generating answer using LLM with retrieved context"},
            "timestamp": time.time() - state.get("start_time", time.time())
        })
        
        context_str = "\n".join(doc.page_content for doc in state["context"])
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. 
        If the context doesn't contain enough information to answer the question, you should say so.
        
        Context:
        {context_str}
        
        Question: {state["question"]}
        
        Instructions:
        - Answer based ONLY on the provided context
        - If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer this question."
        - Be concise and accurate
        
        Answer:"""
        
        llm_start = time.time()
        response = await self.llm.ainvoke(prompt)
        llm_time = time.time() - llm_start
        
        execution_log.append({
            "step": "llm_generation",
            "status": "completed",
            "details": {
                "llm_response_time": f"{llm_time:.2f}s",
                "context_length": len(context_str),
                "prompt_length": len(prompt)
            },
            "timestamp": time.time() - state.get("start_time", time.time())
        })
        
        return {
            **state, 
            "answer": response.content,
            "execution_log": execution_log,
            "workflow_path": workflow_path
        }

    async def _fallback(self, state: State) -> State:
        """Handle cases where no relevant context is found."""
        execution_log = state.get("execution_log", [])
        workflow_path = state.get("workflow_path", [])
        workflow_path.append("fallback")
        
        execution_log.append({
            "step": "fallback_execution",
            "status": "completed",
            "details": {"message": "Executing fallback response due to insufficient context"},
            "timestamp": time.time() - state.get("start_time", time.time())
        })
        
        return {
            **state, 
            "answer": "I don't have enough information in the provided context to answer this question.",
            "execution_log": execution_log,
            "workflow_path": workflow_path
        }

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the workflow."""
        start_time = time.time()
        
        initial_state: State = {
            "question": question,
            "context": None,
            "answer": None,
            "error": None,
            "execution_log": [],
            "workflow_path": [],
            "start_time": start_time,
            "sources_used": []
        }
        
        # Log workflow start
        initial_state["execution_log"].append({
            "step": "workflow_start",
            "status": "initiated",
            "details": {"question": question, "message": "LangGraph workflow initiated"},
            "timestamp": 0.0
        })
        
        result = await self.workflow.ainvoke(initial_state)
        total_time = time.time() - start_time
        
        # Log workflow completion
        result["execution_log"].append({
            "step": "workflow_complete",
            "status": "completed",
            "details": {"total_execution_time": f"{total_time:.2f}s"},
            "timestamp": total_time
        })
        
        return {
            "answer": result["answer"] or "An error occurred while processing your question.",
            "execution_log": result.get("execution_log", []),
            "sources_used": result.get("sources_used", []),
            "workflow_path": result.get("workflow_path", []),
            "total_execution_time": total_time
        }