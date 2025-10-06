from dotenv import load_dotenv
import os
from langchain_nomic import NomicEmbeddings

load_dotenv()

class NomicEmbeddingsService(NomicEmbeddings):
    def __init__(self, model_name="nomic-embed-text-v1.5"):
        if not os.getenv("NOMIC_API_KEY"):
            raise ValueError("NOMIC_API_KEY not found in environment variables")
        
        # Just call super with the correct params that are accepted
        super().__init__(model=model_name)
