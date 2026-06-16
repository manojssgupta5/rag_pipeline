from typing import List
from sentence_transformers import CrossEncoder
from ..base import RetrievedChunk

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def __call__(self, query: str, results: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not results:
            return []
        
        # Prepare pairs for the cross-encoder
        pairs = [[query, chunk.text] for chunk in results]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Update the scores and re-sort the chunks
        for chunk, score in zip(results, scores):
            chunk.score = float(score)  # Replace the fusion score with the cross-encoder score
            chunk.matched_via = "fused_and_reranked"
            
        reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked_results
