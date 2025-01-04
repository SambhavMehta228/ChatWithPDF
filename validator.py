from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processing import TextProcessor

class AdvancedValidator:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_processor = TextProcessor()
        self.tfidf = TfidfVectorizer(stop_words='english')
    
    def semantic_similarity(self, query: str, context: str) -> float:
        """Calculate semantic similarity using embeddings."""
        query_embedding = self.embeddings.embed_query(query)
        context_embedding = self.embeddings.embed_query(context)
        similarity = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(context_embedding).reshape(1, -1)
        )[0][0]
        return float(similarity)
    
    def keyword_similarity(self, query: str, context: str) -> float:
        """Calculate keyword-based similarity."""
        query_tokens = self.text_processor.get_tokens(query)
        context_tokens = self.text_processor.get_tokens(context)
        
        if not query_tokens:
            return 0.0
        
        overlap = len(query_tokens.intersection(context_tokens))
        return overlap / len(query_tokens)
    
    def tfidf_similarity(self, query: str, context: str) -> float:
        """Calculate TF-IDF based similarity."""
        try:
            tfidf_matrix = self.tfidf.fit_transform([query, context])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except:
            return 0.0
    
    def structural_validity(self, context: str) -> float:
        """Assess structural validity of the context."""
        sentences = self.text_processor.get_sentences(context)
        if not sentences:
            return 0.0
        
        valid_sentences = sum(
            1 for sent in sentences
            if self.text_processor.check_sentence_structure(sent)
        )
        return valid_sentences / len(sentences)
    
    def calculate_context_scores(self, query: str, context: str) -> Dict[str, float]:
        """Calculate comprehensive context scoring."""
        scores = {
            'semantic_similarity': self.semantic_similarity(query, context),
            'keyword_similarity': self.keyword_similarity(query, context),
            'tfidf_similarity': self.tfidf_similarity(query, context),
            'structural_validity': self.structural_validity(context),
            'content_density': self.text_processor.calculate_text_density(context)
        }
        
        # Calculate weighted final score
        weights = {
            'semantic_similarity': 0.35,
            'keyword_similarity': 0.25,
            'tfidf_similarity': 0.20,
            'structural_validity': 0.10,
            'content_density': 0.10
        }
        
        scores['final_score'] = sum(
            scores[metric] * weight
            for metric, weight in weights.items()
        )
        
        return scores
    
    def validate_and_rank_contexts(
        self,
        query: str,
        contexts: List[str],
        threshold: float = 0.6
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Validate and rank contexts by relevance."""
        scored_contexts = []
        
        for context in contexts:
            scores = self.calculate_context_scores(query, context)
            if scores['final_score'] >= threshold:
                scored_contexts.append((context, scores))
        
        return sorted(
            scored_contexts,
            key=lambda x: x[1]['final_score'],
            reverse=True
        )