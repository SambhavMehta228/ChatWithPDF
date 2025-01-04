import re
import nltk
from typing import List, Set
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Advanced text cleaning to reduce noise."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove page numbers
        text = re.sub(r'\b\d+\b(?!\s*(?:st|nd|rd|th|years?|dollars?|%))', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,!?;:()"\'%-]', '', text)
        return text.strip()
    
    @staticmethod
    def get_tokens(text: str) -> Set[str]:
        """Get meaningful tokens from text."""
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        return {token for token in tokens if token not in stop_words and len(token) > 2}
    
    @staticmethod
    def get_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)
    
    @staticmethod
    def check_sentence_structure(sentence: str) -> bool:
        """Check if sentence has valid structure."""
        try:
            tokens = nltk.pos_tag(word_tokenize(sentence))
            has_noun = any(tag.startswith('NN') for _, tag in tokens)
            has_verb = any(tag.startswith('VB') for _, tag in tokens)
            return has_noun and has_verb
        except Exception:
            return True  # Default to True if parsing fails
    
    @staticmethod
    def calculate_text_density(text: str) -> float:
        """Calculate information density of text."""
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        meaningful_words = [w for w in words if w not in stop_words and w.isalnum()]
        return len(meaningful_words) / len(words) if words else 0

class ChunkCreator:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks of text."""
        sentences = TextProcessor.get_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Keep overlap sentences
                    overlap_text = current_chunk[-2:]  # Keep last 2 sentences
                    current_chunk = overlap_text + [sentence]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    chunks.append(sentence)
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks