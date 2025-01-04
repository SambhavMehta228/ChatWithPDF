from typing import Dict, Tuple, List
import pdfplumber
import re
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from utils.text_processing import TextProcessor, ChunkCreator

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_creator = ChunkCreator(chunk_size, chunk_overlap)
        self.text_processor = TextProcessor()
    
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, Dict[int, str]]:
        """Extract text with page mapping for source tracking."""
        text = ""
        page_map = {}
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    cleaned_text = self.text_processor.clean_text(page_text)
                    if cleaned_text:
                        page_map[page_num] = cleaned_text
                        text += f"\nPage {page_num}: {cleaned_text}"
        except Exception as e:
            # Fallback to PyPDF2
            pdf_reader = PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                cleaned_text = self.text_processor.clean_text(page_text)
                if cleaned_text:
                    page_map[page_num] = cleaned_text
                    text += f"\nPage {page_num}: {cleaned_text}"
        
        return text.strip(), page_map
    
    def create_documents(self, text: str, metadata: Dict = None) -> List[Document]:
        """Create document objects with metadata."""
        chunks = self.chunk_creator.create_chunks(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Extract page number if available
            page_match = re.search(r'Page (\d+):', chunk)
            page_num = int(page_match.group(1)) if page_match else None
            
            # Create document with metadata
            doc_metadata = {
                'chunk_id': i,
                'page': page_num,
                'density': self.text_processor.calculate_text_density(chunk)
            }
            if metadata:
                doc_metadata.update(metadata)
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_pdf(self, pdf_file) -> Tuple[List[Document], Dict[int, str]]:
        """Process PDF file and return documents and page mapping."""
        text, page_map = self.extract_text_from_pdf(pdf_file)
        documents = self.create_documents(text)
        return documents, page_map