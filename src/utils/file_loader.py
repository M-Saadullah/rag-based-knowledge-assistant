import os
from typing import Union
import mimetypes
import io
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class FileLoader:
    """Utility class for loading and processing different file types."""
    
    @staticmethod
    def load_text_file(content: bytes, filename: str) -> str:
        """Load content from a text-based file."""
        # Determine file type
        mime_type, _ = mimetypes.guess_type(filename)
        
        if mime_type and mime_type.startswith('text/'):
            # Handle text files
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings
                for encoding in ['latin-1', 'cp1252']:
                    try:
                        return content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"Unable to decode file {filename}")
        
        elif filename.lower().endswith(('.md', '.markdown')):
            # Handle Markdown files
            return content.decode('utf-8')
        
        elif filename.lower().endswith('.pdf'):
            # Handle PDF files
            return FileLoader._extract_pdf_text(content)
        
        else:
            # Try to decode as text
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type: {filename}")
    
    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        """Extract text from PDF content using multiple methods."""
        if not PDF_AVAILABLE:
            raise ValueError("PDF processing libraries not installed. Please install pypdf2 and pdfplumber.")
        
        # Try pdfplumber first (better text extraction)
        try:
            text = FileLoader._extract_with_pdfplumber(content)
            if text.strip():
                return text
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        try:
            text = FileLoader._extract_with_pypdf2(content)
            if text.strip():
                return text
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
        
        raise ValueError("Failed to extract text from PDF. The PDF might be image-based or corrupted.")
    
    @staticmethod
    def _extract_with_pdfplumber(content: bytes) -> str:
        """Extract text using pdfplumber (better for complex layouts)."""
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return '\n\n'.join(text_parts)
    
    @staticmethod
    def _extract_with_pypdf2(content: bytes) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        text_parts = []
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        return '\n\n'.join(text_parts)
    
    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """Validate if the file type is supported."""
        supported_extensions = ['.txt', '.md', '.markdown', '.pdf']
        return any(filename.lower().endswith(ext) for ext in supported_extensions)
