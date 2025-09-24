from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from PIL import Image
import io

@dataclass
class PDFPage:
    page_number: int
    image: Optional[Image.Image] = None
    text_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PDFDocument:
    file_path: str
    pages: List[PDFPage]
    total_pages: int
    metadata: Dict[str, Any]
    file_size: Optional[int] = None
    source_type: str = "local"  # "local" or "s3"