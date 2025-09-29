import os
import io
import logging
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import PyPDF2
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageDraw, ImageFont

from ..models.pdf_document import PDFDocument, PDFPage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: str = "us-east-1",
                 s3_bucket_name: Optional[str] = None):
        """
        Initialize PDFProcessor with optional AWS credentials for S3 support
        """
        self.aws_region = aws_region
        self.s3_bucket_name = s3_bucket_name
        self.s3_client = None

        if aws_access_key_id and aws_secret_access_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                logger.info("S3 client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.s3_client = None

    def load_from_local(self, file_path: str) -> PDFDocument:
        """
        Load PDF from local file system
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")

        logger.info(f"Loading PDF from local file: {file_path}")

        try:
            # Get file size
            file_size = os.path.getsize(file_path)

            # Extract PDF metadata
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = self._extract_pdf_metadata(pdf_reader)
                total_pages = len(pdf_reader.pages)

            # Convert PDF pages to images
            images = convert_from_path(file_path, dpi=200)

            # Create PDFPage objects
            pages = []
            for i, image in enumerate(images):
                page = PDFPage(
                    page_number=i + 1,
                    image=image,
                    metadata={"dpi": 200}
                )
                pages.append(page)

            return PDFDocument(
                file_path=file_path,
                pages=pages,
                total_pages=total_pages,
                metadata=metadata,
                file_size=file_size,
                source_type="local"
            )

        except Exception as e:
            logger.error(f"Error processing local PDF: {e}")
            raise

    def load_from_s3(self, s3_key: str, bucket_name: Optional[str] = None) -> PDFDocument:
        """
        Load PDF from S3 bucket
        """
        if not self.s3_client:
            raise ValueError("S3 client not initialized. Please provide AWS credentials.")

        bucket = bucket_name or self.s3_bucket_name
        if not bucket:
            raise ValueError("S3 bucket name not provided")

        logger.info(f"Loading PDF from S3: s3://{bucket}/{s3_key}")

        try:
            # Download file from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            pdf_content = response['Body'].read()
            file_size = len(pdf_content)

            # Extract PDF metadata
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            metadata = self._extract_pdf_metadata(pdf_reader)
            total_pages = len(pdf_reader.pages)

            # Convert PDF pages to images
            images = convert_from_bytes(pdf_content, dpi=200)

            # Create PDFPage objects
            pages = []
            for i, image in enumerate(images):
                page = PDFPage(
                    page_number=i + 1,
                    image=image,
                    metadata={"dpi": 200}
                )
                pages.append(page)

            return PDFDocument(
                file_path=f"s3://{bucket}/{s3_key}",
                pages=pages,
                total_pages=total_pages,
                metadata=metadata,
                file_size=file_size,
                source_type="s3"
            )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: s3://{bucket}/{s3_key}")
            elif error_code == 'NoSuchBucket':
                raise ValueError(f"S3 bucket not found: {bucket}")
            else:
                raise
        except Exception as e:
            logger.error(f"Error processing S3 PDF: {e}")
            raise

    def load_from_url(self, url: str) -> PDFDocument:
        """
        Load PDF from URL (determines if it's S3 or local based on URL format)
        """
        if url.startswith('s3://'):
            # Parse S3 URL
            url_parts = url[5:].split('/', 1)
            if len(url_parts) != 2:
                raise ValueError("Invalid S3 URL format. Expected: s3://bucket/key")
            bucket, key = url_parts
            return self.load_from_s3(key, bucket)
        elif url.startswith(('http://', 'https://')):
            raise NotImplementedError("HTTP/HTTPS URLs not yet supported")
        else:
            # Assume local file path
            return self.load_from_local(url)

    def _extract_pdf_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """
        Extract metadata from PDF
        """
        metadata = {}

        if pdf_reader.metadata:
            metadata.update({
                'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creator': pdf_reader.metadata.get('/Creator', ''),
                'producer': pdf_reader.metadata.get('/Producer', ''),
                'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                'modification_date': pdf_reader.metadata.get('/ModDate', '')
            })

        return metadata

    def add_page_number_to_image(self, image: Image.Image, page_number: int) -> Image.Image:
        """
        Overlay page number on an image for clarity.
        """
        try:
            # Create a copy of the image to avoid modifying the original
            img = image.copy()
            draw = ImageDraw.Draw(img)

            # Try to use a system font with smaller size, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # Reduced from 40 to 20
            except (IOError, OSError):
                try:
                    # Try other common font names
                    font = ImageFont.truetype("Arial.ttf", 20)  # Windows
                except (IOError, OSError):
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)  # macOS
                    except (IOError, OSError):
                        # Use default font if no system font is available
                        font = ImageFont.load_default()

            margin = 10  # Reduced margin
            padding = 6   # Reduced padding
            text = f"{page_number}"  # Just the number, no "Page" text

            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text in top-right corner
            x = img.width - text_width - margin - padding
            y = margin

            # Draw background rectangle with blue color
            rect_x1, rect_y1, rect_x2, rect_y2 = (
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            )
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2],
                          fill="#007bff", outline="#0056b3", width=1)  # Blue background with darker blue border

            # Draw text in white
            draw.text((x, y), text, fill="white", font=font)

            return img

        except Exception as e:
            logger.error(f"Error annotating image with page number {page_number}: {e}")
            return image

    def get_pages_as_images(self, document: PDFDocument, add_page_numbers: bool = True,
                           max_size: tuple = (800, 600)) -> List[Image.Image]:
        """
        Get all pages as images with optional page number overlay
        """
        images = []
        for page in document.pages:
            if not page.image:
                logger.warning(f"No image available for page {page.page_number}")
                continue

            # Create a copy and resize for preview
            image = page.image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Add page number overlay if requested
            if add_page_numbers:
                image = self.add_page_number_to_image(image, page.page_number)

            images.append(image)

        return images

    def get_page_preview(self, document: PDFDocument, page_number: int,
                        max_size: tuple = (800, 600)) -> Image.Image:
        """
        Get a preview image of a specific page
        """
        if page_number < 1 or page_number > document.total_pages:
            raise ValueError(f"Page number {page_number} out of range (1-{document.total_pages})")

        page = document.pages[page_number - 1]
        if not page.image:
            raise ValueError(f"No image available for page {page_number}")

        # Resize image for preview
        image = page.image.copy()
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    def save_page_as_image(self, document: PDFDocument, page_number: int,
                          output_path: str, format: str = 'PNG') -> str:
        """
        Save a specific page as an image file
        """
        if page_number < 1 or page_number > document.total_pages:
            raise ValueError(f"Page number {page_number} out of range (1-{document.total_pages})")

        page = document.pages[page_number - 1]
        if not page.image:
            raise ValueError(f"No image available for page {page_number}")

        page.image.save(output_path, format=format)
        return output_path

    def save_images_with_lifecycle_manager(self, document: PDFDocument, lifecycle_manager, document_id: str) -> List[Dict[str, str]]:
        """
        Save all page images to S3 using the DocumentLifecycleManager.

        Args:
            document: PDFDocument with pages containing images
            lifecycle_manager: DocumentLifecycleManager instance
            document_id: Document ID for tracking

        Returns:
            List of page image information dictionaries
        """
        try:
            # Extract images from document pages
            images = [page.image for page in document.pages if page.image is not None]

            if not images:
                logger.warning(f"No images found in document {document_id}")
                return []

            # Track the start of PDF processing stage
            lifecycle_manager.track_stage_start(
                document_id=document_id,
                stage_name="pdf_processing",
                stage_data={
                    "total_pages": len(images),
                    "processing_dpi": 200,
                    "image_format": "PNG"
                }
            )

            # Save images using lifecycle manager
            page_images = lifecycle_manager.save_page_images(
                document_id=document_id,
                images=images,
                create_thumbnails=True
            )

            # Update document pages with S3 URLs
            for i, page_image_info in enumerate(page_images):
                if i < len(document.pages):
                    document.pages[i].s3_url = page_image_info.s3_original_url
                    document.pages[i].s3_thumbnail_url = page_image_info.s3_thumbnail_url

            # Track completion of PDF processing stage
            lifecycle_manager.track_stage_completion(
                document_id=document_id,
                stage_name="pdf_processing",
                stage_data={
                    "images_saved": len(page_images),
                    "s3_folder": page_images[0].s3_original_url.rsplit('/', 1)[0] if page_images else None
                }
            )

            logger.info(f"Saved {len(page_images)} images to S3 for document {document_id}")

            # Return information for compatibility
            return [
                {
                    "page_number": img.page_number,
                    "s3_url": img.s3_original_url,
                    "s3_thumbnail_url": img.s3_thumbnail_url or "",
                    "dimensions": img.image_dimensions
                }
                for img in page_images
            ]

        except Exception as e:
            logger.error(f"Failed to save images for document {document_id}: {e}")
            # Track error in lifecycle manager
            if lifecycle_manager:
                lifecycle_manager.track_stage_error(
                    document_id=document_id,
                    stage_name="pdf_processing",
                    error_message=str(e)
                )
            raise

    def get_document_info(self, document: PDFDocument) -> Dict[str, Any]:
        """
        Get comprehensive document information
        """
        return {
            'file_path': document.file_path,
            'source_type': document.source_type,
            'total_pages': document.total_pages,
            'file_size': document.file_size,
            'file_size_mb': round(document.file_size / (1024 * 1024), 2) if document.file_size else None,
            'metadata': document.metadata,
            'pages_info': [
                {
                    'page_number': page.page_number,
                    'has_image': page.image is not None,
                    'image_size': page.image.size if page.image else None,
                    's3_url': getattr(page, 's3_url', None),
                    's3_thumbnail_url': getattr(page, 's3_thumbnail_url', None)
                }
                for page in document.pages
            ]
        }