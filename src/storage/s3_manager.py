"""S3 Storage Manager for handling PDF and image uploads."""

import os
import io
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class S3StorageManager:
    """
    Manages S3 storage for PDFs and page images with organized folder structure.

    Folder structure:
    s3://bucket/syia_documents/singlepasschunking/YYYY/MM/DD/doc_YYYY_MMDD_NNN_hash8chars/
    ├── original.pdf
    ├── metadata.json
    ├── images/
    │   ├── page_001.png
    │   ├── page_002.png
    │   └── ...
    ├── thumbnails/
    │   ├── page_001_thumb.png
    │   ├── page_002_thumb.png
    │   └── ...
    └── processing_artifacts/
        └── batch_configs.json
    """

    def __init__(self,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: Optional[str] = None,
                 bucket_name: Optional[str] = None,
                 folder_name: Optional[str] = None):
        """
        Initialize S3StorageManager.

        Args:
            aws_access_key_id: AWS access key (defaults to env var)
            aws_secret_access_key: AWS secret key (defaults to env var)
            aws_region: AWS region (defaults to env var)
            bucket_name: S3 bucket name (defaults to env var)
            folder_name: S3 folder name (defaults to env var or 'documents')
        """
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        self.folder_name = folder_name or os.getenv('S3_FOLDER_NAME', 'syia_documents/singlepasschunking')

        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME must be provided or set as environment variable")

        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=aws_region or os.getenv('AWS_REGION', 'us-east-1')
            )
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3StorageManager initialized with bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Failed to access S3 bucket {self.bucket_name}: {e}")
            raise

    def generate_document_id(self, filename: str, timestamp: Optional[datetime] = None) -> str:
        """
        Generate unique document ID with timestamp and hash.

        Args:
            filename: Original filename
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Document ID in format: doc_YYYY_MMDD_NNN_hash8chars
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Create hash from filename and timestamp
        content = f"{filename}_{timestamp.isoformat()}"
        hash_obj = hashlib.sha256(content.encode())
        hash_short = hash_obj.hexdigest()[:8]

        # Generate sequential number (simplified - in production might use database counter)
        seq_num = timestamp.hour * 100 + timestamp.minute

        return f"doc_{timestamp.strftime('%Y_%m%d')}_{seq_num:03d}_{hash_short}"

    def create_document_folder_structure(self, document_id: str, timestamp: Optional[datetime] = None) -> str:
        """
        Create the base folder structure for a document in S3.

        Args:
            document_id: Unique document identifier
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Base S3 path for the document
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        base_path = f"{self.folder_name}/{timestamp.strftime('%Y/%m/%d')}/{document_id}"

        # Create folder structure by uploading empty objects
        folders = [
            f"{base_path}/images/",
            f"{base_path}/thumbnails/",
            f"{base_path}/processing_artifacts/"
        ]

        for folder in folders:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=folder,
                    Body=b'',
                    ContentType='application/x-directory'
                )
            except ClientError as e:
                logger.warning(f"Could not create folder {folder}: {e}")

        logger.info(f"Created folder structure for document {document_id}")
        return base_path

    def upload_pdf(self,
                   pdf_path: str,
                   document_id: str,
                   base_path: str,
                   metadata: Optional[Dict] = None) -> str:
        """
        Upload PDF file to S3.

        Args:
            pdf_path: Local path to PDF file
            document_id: Unique document identifier
            base_path: Base S3 path for the document
            metadata: Optional metadata to attach

        Returns:
            S3 URL of uploaded PDF
        """
        s3_key = f"{base_path}/original.pdf"

        try:
            # Prepare metadata
            s3_metadata = {
                'document-id': document_id,
                'original-filename': os.path.basename(pdf_path),
                'upload-timestamp': datetime.utcnow().isoformat(),
            }

            if metadata:
                for key, value in metadata.items():
                    # S3 metadata keys must be lowercase and contain only valid characters
                    clean_key = key.lower().replace(' ', '-').replace('_', '-')
                    s3_metadata[clean_key] = str(value)

            # Get file size
            file_size = os.path.getsize(pdf_path)

            # Upload PDF
            with open(pdf_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'application/pdf',
                        'Metadata': s3_metadata
                    }
                )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Uploaded PDF {pdf_path} to {s3_url} ({file_size} bytes)")
            return s3_url

        except ClientError as e:
            logger.error(f"Failed to upload PDF {pdf_path}: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise

    def upload_page_images(self,
                          images: List[Image.Image],
                          document_id: str,
                          base_path: str,
                          create_thumbnails: bool = True,
                          thumbnail_size: Tuple[int, int] = (200, 300)) -> List[Dict[str, str]]:
        """
        Upload page images to S3 with optional thumbnail generation.

        Args:
            images: List of PIL Image objects
            document_id: Unique document identifier
            base_path: Base S3 path for the document
            create_thumbnails: Whether to create thumbnails
            thumbnail_size: Thumbnail dimensions (width, height)

        Returns:
            List of image info dictionaries with S3 URLs
        """
        image_info = []

        for i, image in enumerate(images):
            page_num = i + 1
            page_key = f"{base_path}/images/page_{page_num:03d}.png"

            try:
                # Upload original image
                original_url = self._upload_image(image, page_key, document_id, page_num)

                info = {
                    'page_number': page_num,
                    'original_url': original_url,
                    'thumbnail_url': None,
                    'dimensions': {'width': image.width, 'height': image.height}
                }

                # Create and upload thumbnail if requested
                if create_thumbnails:
                    thumbnail = image.copy()
                    thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                    thumb_key = f"{base_path}/thumbnails/page_{page_num:03d}_thumb.png"
                    thumbnail_url = self._upload_image(thumbnail, thumb_key, document_id, page_num, is_thumbnail=True)
                    info['thumbnail_url'] = thumbnail_url
                    info['thumbnail_dimensions'] = {'width': thumbnail.width, 'height': thumbnail.height}

                image_info.append(info)

            except Exception as e:
                logger.error(f"Failed to upload page {page_num} for document {document_id}: {e}")
                # Continue with other pages
                continue

        logger.info(f"Uploaded {len(image_info)} page images for document {document_id}")
        return image_info

    def _upload_image(self,
                     image: Image.Image,
                     s3_key: str,
                     document_id: str,
                     page_number: int,
                     is_thumbnail: bool = False) -> str:
        """
        Upload a single image to S3.

        Args:
            image: PIL Image object
            s3_key: S3 key for the image
            document_id: Document identifier
            page_number: Page number
            is_thumbnail: Whether this is a thumbnail image

        Returns:
            S3 URL of uploaded image
        """
        # Convert image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)

        # Prepare metadata
        metadata = {
            'document-id': document_id,
            'page-number': str(page_number),
            'image-type': 'thumbnail' if is_thumbnail else 'original',
            'width': str(image.width),
            'height': str(image.height),
            'upload-timestamp': datetime.utcnow().isoformat()
        }

        try:
            self.s3_client.upload_fileobj(
                img_buffer,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png',
                    'Metadata': metadata
                }
            )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            return s3_url

        except ClientError as e:
            logger.error(f"Failed to upload image {s3_key}: {e}")
            raise

    def upload_processing_artifacts(self,
                                  document_id: str,
                                  base_path: str,
                                  artifacts: Dict[str, any]) -> Dict[str, str]:
        """
        Upload processing artifacts (configs, logs, etc.) to S3.

        Args:
            document_id: Document identifier
            base_path: Base S3 path for the document
            artifacts: Dictionary of artifact name -> content

        Returns:
            Dictionary of artifact name -> S3 URL
        """
        uploaded_artifacts = {}

        for name, content in artifacts.items():
            artifact_key = f"{base_path}/processing_artifacts/{name}"

            try:
                # Convert content to JSON string if it's a dict/list
                if isinstance(content, (dict, list)):
                    import json
                    content_bytes = json.dumps(content, indent=2).encode('utf-8')
                    content_type = 'application/json'
                elif isinstance(content, str):
                    content_bytes = content.encode('utf-8')
                    content_type = 'text/plain'
                else:
                    content_bytes = str(content).encode('utf-8')
                    content_type = 'text/plain'

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=artifact_key,
                    Body=content_bytes,
                    ContentType=content_type,
                    Metadata={
                        'document-id': document_id,
                        'artifact-name': name,
                        'upload-timestamp': datetime.utcnow().isoformat()
                    }
                )

                s3_url = f"s3://{self.bucket_name}/{artifact_key}"
                uploaded_artifacts[name] = s3_url

            except Exception as e:
                logger.error(f"Failed to upload artifact {name}: {e}")
                continue

        logger.info(f"Uploaded {len(uploaded_artifacts)} artifacts for document {document_id}")
        return uploaded_artifacts

    def download_pdf(self, s3_url: str, local_path: str) -> bool:
        """
        Download PDF from S3 to local path.

        Args:
            s3_url: S3 URL of the PDF
            local_path: Local path to save the PDF

        Returns:
            True if successful
        """
        try:
            s3_key = self._extract_s3_key(s3_url)

            with open(local_path, 'wb') as file:
                self.s3_client.download_fileobj(self.bucket_name, s3_key, file)

            logger.info(f"Downloaded PDF from {s3_url} to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download PDF from {s3_url}: {e}")
            return False

    def get_document_info(self, document_id: str) -> Optional[Dict]:
        """
        Get information about a document stored in S3.

        Args:
            document_id: Document identifier

        Returns:
            Document information or None if not found
        """
        try:
            # Search for documents with this ID
            prefix = f"{self.folder_name}/"

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            for obj in response.get('Contents', []):
                if document_id in obj['Key'] and obj['Key'].endswith('original.pdf'):
                    # Get object metadata
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )

                    metadata = head_response.get('Metadata', {})

                    return {
                        'document_id': document_id,
                        's3_key': obj['Key'],
                        's3_url': f"s3://{self.bucket_name}/{obj['Key']}",
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'metadata': metadata
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to get document info for {document_id}: {e}")
            return None

    def list_document_images(self, document_id: str, base_path: str) -> List[Dict]:
        """
        List all images for a document.

        Args:
            document_id: Document identifier
            base_path: Base S3 path for the document

        Returns:
            List of image information dictionaries
        """
        try:
            images = []

            # List original images
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{base_path}/images/"
            )

            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.png'):
                    # Extract page number from filename
                    filename = os.path.basename(obj['Key'])
                    if filename.startswith('page_'):
                        page_num_str = filename.split('_')[1].split('.')[0]
                        page_number = int(page_num_str)

                        images.append({
                            'page_number': page_number,
                            'original_url': f"s3://{self.bucket_name}/{obj['Key']}",
                            'size': obj['Size'],
                            'last_modified': obj['LastModified']
                        })

            # Sort by page number
            images.sort(key=lambda x: x['page_number'])

            return images

        except Exception as e:
            logger.error(f"Failed to list images for document {document_id}: {e}")
            return []

    def delete_document(self, base_path: str) -> bool:
        """
        Delete all files for a document.

        Args:
            base_path: Base S3 path for the document

        Returns:
            True if successful
        """
        try:
            # List all objects with this prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=base_path
            )

            objects_to_delete = []
            for obj in response.get('Contents', []):
                objects_to_delete.append({'Key': obj['Key']})

            if objects_to_delete:
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': objects_to_delete}
                )

                logger.info(f"Deleted {len(objects_to_delete)} objects for path {base_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete document at path {base_path}: {e}")
            return False

    def _extract_s3_key(self, s3_url: str) -> str:
        """Extract S3 key from S3 URL."""
        if s3_url.startswith(f's3://{self.bucket_name}/'):
            return s3_url.replace(f's3://{self.bucket_name}/', '')
        else:
            raise ValueError(f"Invalid S3 URL: {s3_url}")

    def get_connection_status(self) -> Dict[str, any]:
        """
        Get S3 connection status.

        Returns:
            Connection status information
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return {
                'connected': True,
                'bucket_name': self.bucket_name,
                'region': self.s3_client.meta.region_name,
                'error': None
            }
        except Exception as e:
            return {
                'connected': False,
                'bucket_name': self.bucket_name,
                'region': None,
                'error': str(e)
            }