#!/usr/bin/env python3
"""
Simple test script for the PDFProcessor class
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.processors import PDFProcessor

def test_local_pdf():
    """Test loading a PDF from local file system"""
    print("Testing local PDF processing...")

    # Get a sample PDF path from user
    pdf_path = input("Enter the path to a local PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return False

    try:
        processor = PDFProcessor()
        document = processor.load_from_local(pdf_path)

        print(f"\n✅ Successfully processed: {pdf_path}")
        print(f"📄 Total pages: {document.total_pages}")
        print(f"📊 File size: {document.file_size} bytes")
        print(f"📋 Metadata: {document.metadata}")

        # Get document info
        doc_info = processor.get_document_info(document)
        print(f"\n📖 Document Info:")
        for key, value in doc_info.items():
            if key != 'pages_info':
                print(f"   {key}: {value}")

        # Save preview of first page
        if document.pages:
            preview_path = "first_page_preview.png"
            processor.save_page_as_image(document, 1, preview_path)
            print(f"💾 Saved first page preview to: {preview_path}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_s3_pdf():
    """Test loading a PDF from S3"""
    print("\nTesting S3 PDF processing...")

    # Load environment variables
    load_dotenv()

    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket = os.getenv('S3_BUCKET_NAME')

    if not all([aws_key, aws_secret, bucket]):
        print("❌ AWS credentials not found in .env file")
        print("Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME")
        return False

    s3_key = input("Enter the S3 key (path) to a PDF file: ").strip()

    try:
        processor = PDFProcessor(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            s3_bucket_name=bucket
        )

        document = processor.load_from_s3(s3_key)

        print(f"\n✅ Successfully processed: s3://{bucket}/{s3_key}")
        print(f"📄 Total pages: {document.total_pages}")
        print(f"📊 File size: {document.file_size} bytes")
        print(f"📋 Metadata: {document.metadata}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔍 PDF Processor Test Script")
    print("=" * 40)

    while True:
        print("\nChoose an option:")
        print("1. Test local PDF file")
        print("2. Test S3 PDF file")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            test_local_pdf()
        elif choice == '2':
            test_s3_pdf()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()