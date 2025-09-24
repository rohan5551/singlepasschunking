#!/usr/bin/env python3
"""
Test script for PDFSplitter functionality
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.processors import PDFProcessor, PDFSplitter
from src.models import SplitConfiguration

def test_splitter():
    """Test PDF splitting functionality"""
    print("🔧 Testing PDF Splitter...")
    print("=" * 50)

    # Get PDF path from user
    pdf_path = input("Enter path to a PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False

    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        pdf_splitter = PDFSplitter()

        # Load document
        print(f"\n📄 Loading PDF: {pdf_path}")
        document = pdf_processor.load_from_local(pdf_path)
        print(f"✅ Loaded document with {document.total_pages} pages")

        # Test different configurations
        configs = [
            SplitConfiguration(batch_size=4, overlap_pages=0),
            SplitConfiguration(batch_size=3, overlap_pages=1),
            SplitConfiguration(batch_size=5, overlap_pages=2),
        ]

        for i, config in enumerate(configs, 1):
            print(f"\n🧪 Test {i}: Batch size={config.batch_size}, Overlap={config.overlap_pages}")

            # Split document
            result = pdf_splitter.split_document(document, config)

            # Display results
            print(f"✅ Created {result.total_batches} batches")

            # Show batch details
            for batch in result.batches:
                print(f"   Batch {batch.batch_number}: Pages {batch.start_page}-{batch.end_page} "
                      f"({batch.page_count} pages)")

            # Get summary
            summary = pdf_splitter.get_batch_summary(result)
            print(f"   📊 Average batch size: {summary['average_batch_size']:.1f}")
            print(f"   📊 Min/Max batch size: {summary['min_batch_size']}/{summary['max_batch_size']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_validation():
    """Test configuration validation"""
    print("\n🔧 Testing Configuration Validation...")
    print("=" * 50)

    # Create a dummy document for testing
    from src.models.pdf_document import PDFDocument, PDFPage
    dummy_document = PDFDocument(
        file_path="test.pdf",
        pages=[],
        total_pages=10,
        metadata={},
        source_type="local"
    )

    pdf_splitter = PDFSplitter()

    test_configs = [
        SplitConfiguration(batch_size=0, overlap_pages=0),  # Invalid: batch_size <= 0
        SplitConfiguration(batch_size=5, overlap_pages=5),  # Invalid: overlap >= batch_size
        SplitConfiguration(batch_size=-1, overlap_pages=0), # Invalid: batch_size < 0
        SplitConfiguration(batch_size=15, overlap_pages=0), # Invalid: batch_size > total_pages
        SplitConfiguration(batch_size=4, overlap_pages=1),  # Valid
    ]

    for i, config in enumerate(test_configs, 1):
        print(f"\nTest {i}: batch_size={config.batch_size}, overlap={config.overlap_pages}")

        errors = pdf_splitter.validate_configuration(config, dummy_document)
        if errors:
            print(f"❌ Validation errors: {', '.join(errors)}")
        else:
            print("✅ Configuration is valid")

def interactive_test():
    """Interactive testing interface"""
    print("🎮 Interactive PDF Splitter Test")
    print("=" * 50)

    while True:
        print("\nChoose an option:")
        print("1. Test PDF splitting")
        print("2. Test configuration validation")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            test_splitter()
        elif choice == '2':
            test_validation()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    interactive_test()