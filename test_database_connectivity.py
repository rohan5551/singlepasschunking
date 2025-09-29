#!/usr/bin/env python3
"""Test script to validate database connectivity and document lifecycle tracking."""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

# Import our components
try:
    from src.database.mongodb_manager import MongoDBManager
    from src.storage.s3_manager import S3StorageManager
    from src.database.document_lifecycle_models import ProcessingStatus

    # For DocumentLifecycleManager, we'll test components separately first
    mongodb_manager = None
    s3_manager = None

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_mongodb_connection():
    """Test MongoDB connection and operations."""
    print("\n=== Testing MongoDB Connection ===")

    try:
        # Initialize MongoDB manager
        mongodb_manager = MongoDBManager()

        # Test connection
        status = mongodb_manager.get_connection_status()
        print(f"MongoDB Status: {status}")

        if status['connected']:
            print("✅ MongoDB connection successful")

            # List existing documents
            documents = mongodb_manager.list_documents(limit=5)
            print(f"📋 Found {len(documents)} existing documents")

            for doc in documents:
                print(f"  - Document ID: {doc.document_id}")
                print(f"    Status: {doc.current_status}")
                print(f"    Created: {doc.created_at}")
                print(f"    Filename: {doc.original_filename}")

                # Get statistics for this document
                stats = mongodb_manager.get_document_statistics(doc.document_id)
                print(f"    Stats: {stats['chunks_count']} chunks, {stats['batches_count']} batches")
                print()

            return True
        else:
            print("❌ MongoDB connection failed")
            return False

    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        return False


def test_s3_connection():
    """Test S3 connection and operations."""
    print("\n=== Testing S3 Connection ===")

    try:
        # Initialize S3 manager
        s3_manager = S3StorageManager()

        # Test connection
        status = s3_manager.get_connection_status()
        print(f"S3 Status: {status}")

        if status['connected']:
            print("✅ S3 connection successful")
            print(f"📦 Bucket: {status['bucket_name']}")
            print(f"🌍 Region: {status['region']}")

            # Test folder name integration
            print(f"📁 Folder: {s3_manager.folder_name}")

            return True
        else:
            print("❌ S3 connection failed")
            return False

    except Exception as e:
        print(f"❌ S3 test failed: {e}")
        return False


def test_lifecycle_integration():
    """Test integrated functionality of MongoDB and S3 managers."""
    print("\n=== Testing Lifecycle Integration ===")

    try:
        # Test creating a simple document record directly
        mongodb_mgr = MongoDBManager()
        s3_mgr = S3StorageManager()

        print("✅ Both managers initialized successfully")

        # Create a test document record using the new models
        from src.database.document_lifecycle_models import DocumentRecord

        test_filename = "test_connectivity_document.pdf"
        document_id = s3_mgr.generate_document_id(test_filename)

        print(f"📄 Generated test document ID: {document_id}")

        # Create document record
        doc_record = DocumentRecord(
            document_id=document_id,
            session_id=f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_filename=test_filename,
            s3_pdf_url="s3://test/path.pdf",  # Mock URL
            current_status=ProcessingStatus.PROCESSING,
            pdf_metadata={"test": True, "connectivity_check": True},
            total_batches=0,
            total_chunks=0
        )

        # Save to database
        success = mongodb_mgr.create_document(doc_record)

        if success:
            print(f"✅ Test document saved to database: {document_id}")

            # Retrieve and verify
            retrieved_doc = mongodb_mgr.get_document(document_id)
            if retrieved_doc:
                print(f"📊 Document status: {retrieved_doc.current_status}")
                print(f"📁 Original filename: {retrieved_doc.original_filename}")
                print(f"🆔 Session ID: {retrieved_doc.session_id}")

                # Clean up test document
                mongodb_mgr.delete_document_data(document_id)
                print("🧹 Test document cleaned up")

                return True
            else:
                print("❌ Failed to retrieve test document")
                return False
        else:
            print("❌ Failed to save test document")
            return False

    except Exception as e:
        print(f"❌ Lifecycle integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variables():
    """Test that all required environment variables are available."""
    print("\n=== Testing Environment Variables ===")

    required_vars = [
        'MONGODB_URI',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'S3_BUCKET_NAME',
        'S3_FOLDER_NAME',
        'OPENROUTER_API_KEY'
    ]

    all_present = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'URI' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
            all_present = False

    return all_present


def main():
    """Run all connectivity tests."""
    print("🧪 Document Processing System - Connectivity Test")
    print("=" * 50)

    # Test environment variables
    env_ok = test_environment_variables()

    # Test MongoDB connection
    mongodb_ok = test_mongodb_connection()

    # Test S3 connection
    s3_ok = test_s3_connection()

    # Test lifecycle integration
    lifecycle_ok = test_lifecycle_integration()

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    print("=" * 50)
    print(f"Environment Variables: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"MongoDB Connection:    {'✅ PASS' if mongodb_ok else '❌ FAIL'}")
    print(f"S3 Connection:         {'✅ PASS' if s3_ok else '❌ FAIL'}")
    print(f"Lifecycle Manager:     {'✅ PASS' if lifecycle_ok else '❌ FAIL'}")

    all_tests_passed = env_ok and mongodb_ok and s3_ok and lifecycle_ok

    if all_tests_passed:
        print("\n🎉 All tests passed! The system is ready for document processing.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and connections.")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)