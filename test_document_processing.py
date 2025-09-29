#!/usr/bin/env python3
"""Test document processing with database persistence."""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.database.mongodb_manager import MongoDBManager
from src.database.document_lifecycle_models import ProcessingStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_manual_processing_integration():
    """Test that manual processing saves documents to database."""
    print("\n=== Testing Manual Processing Integration ===")

    try:
        # Check if we have any processed documents in the database
        mongodb_mgr = MongoDBManager()

        print("📋 Checking existing documents in database...")
        documents = mongodb_mgr.list_documents(limit=10)

        if documents:
            print(f"✅ Found {len(documents)} documents in database:")
            for doc in documents:
                print(f"  - ID: {doc.document_id}")
                print(f"    Filename: {doc.original_filename}")
                print(f"    Status: {doc.current_status}")
                print(f"    Created: {doc.created_at}")

                # Get chunks for this document
                chunks = mongodb_mgr.get_document_chunks(doc.document_id)
                print(f"    Chunks: {len(chunks)}")

                # Get batches for this document
                batches = mongodb_mgr.get_document_batches(doc.document_id)
                print(f"    Batches: {len(batches)}")

                # Get statistics
                stats = mongodb_mgr.get_document_statistics(doc.document_id)
                print(f"    Stats: {stats}")
                print()

            return True
        else:
            print("❌ No documents found in database")
            print("💡 This suggests that document processing is not saving to the new database structure")
            print("💡 Please run a document through the application and check again")
            return False

    except Exception as e:
        print(f"❌ Manual processing integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_database():
    """Check if documents are being saved to the legacy database."""
    print("\n=== Testing Legacy Database ===")

    try:
        from pymongo import MongoClient

        mongodb_uri = os.getenv('MONGODB_URI')
        collection_name = os.getenv('COLLECTION', 'singlepasschunking')

        client = MongoClient(mongodb_uri)
        db_name = mongodb_uri.split('/')[-1].split('?')[0]
        db = client[db_name]
        collection = db[collection_name]

        # Count documents in legacy collection
        legacy_count = collection.count_documents({})
        print(f"📊 Legacy collection '{collection_name}' has {legacy_count} documents")

        if legacy_count > 0:
            # Get a few sample documents
            sample_docs = list(collection.find().limit(3))
            print("📝 Sample legacy documents:")
            for i, doc in enumerate(sample_docs, 1):
                doc.pop('_id', None)  # Remove MongoDB ID for cleaner output
                print(f"  {i}. Keys: {list(doc.keys())}")
                if 'chunk_id' in doc:
                    print(f"     Chunk ID: {doc.get('chunk_id', 'N/A')}")
                if 'content' in doc:
                    content_preview = doc['content'][:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
                    print(f"     Content: {content_preview}")
                print()

        client.close()
        return legacy_count > 0

    except Exception as e:
        print(f"❌ Legacy database test failed: {e}")
        return False


def main():
    """Run document processing tests."""
    print("🧪 Document Processing Integration Test")
    print("=" * 50)

    # Test new document lifecycle database
    lifecycle_ok = test_manual_processing_integration()

    # Test legacy database
    legacy_ok = test_legacy_database()

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Processing Test Summary")
    print("=" * 50)
    print(f"New Lifecycle Database: {'✅ PASS' if lifecycle_ok else '❌ EMPTY'}")
    print(f"Legacy Database:        {'✅ PASS' if legacy_ok else '❌ EMPTY'}")

    if not lifecycle_ok and not legacy_ok:
        print("\n⚠️  No documents found in either database.")
        print("💡 Please process a document through the application first:")
        print("   1. Start the application: python main.py")
        print("   2. Process a document using either human-in-loop or bulk mode")
        print("   3. Run this test again")
    elif legacy_ok and not lifecycle_ok:
        print("\n🔄 Documents found in legacy database but not in new lifecycle database.")
        print("💡 The lifecycle manager integration may not be working properly.")
    elif lifecycle_ok:
        print("\n🎉 Documents found in new lifecycle database! Integration is working.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)