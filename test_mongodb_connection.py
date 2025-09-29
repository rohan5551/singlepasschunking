#!/usr/bin/env python3
"""
Test script to verify MongoDB connection and basic operations.
"""

import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mongodb_connection():
    """Test MongoDB connection and basic operations."""

    mongodb_uri = os.getenv('MONGODB_URI')
    collection_name = os.getenv('COLLECTION', 'singlepasschunking')

    logger.info(f"Testing MongoDB connection...")
    logger.info(f"URI: {mongodb_uri[:30]}..." if mongodb_uri else "No URI found")
    logger.info(f"Collection: {collection_name}")

    if not mongodb_uri:
        logger.error("MONGODB_URI not found in environment variables")
        return False

    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError

        # Initialize client
        client = MongoClient(mongodb_uri)

        # Extract database name
        if '/' in mongodb_uri and '?' in mongodb_uri:
            db_part = mongodb_uri.split('/')[-1]
            db_name = db_part.split('?')[0]
        elif '/' in mongodb_uri:
            db_name = mongodb_uri.split('/')[-1]
        else:
            db_name = "one-sea"

        logger.info(f"Database name: {db_name}")

        # Get database and collection
        db = client[db_name]
        collection = db[collection_name]

        # Test connection
        client.admin.command('ping')
        logger.info("✓ MongoDB connection successful")

        # Test database access
        db_info = db.command("dbStats")
        logger.info(f"✓ Database access successful - Collections: {db_info.get('collections', 0)}")

        # Test collection access
        count = collection.count_documents({})
        logger.info(f"✓ Collection access successful - Documents: {count}")

        # Test write operation
        test_doc = {
            "test_id": "connection_test",
            "timestamp": datetime.utcnow(),
            "message": "MongoDB connection test"
        }

        result = collection.insert_one(test_doc)
        logger.info(f"✓ Write test successful - Inserted ID: {result.inserted_id}")

        # Clean up test document
        collection.delete_one({"test_id": "connection_test"})
        logger.info("✓ Cleanup successful")

        # List all collections
        collections = db.list_collection_names()
        logger.info(f"Available collections: {collections}")

        client.close()
        return True

    except PyMongoError as e:
        logger.error(f"✗ MongoDB error: {e}")
        return False
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Make sure pymongo is installed: pip install pymongo")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False

def test_chunk_manager():
    """Test the ChunkManager class."""
    logger.info("\n" + "="*50)
    logger.info("Testing ChunkManager...")

    try:
        from src.processors.chunk_manager import ChunkManager

        # Initialize ChunkManager
        chunk_manager = ChunkManager()
        logger.info("✓ ChunkManager initialized successfully")

        # Test basic connection info
        logger.info(f"Collection name: {chunk_manager.collection_name}")
        logger.info(f"Database name: {chunk_manager.db.name}")

        return True

    except Exception as e:
        logger.error(f"✗ ChunkManager test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("MongoDB Connection Test")
    logger.info("=" * 50)

    # Test basic connection
    connection_ok = test_mongodb_connection()

    if connection_ok:
        # Test ChunkManager if basic connection works
        chunk_manager_ok = test_chunk_manager()

        if chunk_manager_ok:
            logger.info("\n✅ All tests passed! MongoDB is ready for use.")
        else:
            logger.error("\n❌ ChunkManager test failed.")
    else:
        logger.error("\n❌ MongoDB connection failed.")
        logger.error("Please check your MONGODB_URI and network connection.")