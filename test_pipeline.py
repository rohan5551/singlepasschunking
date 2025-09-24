#!/usr/bin/env python3
"""
Test script for the complete PDF processing pipeline
"""

import os
import sys
import requests
import time

# Test the new pipeline functionality
def test_pipeline_api():
    """Test the pipeline API endpoints"""
    base_url = "http://localhost:8000"

    print("🧪 Testing PDF Processing Pipeline")
    print("=" * 50)

    try:
        # Test 1: Check if server is running
        print("\n1️⃣ Checking server status...")
        response = requests.get(f"{base_url}/pipeline")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server not accessible")
            return

        # Test 2: Get initial task list
        print("\n2️⃣ Getting initial task list...")
        response = requests.get(f"{base_url}/api/pipeline/tasks")
        if response.status_code == 200:
            tasks = response.json()
            print(f"✅ Found {len(tasks)} existing tasks")
        else:
            print("❌ Could not fetch tasks")
            return

        # Test 3: Check pipeline status
        print("\n3️⃣ Checking pipeline status...")
        response = requests.get(f"{base_url}/api/pipeline/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Pipeline status: {status}")
        else:
            print("❌ Could not fetch pipeline status")

        # Test 4: Test file upload (if PDF file available)
        test_pdf_path = input("\n4️⃣ Enter path to a test PDF file (or press Enter to skip): ").strip()

        if test_pdf_path and os.path.exists(test_pdf_path):
            print(f"📤 Uploading {test_pdf_path}...")

            with open(test_pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(test_pdf_path), f, 'application/pdf')}
                data = {
                    'batch_size': 4,
                    'overlap_pages': 0
                }

                response = requests.post(f"{base_url}/api/pipeline/submit", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    task_id = result['task_id']
                    print(f"✅ File submitted successfully! Task ID: {task_id}")

                    # Monitor the task
                    print("\n⏱️ Monitoring task progress...")
                    for i in range(30):  # Wait up to 30 seconds
                        time.sleep(1)
                        task_response = requests.get(f"{base_url}/api/pipeline/tasks/{task_id}")

                        if task_response.status_code == 200:
                            task = task_response.json()
                            status = task.get('status', 'unknown')
                            progress = task.get('progress', 0)

                            print(f"   Status: {status} | Progress: {progress:.0f}%")

                            if status in ['completed', 'error']:
                                break
                        else:
                            print("   ❌ Could not fetch task status")
                            break

                    # Final status check
                    final_response = requests.get(f"{base_url}/api/pipeline/tasks/{task_id}")
                    if final_response.status_code == 200:
                        final_task = final_response.json()
                        print(f"\n📊 Final Result:")
                        print(f"   Status: {final_task.get('status', 'unknown')}")
                        print(f"   Progress: {final_task.get('progress', 0):.0f}%")
                        print(f"   Total Batches: {final_task.get('total_batches', 0)}")
                        print(f"   Completed Batches: {final_task.get('completed_batches', 0)}")

                        if final_task.get('error_message'):
                            print(f"   Error: {final_task['error_message']}")

                        # Show batch details if available
                        if final_task.get('batches'):
                            print(f"\n📄 Batch Details:")
                            for batch in final_task['batches']:
                                print(f"   Batch {batch['batch_number']}: Pages {batch['start_page']}-{batch['end_page']} ({batch['page_count']} pages) - {batch['status']}")

                else:
                    print(f"❌ Upload failed: {response.text}")
        else:
            print("⏭️ Skipping file upload test")

        print(f"\n🌐 Open your browser to view the pipeline dashboard:")
        print(f"   {base_url}/pipeline")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running with: python main.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

def show_usage():
    """Show usage instructions"""
    print("📋 Pipeline Test Instructions:")
    print("=" * 40)
    print("1. Start the server: python main.py")
    print("2. Run this test: python test_pipeline.py")
    print("3. Open browser: http://localhost:8000/pipeline")
    print("")
    print("🎯 What to test:")
    print("• Upload multiple PDF files")
    print("• Click on pipeline stages")
    print("• Select documents to view output")
    print("• Monitor real-time progress")
    print("• Test batch configuration")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
    else:
        test_pipeline_api()