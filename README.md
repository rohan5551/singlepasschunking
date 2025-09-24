# PDF Processing Application

A web-based PDF processing application built with FastAPI that allows users to upload, process, and analyze PDF documents from local files, file paths, or S3 URLs.

## Features

- **Multiple Input Methods**: Upload files, provide local paths, or use S3 URLs
- **PDF Processing**: Extract metadata, convert pages to images, and analyze document structure
- **Web Interface**: Clean, responsive web UI built with Bootstrap
- **S3 Integration**: Direct support for processing PDFs stored in Amazon S3
- **Preview Generation**: Automatic preview generation for PDF pages

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /Users/rohan/Desktop/new\ projects/singlepasschunking
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (for PDF processing)

   **On macOS:**
   ```bash
   brew install poppler
   ```

   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get install poppler-utils
   ```

   **On Windows:**
   - Download Poppler from: https://poppler.freedesktop.org/
   - Add to PATH environment variable

4. **Set up environment variables** (optional, for S3 support)
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
   ```

## Usage

### Start the Web Application

```bash
python main.py
```

The application will be available at: http://localhost:8000

### Two Interface Options

#### 1. Single Document Processing (`/`)
- **Upload File**: Select a PDF file from your computer
- **Local Path**: Enter the full path to a PDF file on the server
- **S3 URL**: Enter an S3 URL in the format `s3://bucket-name/path/to/file.pdf`

#### 2. Pipeline Dashboard (`/pipeline`) - **NEW!**
- **Multi-document processing**: Upload and process multiple PDFs simultaneously
- **Real-time monitoring**: Track documents through each processing stage
- **Batch configuration**: Configure page batching and overlap settings
- **Stage-based navigation**: Click stages to see documents, click documents to see detailed output
- **Progress tracking**: Monitor processing progress with visual indicators

### Using the Pipeline Dashboard

1. **Upload Multiple Documents**: Drag and drop or browse multiple PDF files
2. **Configure Processing**: Set batch size (2-6 pages) and overlap (0-3 pages)
3. **Monitor Progress**: Watch documents move through processing stages:
   - **Upload Queue**: Documents waiting to be processed
   - **PDF Processing**: Extracting content and metadata
   - **Splitting**: Creating page batches for processing
   - **Completed**: Successfully processed documents
4. **View Details**: Click any stage to see documents, click documents to see batch details
5. **Manage Tasks**: Cancel running tasks or clear completed ones

### Test the Complete Pipeline

Run the pipeline test script:

```bash
python test_pipeline.py
```

Or test individual components:

```bash
# Test PDF processor only
python test_processor.py

# Test PDF splitter only
python test_splitter.py
```

### Test the PDFProcessor Class

Run the test script to test the core functionality:

```bash
python test_processor.py
```

### API Endpoints

- `GET /` - Main upload interface
- `POST /process-pdf` - Process PDF from file/path/S3
- `GET /api/document-info/{source_type}` - Get document information
- `GET /api/page-preview/{source_type}/{page_number}` - Get page preview

## Project Structure

```
singlepasschunking/
├── main.py                 # FastAPI application
├── test_processor.py       # Test script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pdf_document.py # PDF document models
│   └── processors/
│       ├── __init__.py
│       └── pdf_processor.py # Core PDF processing logic
├── templates/
│   ├── base.html          # Base template
│   ├── index.html         # Upload interface
│   └── result.html        # Results display
└── static/
    ├── style.css          # Custom styles
    └── script.js          # JavaScript functionality
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables for S3 support:

```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
```

## Architecture

This application implements the first stage of the Vision-Guided Chunking System:

- **PDFProcessor**: Core class for loading and processing PDF documents
- **PDFDocument/PDFPage**: Data models representing processed documents
- **Web Interface**: FastAPI-based web application for user interaction
- **S3 Integration**: Built-in support for Amazon S3 storage

## Development

### Adding New Features

1. **Core Logic**: Add new processors in `src/processors/`
2. **Data Models**: Define new models in `src/models/`
3. **Web Interface**: Update templates and routes in `main.py`
4. **Static Assets**: Add CSS/JS files in `static/`

### Running in Development Mode

```bash
# With auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Next Steps

This is the foundation for the Vision-Guided Chunking System. Future stages will include:

1. **PDFSplitter**: Split documents into configurable page batches
2. **LMMProcessor**: Integration with Large Multimodal Models
3. **ChunkExtractor**: Parse and extract chunks from LMM responses
4. **ContextManager**: Maintain context across batch boundaries
5. **Vector Database Integration**: Store processed chunks for RAG

## Troubleshooting

### Common Issues

1. **Poppler not found**: Install poppler-utils system dependency
2. **AWS credentials error**: Check .env file and AWS permissions
3. **File not found**: Verify file paths and permissions
4. **Large file upload**: Check file size limits (default: 50MB)

### Logs

Check the application logs for detailed error information:
```bash
python main.py  # Logs will appear in terminal
```

## License

MIT License - see LICENSE file for details