// JavaScript for PDF Processing Application

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // Tab switching logic to clear other inputs
    const tabButtons = document.querySelectorAll('[data-bs-toggle="pill"]');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Clear all inputs when switching tabs
            document.getElementById('file').value = '';
            document.getElementById('file_path').value = '';
            document.getElementById('s3_url').value = '';
        });
    });

    // Form submission handling
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const activeTab = document.querySelector('.nav-link.active');
            let hasInput = false;

            // Check which tab is active and validate accordingly
            if (activeTab.id === 'pills-upload-tab') {
                const fileInput = document.getElementById('file');
                if (fileInput.files.length > 0) {
                    hasInput = true;
                }
            } else if (activeTab.id === 'pills-local-tab') {
                const pathInput = document.getElementById('file_path');
                if (pathInput.value.trim()) {
                    hasInput = true;
                }
            } else if (activeTab.id === 'pills-s3-tab') {
                const s3Input = document.getElementById('s3_url');
                if (s3Input.value.trim()) {
                    hasInput = true;
                }
            }

            if (!hasInput) {
                e.preventDefault();
                alert('Please provide a file, file path, or S3 URL before submitting.');
                return;
            }

            // Show loading indicator
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            loadingIndicator.style.display = 'block';
        });
    }

    // File input validation
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Check file type
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    alert('Please select a PDF file.');
                    this.value = '';
                    return;
                }

                // Check file size (50MB limit)
                const maxSize = 50 * 1024 * 1024; // 50MB in bytes
                if (file.size > maxSize) {
                    alert('File size must be less than 50MB.');
                    this.value = '';
                    return;
                }
            }
        });
    }

    // S3 URL validation
    const s3Input = document.getElementById('s3_url');
    if (s3Input) {
        s3Input.addEventListener('blur', function() {
            const url = this.value.trim();
            if (url && !url.startsWith('s3://')) {
                alert('S3 URL must start with "s3://"');
                this.focus();
            }
        });
    }

    // File path validation
    const pathInput = document.getElementById('file_path');
    if (pathInput) {
        pathInput.addEventListener('blur', function() {
            const path = this.value.trim();
            if (path && !path.toLowerCase().endsWith('.pdf')) {
                alert('File path must point to a PDF file (.pdf extension).');
                this.focus();
            }
        });
    }
});

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);

    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Image modal functionality
let currentPageData = [];
let currentPageIndex = 0;

function openImageModal(pageNumber, imageData) {
    // Store page data if not already stored
    if (currentPageData.length === 0) {
        // Collect all page images from the DOM
        const pageImages = document.querySelectorAll('[onclick*="openImageModal"]');
        pageImages.forEach((img, index) => {
            const onclickAttr = img.getAttribute('onclick');
            const matches = onclickAttr.match(/openImageModal\((\d+), '([^']+)'\)/);
            if (matches) {
                currentPageData.push({
                    pageNumber: parseInt(matches[1]),
                    imageData: matches[2]
                });
            }
        });
    }

    // Find the index of the clicked page
    currentPageIndex = currentPageData.findIndex(p => p.pageNumber === pageNumber);

    if (currentPageIndex === -1) {
        currentPageIndex = 0;
    }

    showModalPage(currentPageIndex);

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
}

function showModalPage(index) {
    if (index < 0 || index >= currentPageData.length) return;

    const pageData = currentPageData[index];
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalPageTitle');
    const pageCounter = document.getElementById('pageCounter');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');

    // Update modal content
    modalImage.src = `data:image/png;base64,${pageData.imageData}`;
    modalTitle.textContent = `Page ${pageData.pageNumber}`;
    pageCounter.textContent = `Page ${index + 1} of ${currentPageData.length}`;

    // Update navigation buttons
    prevBtn.disabled = (index === 0);
    nextBtn.disabled = (index === currentPageData.length - 1);

    currentPageIndex = index;
}

function navigatePage(direction) {
    const newIndex = currentPageIndex + direction;
    if (newIndex >= 0 && newIndex < currentPageData.length) {
        showModalPage(newIndex);
    }
}

// Keyboard navigation for modal
document.addEventListener('keydown', function(e) {
    const modal = document.getElementById('imageModal');
    if (modal.classList.contains('show')) {
        if (e.key === 'ArrowLeft') {
            navigatePage(-1);
        } else if (e.key === 'ArrowRight') {
            navigatePage(1);
        } else if (e.key === 'Escape') {
            bootstrap.Modal.getInstance(modal).hide();
        }
    }
});

// Reset page data when modal is closed
document.getElementById('imageModal').addEventListener('hidden.bs.modal', function () {
    currentPageData = [];
    currentPageIndex = 0;
});