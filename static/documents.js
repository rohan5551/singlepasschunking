// JavaScript for Documents History Page

document.addEventListener('DOMContentLoaded', function() {
    loadDocuments();
});

async function loadDocuments() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorState = document.getElementById('errorState');
    const emptyState = document.getElementById('emptyState');
    const documentsContainer = document.getElementById('documentsContainer');
    const summaryStats = document.getElementById('summaryStats');

    // Show loading state
    loadingIndicator.style.display = 'block';
    errorState.style.display = 'none';
    emptyState.style.display = 'none';
    documentsContainer.style.display = 'none';
    summaryStats.style.display = 'none';

    try {
        const response = await fetch('/api/documents');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const documents = data.documents || [];

        loadingIndicator.style.display = 'none';

        if (documents.length === 0) {
            emptyState.style.display = 'block';
            return;
        }

        // Show documents table and stats
        documentsContainer.style.display = 'block';
        summaryStats.style.display = 'block';

        // Populate table
        populateDocumentsTable(documents);

        // Update summary stats
        updateSummaryStats(documents);

    } catch (error) {
        console.error('Error loading documents:', error);
        loadingIndicator.style.display = 'none';
        errorState.style.display = 'block';
        document.getElementById('errorMessage').textContent = error.message;
    }
}

function populateDocumentsTable(documents) {
    const tableBody = document.getElementById('documentsTableBody');
    tableBody.innerHTML = '';

    documents.forEach(doc => {
        const row = document.createElement('tr');
        row.style.cursor = 'pointer';
        row.onclick = () => showDocumentDetails(doc.document_id);

        const statusBadge = getStatusBadge(doc.status);
        const modeBadge = getModeBadge(doc.processing_mode);
        const fileSize = formatFileSize(doc.file_size_bytes);
        const createdDate = formatDate(doc.created_at);

        row.innerHTML = `
            <td>
                <div class="d-flex align-items-center">
                    <i class="fas fa-file-pdf text-danger me-2"></i>
                    <div>
                        <div class="fw-bold">${escapeHtml(doc.original_filename)}</div>
                        <small class="text-muted">${doc.document_id}</small>
                    </div>
                </div>
            </td>
            <td>${statusBadge}</td>
            <td>${modeBadge}</td>
            <td>${doc.total_pages || 'N/A'}</td>
            <td>${fileSize}</td>
            <td>${createdDate}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="event.stopPropagation(); showDocumentDetails('${doc.document_id}')">
                    <i class="fas fa-eye me-1"></i>
                    View Details
                </button>
            </td>
        `;

        tableBody.appendChild(row);
    });
}

function updateSummaryStats(documents) {
    const stats = {
        total: documents.length,
        completed: documents.filter(d => d.status === 'completed').length,
        processing: documents.filter(d => d.status === 'processing').length,
        error: documents.filter(d => d.status === 'error').length
    };

    document.getElementById('totalDocuments').textContent = stats.total;
    document.getElementById('completedDocuments').textContent = stats.completed;
    document.getElementById('processingDocuments').textContent = stats.processing;
    document.getElementById('errorDocuments').textContent = stats.error;
}

async function showDocumentDetails(documentId) {
    try {
        const response = await fetch(`/api/documents/${documentId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        populateDocumentDetailModal(data);

        const modal = new bootstrap.Modal(document.getElementById('documentDetailModal'));
        modal.show();

    } catch (error) {
        console.error('Error loading document details:', error);
        alert('Failed to load document details: ' + error.message);
    }
}

function populateDocumentDetailModal(data) {
    const doc = data.document;

    // Basic document info
    document.getElementById('docFilename').textContent = doc.original_filename;
    document.getElementById('docStatus').innerHTML = getStatusBadge(doc.status);
    document.getElementById('docMode').innerHTML = getModeBadge(doc.processing_mode);
    document.getElementById('docPages').textContent = doc.total_pages || 'N/A';
    document.getElementById('docSize').textContent = formatFileSize(doc.file_size_bytes);
    document.getElementById('docCreated').textContent = formatDate(doc.created_at);

    // Processing stages
    const stagesList = document.getElementById('stagesList');
    stagesList.innerHTML = '';
    if (data.stages && data.stages.length > 0) {
        data.stages.forEach(stage => {
            const stageCard = document.createElement('div');
            stageCard.className = 'card mb-2';
            stageCard.innerHTML = `
                <div class="card-body py-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${escapeHtml(stage.stage_name)}</strong>
                            ${getStatusBadge(stage.status)}
                        </div>
                        <div class="text-muted">
                            ${stage.started_at ? formatDate(stage.started_at) : 'Not started'}
                        </div>
                    </div>
                    ${stage.error_message ? `<div class="text-danger mt-1"><small>${escapeHtml(stage.error_message)}</small></div>` : ''}
                </div>
            `;
            stagesList.appendChild(stageCard);
        });
    } else {
        stagesList.innerHTML = '<p class="text-muted">No processing stages recorded</p>';
    }

    // Batches
    const batchesList = document.getElementById('batchesList');
    batchesList.innerHTML = '';
    if (data.batches && data.batches.length > 0) {
        data.batches.forEach(batch => {
            const batchCard = document.createElement('div');
            batchCard.className = 'card mb-2';
            batchCard.innerHTML = `
                <div class="card-body py-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>Batch ${batch.batch_number}</strong>
                            <span class="text-muted">(Pages ${batch.start_page}-${batch.end_page})</span>
                            ${getStatusBadge(batch.status)}
                        </div>
                        <div class="text-muted">
                            ${batch.chunk_count || 0} chunks
                        </div>
                    </div>
                </div>
            `;
            batchesList.appendChild(batchCard);
        });
    } else {
        batchesList.innerHTML = '<p class="text-muted">No batches created</p>';
    }

    // Images
    const imagesList = document.getElementById('imagesList');
    imagesList.innerHTML = '';
    if (data.images && data.images.length > 0) {
        const imagesGrid = document.createElement('div');
        imagesGrid.className = 'row';
        data.images.forEach(image => {
            const imageCol = document.createElement('div');
            imageCol.className = 'col-md-3 mb-2';
            imageCol.innerHTML = `
                <div class="card">
                    <div class="card-body text-center py-2">
                        <i class="fas fa-image fa-2x text-info mb-2"></i>
                        <div><small>Page ${image.page_number}</small></div>
                        ${image.s3_image_url ? '<div class="text-success"><small><i class="fas fa-check"></i> Stored</small></div>' : ''}
                    </div>
                </div>
            `;
            imagesGrid.appendChild(imageCol);
        });
        imagesList.appendChild(imagesGrid);
    } else {
        imagesList.innerHTML = '<p class="text-muted">No images stored</p>';
    }

    // Chunks
    const chunksList = document.getElementById('chunksList');
    chunksList.innerHTML = '';
    if (data.chunks && data.chunks.length > 0) {
        data.chunks.forEach(chunk => {
            const chunkCard = document.createElement('div');
            chunkCard.className = 'card mb-2';
            chunkCard.innerHTML = `
                <div class="card-body py-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>Chunk ${chunk.chunk_number}</strong>
                            <span class="text-muted">(${chunk.full_content_length} chars)</span>
                            ${chunk.page_numbers.length > 0 ? `<span class="badge bg-secondary">Pages: ${chunk.page_numbers.join(', ')}</span>` : ''}
                        </div>
                        <button class="btn btn-sm btn-outline-primary" onclick="showChunkContent('${doc.document_id}', '${chunk.chunk_id}')">
                            <i class="fas fa-eye me-1"></i>
                            View Content
                        </button>
                    </div>
                    <div class="mt-2">
                        <small class="text-muted">${escapeHtml(chunk.content)}</small>
                    </div>
                </div>
            `;
            chunksList.appendChild(chunkCard);
        });
    } else {
        chunksList.innerHTML = '<p class="text-muted">No chunks generated</p>';
    }
}

async function showChunkContent(documentId, chunkId) {
    try {
        const response = await fetch(`/api/documents/${documentId}/chunks/${chunkId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const chunk = await response.json();
        document.getElementById('chunkContent').textContent = chunk.content || 'No content available';

        const modal = new bootstrap.Modal(document.getElementById('chunkContentModal'));
        modal.show();

    } catch (error) {
        console.error('Error loading chunk content:', error);
        alert('Failed to load chunk content: ' + error.message);
    }
}

function refreshDocuments() {
    loadDocuments();
}

function getStatusBadge(status) {
    const statusClasses = {
        'completed': 'bg-success',
        'processing': 'bg-warning',
        'error': 'bg-danger',
        'uploaded': 'bg-info',
        'cancelled': 'bg-secondary',
        'pending': 'bg-secondary'
    };

    const className = statusClasses[status] || 'bg-secondary';
    return `<span class="badge ${className}">${status}</span>`;
}

function getModeBadge(mode) {
    const modeClasses = {
        'auto': 'bg-primary',
        'human_loop': 'bg-warning text-dark',
        'bulk': 'bg-info'
    };

    const className = modeClasses[mode] || 'bg-secondary';
    const displayName = mode === 'human_loop' ? 'Human Loop' : (mode || 'unknown').charAt(0).toUpperCase() + (mode || 'unknown').slice(1);
    return `<span class="badge ${className}">${displayName}</span>`;
}

function formatFileSize(bytes) {
    if (!bytes) return 'N/A';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}