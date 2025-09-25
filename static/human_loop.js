document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('humanLoopContainer');
    if (!container) {
        return;
    }

    const sessionId = container.getAttribute('data-session-id');
    const steps = Array.from(container.querySelectorAll('.manual-step'));
    const stepIndicators = Array.from(container.querySelectorAll('.manual-stepper .step-item'));

    const proceedBtn = document.getElementById('proceedToBatchingBtn');
    const backBtn = document.getElementById('backToReviewBtn');
    const finalizeBtn = document.getElementById('finalizeBatchesBtn');
    const addBatchBtn = document.getElementById('addBatchBtn');
    const clearSelectionBtn = document.getElementById('clearSelectionBtn');
    const batchNameInput = document.getElementById('batchNameInput');
    const selectedPagesDisplay = document.getElementById('selectedPagesDisplay');
    const selectedCounter = document.getElementById('selectedPagesCounter');
    const batchListContainer = document.getElementById('batchList');
    const batchSummaryContainer = document.getElementById('batchSummaryContainer');
    const instructionsInput = document.getElementById('llmInstructionsInput');
    const defaultInstructionsBtn = document.getElementById('useDefaultInstructionsBtn');
    const startProcessingBtn = document.getElementById('startProcessingBtn');
    const backToBatchingBtn = document.getElementById('backToBatchingBtn');
    const processingStatusContainer = document.getElementById('processingStatusContainer');
    const processingProgressBar = document.getElementById('processingProgressBar');
    const processingStatusMessage = document.getElementById('processingStatusMessage');

    const checkboxes = Array.from(container.querySelectorAll('.page-checkbox'));

    let currentStep = 1;
    let batches = [];
    let manualTask = null;
    let manualBatches = [];
    let manualDefaultPrompt = '';
    let processingPoller = null;
    let instructionsDirty = false;

    function escapeHtml(value) {
        if (typeof value !== 'string') {
            value = value?.toString() ?? '';
        }
        return value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function setStep(stepNumber) {
        currentStep = stepNumber;
        steps.forEach(step => {
            const stepValue = parseInt(step.getAttribute('data-step'), 10);
            step.style.display = stepValue === stepNumber ? '' : 'none';
        });

        stepIndicators.forEach(indicator => {
            const indicatorStep = parseInt(indicator.getAttribute('data-step'), 10);
            if (indicatorStep <= stepNumber) {
                indicator.classList.add('active');
            } else {
                indicator.classList.remove('active');
            }
        });
    }

    function getSelectedPages() {
        return checkboxes
            .filter(cb => cb.checked)
            .map(cb => parseInt(cb.getAttribute('data-page-number'), 10))
            .sort((a, b) => a - b);
    }

    function updateSelectionView() {
        const selectedPages = getSelectedPages();
        if (selectedPages.length === 0) {
            selectedPagesDisplay.textContent = 'No pages selected yet.';
            selectedCounter.textContent = '0 pages selected';
        } else {
            selectedPagesDisplay.textContent = selectedPages.join(', ');
            selectedCounter.textContent = `${selectedPages.length} page${selectedPages.length > 1 ? 's' : ''} selected`;
        }

        checkboxes.forEach(cb => {
            const card = cb.closest('.manual-page-card');
            if (card) {
                card.classList.toggle('selected', cb.checked);
            }
        });

        if (!batchNameInput.value.trim()) {
            batchNameInput.value = `Batch ${batches.length + 1}`;
        }
    }

    function resetSelection() {
        checkboxes.forEach(cb => {
            cb.checked = false;
        });
        updateSelectionView();
    }

    function renderBatches() {
        batchListContainer.innerHTML = '';
        if (batches.length === 0) {
            const placeholder = document.createElement('div');
            placeholder.className = 'text-muted small';
            placeholder.textContent = 'No batches yet. Select pages and click "Add Batch".';
            batchListContainer.appendChild(placeholder);
        } else {
            batches.forEach((batch, index) => {
                const item = document.createElement('div');
                item.className = 'batch-item';

                const header = document.createElement('div');
                header.className = 'd-flex justify-content-between align-items-center';
                header.innerHTML = `
                    <div>
                        <strong>${batch.name}</strong>
                        <span class="badge bg-light text-dark ms-2">${batch.pages.length} page${batch.pages.length > 1 ? 's' : ''}</span>
                    </div>
                    <button class="btn btn-link p-0" data-index="${index}">
                        <i class="fas fa-trash-alt me-1"></i>Remove
                    </button>
                `;

                const body = document.createElement('div');
                body.className = 'mt-2';
                body.innerHTML = `<small class="text-muted">Pages:</small> ${batch.pages.join(', ')}`;

                item.appendChild(header);
                item.appendChild(body);
                batchListContainer.appendChild(item);
            });
        }

        finalizeBtn.disabled = batches.length === 0;
    }

    function showToast(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        batchListContainer.prepend(alert);
        setTimeout(() => alert.remove(), 3000);
    }

    function stopProcessingPoller() {
        if (processingPoller) {
            clearInterval(processingPoller);
            processingPoller = null;
        }
    }

    function startProcessingPoller() {
        stopProcessingPoller();
        fetchProcessingStatus({ silent: true });
        processingPoller = setInterval(() => fetchProcessingStatus({ silent: true }), 2000);
    }

    function getStatusDetails(status) {
        switch ((status || '').toLowerCase()) {
            case 'completed':
                return { label: 'Completed', badge: 'bg-success' };
            case 'processing':
            case 'lmm_processing':
                return { label: 'Processing', badge: 'bg-info text-dark' };
            case 'splitting':
                return { label: 'Splitting', badge: 'bg-secondary' };
            case 'upload':
            case 'pdf_processing':
                return { label: 'Preparing', badge: 'bg-secondary' };
            case 'chunking':
                return { label: 'Chunking', badge: 'bg-info text-dark' };
            case 'error':
                return { label: 'Error', badge: 'bg-danger' };
            default:
                return { label: 'Pending', badge: 'bg-secondary' };
        }
    }

    function formatStatusBadge(status) {
        const details = getStatusDetails(status);
        return `<span class="badge ${details.badge}">${details.label}</span>`;
    }

    function renderProcessingStatus(batchesData, taskData) {
        if (!processingStatusContainer) {
            return;
        }

        const items = Array.isArray(batchesData) ? batchesData : [];
        processingStatusContainer.innerHTML = '';

        if (items.length === 0) {
            const placeholder = document.createElement('p');
            placeholder.className = 'text-muted mb-0';
            placeholder.textContent = 'No batches available. Create batches to begin.';
            processingStatusContainer.appendChild(placeholder);
        } else {
            items.forEach((batch) => {
                const details = getStatusDetails(batch.status);
                const item = document.createElement('div');
                item.className = 'processing-status-item border rounded p-3 mb-2';
                const pages = Array.isArray(batch.page_numbers) ? batch.page_numbers.join(', ') : '—';
                item.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${batch.name}</strong>
                            <div class="text-muted small">Pages: ${pages}</div>
                        </div>
                        <span class="badge ${details.badge}">${details.label}</span>
                    </div>
                `;
                if (batch.error_message) {
                    const errorAlert = document.createElement('div');
                    errorAlert.className = 'alert alert-danger mt-3 mb-0';
                    errorAlert.textContent = batch.error_message;
                    item.appendChild(errorAlert);
                }
                if (Array.isArray(batch.warnings) && batch.warnings.length > 0) {
                    const warningAlert = document.createElement('div');
                    warningAlert.className = 'alert alert-warning mt-3 mb-0';
                    warningAlert.innerHTML = `
                        <strong>Heads up:</strong><br>
                        ${batch.warnings.map(msg => `<span class="d-block">${escapeHtml(msg)}</span>`).join('')}
                    `;
                    item.appendChild(warningAlert);
                }
                processingStatusContainer.appendChild(item);
            });
        }

        if (processingProgressBar) {
            const progressValue = Math.round(taskData?.progress ?? 0);
            processingProgressBar.style.width = `${progressValue}%`;
            processingProgressBar.textContent = `${progressValue}%`;
            processingProgressBar.setAttribute('aria-valuenow', `${progressValue}`);
        }

        if (processingStatusMessage) {
            const status = taskData?.status;
            if (status === 'lmm_processing') {
                const completed = taskData?.completed_batches ?? items.filter(item => item.status === 'completed').length;
                processingStatusMessage.textContent = `Processing batch ${Math.min(completed + 1, items.length)} of ${items.length}...`;
            } else if (status === 'completed') {
                processingStatusMessage.textContent = 'All batches processed successfully.';
            } else if (status === 'error') {
                processingStatusMessage.textContent = taskData?.error_message || 'Processing encountered an error.';
            } else {
                processingStatusMessage.textContent = 'Batches are waiting to be processed.';
            }
        }
    }

    function updateManualSessionState(data, { initialisePrompt = false } = {}) {
        manualTask = data?.task || null;
        manualBatches = Array.isArray(data?.batches) ? data.batches : [];

        if (data?.default_prompt) {
            manualDefaultPrompt = data.default_prompt;
        }

        if (instructionsInput) {
            const promptTemplate = data?.processing_prompt || manualDefaultPrompt || '';
            if (initialisePrompt || !instructionsDirty) {
                instructionsInput.value = promptTemplate;
                instructionsDirty = false;
            }
        }

        renderProcessingStatus(manualBatches, manualTask);

        if (startProcessingBtn) {
            if (manualTask?.status === 'lmm_processing') {
                startProcessingBtn.disabled = true;
                startProcessingBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            } else {
                startProcessingBtn.disabled = false;
                startProcessingBtn.innerHTML = '<i class="fas fa-rocket me-2"></i>Start LLM Processing';
            }
        }
    }

    async function fetchProcessingStatus({ silent = false } = {}) {
        if (!sessionId) {
            return;
        }

        try {
            const response = await fetch(`/api/manual/${sessionId}/status`);
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || 'Failed to fetch processing status');
            }

            const data = await response.json();
            updateManualSessionState(data);

            if (manualTask?.status === 'completed') {
                stopProcessingPoller();
                renderBatchSummary(data);
                setStep(4);
            } else if (manualTask?.status === 'error') {
                stopProcessingPoller();
                if (!silent) {
                    showToast(manualTask.error_message || 'LLM processing failed.', 'danger');
                }
            }
        } catch (error) {
            if (silent) {
                console.error(error);
            } else {
                showToast(error.message, 'danger');
            }
        }
    }

    proceedBtn?.addEventListener('click', () => setStep(2));
    backBtn?.addEventListener('click', () => setStep(1));
    backToBatchingBtn?.addEventListener('click', () => {
        stopProcessingPoller();
        setStep(2);
    });

    instructionsInput?.addEventListener('input', () => {
        instructionsDirty = true;
    });

    defaultInstructionsBtn?.addEventListener('click', (event) => {
        event.preventDefault();
        if (!instructionsInput) {
            return;
        }
        instructionsInput.value = manualDefaultPrompt || '';
        instructionsDirty = false;
    });

    checkboxes.forEach(cb => {
        cb.addEventListener('change', updateSelectionView);
    });

    clearSelectionBtn?.addEventListener('click', (event) => {
        event.preventDefault();
        resetSelection();
    });

    batchListContainer.addEventListener('click', (event) => {
        const target = event.target.closest('button[data-index]');
        if (!target) {
            return;
        }
        event.preventDefault();
        const index = parseInt(target.getAttribute('data-index'), 10);
        batches.splice(index, 1);
        renderBatches();
        updateSelectionView();
    });

    addBatchBtn?.addEventListener('click', (event) => {
        event.preventDefault();
        const selectedPages = getSelectedPages();

        if (selectedPages.length === 0) {
            showToast('Select at least one page before adding a batch.', 'warning');
            return;
        }

        const name = batchNameInput.value.trim() || `Batch ${batches.length + 1}`;

        batches.push({
            name,
            pages: selectedPages
        });

        batchNameInput.value = '';
        resetSelection();
        renderBatches();
    });

    finalizeBtn?.addEventListener('click', async (event) => {
        event.preventDefault();
        if (!sessionId) {
            showToast('Session has expired. Please restart the process.', 'danger');
            return;
        }

        finalizeBtn.disabled = true;
        finalizeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Finalising...';

        try {
            const response = await fetch(`/api/manual/${sessionId}/create-batches`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ batches })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to create batches');
            }

            const data = await response.json();
            updateManualSessionState(data, { initialisePrompt: true });
            setStep(3);
        } catch (error) {
            showToast(error.message, 'danger');
        } finally {
            finalizeBtn.innerHTML = '<i class="fas fa-check me-2"></i>Finalise Batches';
        }
    });

    startProcessingBtn?.addEventListener('click', async (event) => {
        event.preventDefault();
        if (!sessionId || !instructionsInput) {
            showToast('Session is not available. Please restart the process.', 'danger');
            return;
        }

        startProcessingBtn.disabled = true;
        startProcessingBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';

        try {
            const response = await fetch(`/api/manual/${sessionId}/process-batches`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ instructions: instructionsInput.value })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || 'Failed to start LLM processing');
            }

            instructionsDirty = false;
            startProcessingBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            startProcessingPoller();
        } catch (error) {
            showToast(error.message, 'danger');
            startProcessingBtn.disabled = false;
            startProcessingBtn.innerHTML = '<i class="fas fa-rocket me-2"></i>Start LLM Processing';
        }
    });

    function renderBatchSummary(data) {
        batchSummaryContainer.innerHTML = '';

        const meta = document.createElement('div');
        meta.className = 'mb-3';
        const batchesList = Array.isArray(data.batches) ? data.batches : [];
        const totalBatches = data.total_batches ?? batchesList.length;
        const completedBatches = data.task?.completed_batches ?? batchesList.filter(batch => batch.status === 'completed').length;
        const summaryStatus = data.task?.status || (completedBatches === totalBatches && totalBatches > 0 ? 'completed' : 'pending');
        const processingTime = data.task?.processing_time;
        const formattedTime = typeof processingTime === 'number' ? `${processingTime.toFixed(1)}s` : '—';

        meta.innerHTML = `
            <div class="row text-center g-3">
                <div class="col-md-3">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${totalBatches}</h5>
                        <small class="text-muted">Total Batches</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${completedBatches}</h5>
                        <small class="text-muted">Completed Batches</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${data.total_pages ?? '—'}</h5>
                        <small class="text-muted">Total Pages</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${formattedTime}</h5>
                        <small class="text-muted">Processing Time</small>
                    </div>
                </div>
            </div>
            <div class="text-center mt-3">
                ${formatStatusBadge(summaryStatus)}
                <small class="d-block text-muted mt-2">Task Status</small>
            </div>
        `;
        batchSummaryContainer.appendChild(meta);

        const table = document.createElement('table');
        table.className = 'table table-striped batch-summary-table mt-4';
        table.innerHTML = `
            <thead class="table-light">
                <tr>
                    <th>Batch</th>
                    <th>Status</th>
                    <th>Pages</th>
                    <th>Processed</th>
                </tr>
            </thead>
            <tbody>
                ${batchesList.map(batch => `
                    <tr>
                        <td>
                            ${escapeHtml(batch.name)}
                            ${Array.isArray(batch.warnings) && batch.warnings.length > 0 ? '<span class="badge bg-warning text-dark ms-2">Warning</span>' : ''}
                        </td>
                        <td>${formatStatusBadge(batch.status)}</td>
                        <td>
                            <span class="badge bg-primary">${batch.page_count} page${batch.page_count > 1 ? 's' : ''}</span><br>
                            <small class="text-muted">${Array.isArray(batch.page_numbers) ? batch.page_numbers.join(', ') : '—'}</small>
                        </td>
                        <td>${batch.processed_at ? new Date(batch.processed_at).toLocaleString() : '—'}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;

        batchSummaryContainer.appendChild(table);
    }

    // Initialise view
    renderBatches();
    updateSelectionView();
    setStep(1);
});
