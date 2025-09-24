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

    const checkboxes = Array.from(container.querySelectorAll('.page-checkbox'));

    let currentStep = 1;
    let batches = [];

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

    proceedBtn?.addEventListener('click', () => setStep(2));
    backBtn?.addEventListener('click', () => setStep(1));

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
            renderBatchSummary(data);
            setStep(3);
        } catch (error) {
            showToast(error.message, 'danger');
        } finally {
            finalizeBtn.innerHTML = '<i class="fas fa-check me-2"></i>Finalise Batches';
        }
    });

    function renderBatchSummary(data) {
        batchSummaryContainer.innerHTML = '';

        const meta = document.createElement('div');
        meta.className = 'mb-3';
        meta.innerHTML = `
            <div class="row text-center">
                <div class="col-md-4">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${data.total_batches}</h5>
                        <small class="text-muted">Batches</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${data.total_pages}</h5>
                        <small class="text-muted">Total Pages</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-3 border rounded">
                        <h5 class="mb-1">${data.unassigned_pages}</h5>
                        <small class="text-muted">Unassigned Pages</small>
                    </div>
                </div>
            </div>
        `;
        batchSummaryContainer.appendChild(meta);

        const table = document.createElement('table');
        table.className = 'table table-striped batch-summary-table mt-4';
        table.innerHTML = `
            <thead class="table-light">
                <tr>
                    <th>Batch</th>
                    <th>Pages</th>
                    <th>Range</th>
                </tr>
            </thead>
            <tbody>
                ${data.batches.map(batch => `
                    <tr>
                        <td>${batch.name}</td>
                        <td><span class="badge bg-primary">${batch.page_count} page${batch.page_count > 1 ? 's' : ''}</span></td>
                        <td>${batch.page_numbers.join(', ')}</td>
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
