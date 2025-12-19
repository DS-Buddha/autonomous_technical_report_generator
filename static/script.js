// Global variables
let currentReportPath = null;
let currentJobId = null;
let pollingInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadReports();
    setupFormSubmission();
});

// Setup form submission
function setupFormSubmission() {
    const form = document.getElementById('reportForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await generateReport();
    });
}

// Generate report
async function generateReport() {
    const topic = document.getElementById('topic').value;
    const depth = document.getElementById('depth').value;
    const maxIterations = parseInt(document.getElementById('maxIterations').value);
    const codeExamples = document.getElementById('codeExamples').checked;

    // Hide previous results
    hideAllSections();

    // Show progress section
    showProgress(topic);

    // Disable form
    toggleForm(false);

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                topic,
                depth,
                code_examples: codeExamples,
                max_iterations: maxIterations
            })
        });

        if (!response.ok) {
            throw new Error('Failed to start report generation');
        }

        const data = await response.json();

        // Start polling for real job status instead of simulating
        // The API returns immediately, generation happens in background
        currentJobId = `${new Date().toISOString().split('T')[0].replace(/-/g, '')}_${Date.now()}_${topic.substring(0, 20).replace(/\s/g, '_')}`;

        // Poll for completion with real status updates
        pollForCompletion(topic)

    } catch (error) {
        console.error('Error:', error);

        // Clear any active polling
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }

        showError(error.message);
        toggleForm(true);
    }
}

// Poll for completion with real status checking
function pollForCompletion(topic) {
    const progressBar = document.getElementById('progressBarFill');
    const statusText = document.getElementById('progressStatus');

    let progress = 0;
    let pollCount = 0;
    const maxPolls = 60; // Max 5 minutes (60 * 5 seconds)

    const statusMessages = [
        'Initializing agents...',
        'Planning research strategy...',
        'Searching academic papers...',
        'Analyzing research findings...',
        'Generating code examples...',
        'Validating and testing code...',
        'Synthesizing final report...',
        'Finalizing document...'
    ];

    pollingInterval = setInterval(async () => {
        pollCount++;

        // Update progress incrementally
        progress = Math.min(95, Math.floor((pollCount / maxPolls) * 95));
        progressBar.style.width = progress + '%';

        // Cycle through status messages
        const messageIndex = Math.floor((pollCount / 5) % statusMessages.length);
        statusText.textContent = statusMessages[messageIndex];

        // Check if report actually exists by looking for new reports
        try {
            const response = await fetch('/api/reports');
            const data = await response.json();

            // Find the most recent report (created in last 2 minutes)
            const twoMinutesAgo = Date.now() - (2 * 60 * 1000);
            const recentReports = data.reports.filter(r => {
                const modifiedTime = new Date(r.modified_at).getTime();
                return modifiedTime > twoMinutesAgo;
            });

            // If we found a recent report, consider it completed
            if (recentReports.length > 0) {
                clearInterval(pollingInterval);
                progressBar.style.width = '100%';
                statusText.textContent = 'Report generated successfully!';

                setTimeout(() => {
                    hideAllSections();
                    showSuccess(`Report for "${topic}" has been generated successfully!`);
                    toggleForm(true);
                    loadReports();
                }, 1000);
            }

            // Timeout after max polls
            if (pollCount >= maxPolls) {
                clearInterval(pollingInterval);
                showError('Report generation is taking longer than expected. Please check the reports list below.');
                toggleForm(true);
                loadReports();
            }

        } catch (error) {
            console.error('Error checking completion:', error);
        }

    }, 5000); // Poll every 5 seconds
}

// Check for completion
async function checkCompletion(topic) {
    // Reload reports and check if new report exists
    await loadReports();

    const progressBar = document.getElementById('progressBarFill');
    progressBar.style.width = '100%';

    setTimeout(() => {
        hideAllSections();
        showSuccess(`Report for "${topic}" has been generated successfully!`);
        toggleForm(true);
        loadReports();
    }, 1000);
}

// Show progress section
function showProgress(topic) {
    const progressSection = document.getElementById('progressSection');
    const progressTopic = document.getElementById('progressTopic');
    const progressBar = document.getElementById('progressBarFill');

    progressTopic.textContent = topic;
    progressBar.style.width = '0%';
    progressSection.style.display = 'block';
}

// Show success
function showSuccess(message) {
    const resultSection = document.getElementById('resultSection');
    const resultMessage = document.getElementById('resultMessage');

    resultMessage.textContent = message;
    resultSection.style.display = 'block';
}

// Show error
function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');

    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

// Hide all sections
function hideAllSections() {
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
}

// Toggle form state
function toggleForm(enabled) {
    const form = document.getElementById('reportForm');
    const inputs = form.querySelectorAll('input, select, button');
    const generateBtn = document.getElementById('generateBtn');
    const btnText = generateBtn.querySelector('.btn-text');
    const btnLoader = generateBtn.querySelector('.btn-loader');

    inputs.forEach(input => {
        input.disabled = !enabled;
    });

    if (enabled) {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    } else {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline';
    }
}

// Reset form
function resetForm() {
    // Clear any active polling
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }

    hideAllSections();
    toggleForm(true);
    document.getElementById('reportForm').reset();
    document.getElementById('codeExamples').checked = true;
    currentJobId = null;
    currentReportPath = null;
}

// Load reports
async function loadReports() {
    const reportsList = document.getElementById('reportsList');

    try {
        reportsList.innerHTML = '<div class="loading-state">Loading reports...</div>';

        const response = await fetch('/api/reports');
        const data = await response.json();

        if (data.reports.length === 0) {
            reportsList.innerHTML = '<div class="empty-state">No reports generated yet. Create your first report above!</div>';
            return;
        }

        reportsList.innerHTML = data.reports.map(report => `
            <div class="report-item">
                <div class="report-info">
                    <div class="report-name">${report.filename}</div>
                    <div class="report-meta">
                        <span>📅 ${formatDate(report.modified_at)}</span>
                        <span>📊 ${formatSize(report.size)}</span>
                    </div>
                </div>
                <div class="report-actions">
                    <button class="btn btn-small btn-secondary" onclick="viewReportByFilename('${report.filename}')">
                        👁️ View
                    </button>
                    <button class="btn btn-small btn-outline" onclick="downloadReportByFilename('${report.filename}')">
                        ⬇️ Download
                    </button>
                    <button class="btn btn-small btn-danger" onclick="deleteReport('${report.filename}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading reports:', error);
        reportsList.innerHTML = '<div class="empty-state">Failed to load reports. Please try again.</div>';
    }
}

// View report
async function viewReportByFilename(filename) {
    try {
        const response = await fetch(`/api/report/${filename}`);
        const data = await response.json();

        currentReportPath = filename;

        // Convert markdown to HTML (basic conversion)
        const htmlContent = markdownToHtml(data.content);

        document.getElementById('modalTitle').textContent = filename;
        document.getElementById('reportContent').innerHTML = htmlContent;
        document.getElementById('reportModal').style.display = 'flex';

    } catch (error) {
        console.error('Error viewing report:', error);
        alert('Failed to load report. Please try again.');
    }
}

// View report (called from success section)
function viewReport() {
    loadReports();
}

// Download report
async function downloadReportByFilename(filename) {
    try {
        window.location.href = `/api/download/${filename}`;
    } catch (error) {
        console.error('Error downloading report:', error);
        alert('Failed to download report. Please try again.');
    }
}

// Download current report from modal
function downloadCurrentReport() {
    if (currentReportPath) {
        downloadReportByFilename(currentReportPath);
    }
}

// Delete report
async function deleteReport(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/report/${filename}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            await loadReports();
            alert('Report deleted successfully');
        } else {
            throw new Error('Failed to delete report');
        }

    } catch (error) {
        console.error('Error deleting report:', error);
        alert('Failed to delete report. Please try again.');
    }
}

// Close modal
function closeModal() {
    document.getElementById('reportModal').style.display = 'none';
    currentReportPath = null;
}

// Close modal on outside click
window.onclick = function(event) {
    const modal = document.getElementById('reportModal');
    if (event.target === modal) {
        closeModal();
    }
}

// Utility: Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
        return 'Today';
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return `${diffDays} days ago`;
    } else {
        return date.toLocaleDateString();
    }
}

// Utility: Format file size
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Basic markdown to HTML converter
function markdownToHtml(markdown) {
    let html = markdown;

    // Code blocks
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Lists
    html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Paragraphs
    html = html.split('\n\n').map(para => {
        if (para.startsWith('<') || para.trim() === '') {
            return para;
        }
        return `<p>${para}</p>`;
    }).join('\n');

    return html;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
