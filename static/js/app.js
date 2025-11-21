// ===================================
// GLOBAL STATE
// ===================================
let selectedModel = 'bert';
let allImages = [];
let currentImageIndex = 0;

// ===================================
// THEME MANAGEMENT
// ===================================
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const themeIcon = document.querySelector('.theme-icon');
    if (themeIcon) {
        themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}

// ===================================
// MODEL SELECTION
// ===================================
function selectModel(model) {
    selectedModel = model;
    
    // Update UI
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    const selectedCard = document.getElementById(`model-${model}`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
    }
    
    // Show/hide top-k option for BERT
    const topkContainer = document.getElementById('topk-container');
    if (topkContainer) {
        topkContainer.style.display = model === 'bert' ? 'flex' : 'none';
    }
}

// ===================================
// PREDICTION
// ===================================
async function predict() {
    const text = document.getElementById('input-text').value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    const topK = parseInt(document.getElementById('top-k')?.value || '3', 10);
    const useGpu = document.getElementById('use-gpu')?.checked || false;
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: selectedModel === 'bert' ? 'bert' : 'baseline',
                text: text,
                top_k: topK,
                use_gpu: useGpu
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        await displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// ===================================
// HELPER FUNCTION - GET LABEL NAMES
// ===================================
let cachedLabelNames = null;

async function getLabelNames() {
    if (cachedLabelNames) {
        return cachedLabelNames;
    }
    
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        // Try to get label names from CSV files
        if (data.csvs && data.csvs.length > 0) {
            for (const csvFile of data.csvs) {
                if (csvFile.includes('bert_per_class') || csvFile.includes('compare_per_class')) {
                    const csvResponse = await fetch(`/csvs/${csvFile}`);
                    const csvText = await csvResponse.text();
                    const lines = csvText.trim().split('\n');
                    
                    if (lines.length > 1) {
                        const headers = lines[0].split(',');
                        const labelIdIndex = headers.findIndex(h => h.toLowerCase().includes('label_id'));
                        const labelNameIndex = headers.findIndex(h => h.toLowerCase() === 'label');
                        
                        if (labelIdIndex !== -1 && labelNameIndex !== -1) {
                            // Create array of 14 elements (max ID is 12, but we need index 13)
                            const labelMap = new Array(14).fill('Unknown');
                            
                            // Parse each line and map label_id to label name
                            lines.slice(1).forEach(line => {
                                const cols = line.split(',');
                                const id = parseInt(cols[labelIdIndex]);
                                const name = cols[labelNameIndex];
                                if (!isNaN(id) && name) {
                                    labelMap[id] = name;
                                }
                            });
                            
                            cachedLabelNames = labelMap;
                            return cachedLabelNames;
                        }
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error loading label names:', error);
    }
    
    // Fallback to generic labels
    cachedLabelNames = [
        'Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process',
        'Privacy', 'Attorneys', 'Unions', 'Economic Activity',
        'Judicial Power', 'Federalism', 'Interstate Relations', 'Federal Taxation',
        'Miscellaneous', 'Other'
    ];
    return cachedLabelNames;
}

// ===================================
// DISPLAY RESULTS
// ===================================
async function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    
    if (!resultsSection || !resultsContent) return;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    let html = '';
    
    if (data.model === 'baseline') {
        // Baseline model results
        html = `
            <div class="result-item">
                <div class="result-label">
                    🎯 Predicted Category: <span style="color: var(--accent-primary)">${data.label}</span>
                </div>
                <div style="margin-top: 0.5rem; color: var(--text-secondary);">
                    Category ID: ${data.pred_id}
                </div>
            </div>
        `;
        
        if (data.probs && data.probs.length > 0) {
            // Get label names from the API response
            const labelNames = await getLabelNames();
            
            const topProbs = data.probs
                .map((p, i) => ({ id: i, prob: p, name: labelNames[i] || `Category ${i}` }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, 5);
            
            html += '<div style="margin-top: 1.5rem;"><h3 style="margin-bottom: 1rem;">Top Probabilities:</h3>';
            
            topProbs.forEach((item, index) => {
                const percentage = (item.prob * 100).toFixed(2);
                html += `
                    <div class="result-item" style="animation-delay: ${index * 0.1}s;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600;">${item.name}</span>
                            <span class="confidence-text">${percentage}%</span>
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.25rem;">
                            Category ID: ${item.id}
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${percentage}%;"></div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
        }
    } else {
        // BERT model results
        html = '<div style="margin-bottom: 1rem;"><h3>Top Predictions:</h3></div>';
        
        data.preds.forEach((pred, index) => {
            const percentage = (pred.prob * 100).toFixed(2);
            html += `
                <div class="result-item" style="animation-delay: ${index * 0.1}s;">
                    <div class="result-label">
                        ${index + 1}. ${pred.label}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">
                        Category ID: ${pred.id}
                    </div>
                    <div class="result-confidence">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${percentage}%;"></div>
                        </div>
                        <span class="confidence-text">${percentage}%</span>
                    </div>
                </div>
            `;
        });
    }
    
    resultsContent.innerHTML = html;
}

// ===================================
// VISUALIZATIONS PAGE
// ===================================
async function loadVisualizations() {
    showLoading(true);
    
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        // Load figures
        if (data.figures && data.figures.length > 0) {
            allImages = data.figures;
            displayImages(data.figures);
        }
        
        // Load CSV tables
        if (data.csvs && data.csvs.length > 0) {
            await displayCSVTables(data.csvs);
        }
    } catch (error) {
        console.error('Error loading visualizations:', error);
        showError('Failed to load visualizations');
    } finally {
        showLoading(false);
    }
}

function displayImages(figures) {
    const vizGrid = document.getElementById('viz-grid');
    if (!vizGrid) return;
    
    vizGrid.innerHTML = '';
    
    figures.forEach((filename, index) => {
        const vizItem = document.createElement('div');
        vizItem.className = 'viz-item';
        vizItem.style.animationDelay = `${index * 0.1}s`;
        
        const img = document.createElement('img');
        img.src = `/figures/${filename}`;
        img.alt = filename;
        
        const caption = document.createElement('div');
        caption.className = 'viz-caption';
        caption.textContent = formatFilename(filename);
        
        vizItem.appendChild(img);
        vizItem.appendChild(caption);
        
        vizItem.addEventListener('click', () => openModal(index));
        
        vizGrid.appendChild(vizItem);
    });
}

async function displayCSVTables(csvFiles) {
    const csvContainer = document.getElementById('csv-container');
    if (!csvContainer) return;
    
    csvContainer.innerHTML = '';
    
    for (const filename of csvFiles) {
        try {
            const response = await fetch(`/csvs/${filename}`);
            const text = await response.text();
            
            const tableWrapper = document.createElement('div');
            tableWrapper.style.marginBottom = '2rem';
            
            const title = document.createElement('h3');
            title.textContent = formatFilename(filename);
            title.style.marginBottom = '1rem';
            
            const table = createTableFromCSV(text);
            
            tableWrapper.appendChild(title);
            tableWrapper.appendChild(table);
            csvContainer.appendChild(tableWrapper);
        } catch (error) {
            console.error(`Error loading ${filename}:`, error);
        }
    }
}

function createTableFromCSV(csvText) {
    const rows = csvText.trim().split('\n').map(row => row.split(','));
    const table = document.createElement('table');
    table.className = 'csv-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    rows[0].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body (limit to 50 rows)
    const tbody = document.createElement('tbody');
    const maxRows = Math.min(rows.length, 51);
    for (let i = 1; i < maxRows; i++) {
        const tr = document.createElement('tr');
        rows[i].forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    
    return table;
}

// ===================================
// MODAL FUNCTIONALITY
// ===================================
function openModal(index) {
    const modal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCaption = document.getElementById('modal-caption');
    
    if (!modal || !modalImage || !modalCaption) return;
    
    currentImageIndex = index;
    modalImage.src = `/figures/${allImages[index]}`;
    modalCaption.textContent = formatFilename(allImages[index]);
    modal.classList.add('active');
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    const modal = document.getElementById('image-modal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function nextImage() {
    currentImageIndex = (currentImageIndex + 1) % allImages.length;
    openModal(currentImageIndex);
}

function previousImage() {
    currentImageIndex = (currentImageIndex - 1 + allImages.length) % allImages.length;
    openModal(currentImageIndex);
}

// ===================================
// UTILITY FUNCTIONS
// ===================================
function formatFilename(filename) {
    return filename
        .replace(/\.(png|jpg|jpeg|csv)$/i, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.toggle('active', show);
    }
}

function showError(message) {
    alert(`Error: ${message}`);
}

// ===================================
// EVENT LISTENERS
// ===================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme
    initTheme();
    
    // Theme toggle button
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Predict button
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', predict);
    }
    
    // Default model selection
    if (document.getElementById('model-bert')) {
        selectModel('bert');
    }
    
    // Load visualizations if on visualizations page
    if (document.getElementById('viz-grid')) {
        loadVisualizations();
    }
    
    // Keyboard navigation for modal
    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('image-modal');
        if (modal && modal.classList.contains('active')) {
            if (e.key === 'Escape') {
                closeModal();
            } else if (e.key === 'ArrowRight') {
                nextImage();
            } else if (e.key === 'ArrowLeft') {
                previousImage();
            }
        }
    });
});

// Make functions global for onclick handlers
window.selectModel = selectModel;
window.closeModal = closeModal;
window.nextImage = nextImage;
window.previousImage = previousImage;

