const API_BASE = 'http://localhost:8000';

// --- Tab switching ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.mode}`).classList.add('active');
    hideResults();
  });
});

// --- Image upload ---
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropContent = document.getElementById('dropContent');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const searchImageBtn = document.getElementById('searchImageBtn');
let selectedFile = null;

dropZone.addEventListener('click', e => {
  if (e.target === dropZone || e.target.closest('#dropContent')) fileInput.click();
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setImageFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setImageFile(fileInput.files[0]);
});

document.getElementById('removeImg').addEventListener('click', e => {
  e.stopPropagation();
  clearImage();
});

function setImageFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropContent.style.display = 'none';
    imagePreview.style.display = 'block';
    searchImageBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  dropContent.style.display = 'flex';
  imagePreview.style.display = 'none';
  searchImageBtn.disabled = true;
}

// --- Search handlers ---
searchImageBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append('file', selectedFile);
  doSearch('/api/search/image', formData, true);
});

document.getElementById('searchNameBtn').addEventListener('click', () => {
  const name = document.getElementById('nameInput').value.trim();
  if (!name) return;
  doSearch('/api/search/name', { query: name });
});

document.getElementById('searchDescBtn').addEventListener('click', () => {
  const desc = document.getElementById('descInput').value.trim();
  if (!desc) return;
  doSearch('/api/search/description', { query: desc });
});

// allow Enter key on text inputs
document.getElementById('nameInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('searchNameBtn').click();
});

// --- Core search function ---
async function doSearch(endpoint, body, isFormData = false) {
  showLoading();
  try {
    const options = { method: 'POST' };
    if (isFormData) {
      options.body = body;
    } else {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(body);
    }
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    showResults(data.results);
  } catch (err) {
    showError(err.message.includes('Failed to fetch')
      ? 'Could not connect to server. Make sure the backend is running.'
      : err.message);
  }
}

// --- UI helpers ---
function showLoading() {
  document.getElementById('loading').style.display = 'flex';
  document.getElementById('resultsCard').style.display = 'none';
  document.getElementById('errorCard').style.display = 'none';
}

function hideResults() {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('resultsCard').style.display = 'none';
  document.getElementById('errorCard').style.display = 'none';
}

function showError(msg) {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('errorCard').style.display = 'flex';
  document.getElementById('errorMsg').textContent = msg;
}

function showResults(results) {
  document.getElementById('loading').style.display = 'none';
  const card = document.getElementById('resultsCard');
  const list = document.getElementById('resultsList');
  const count = document.getElementById('resultCount');

  if (!results || results.length === 0) {
    card.style.display = 'block';
    count.textContent = '0 found';
    list.innerHTML = `<div style="text-align:center;padding:32px;color:#94a3b8;">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:12px"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
      <p>No parts found. Try a different search.</p>
    </div>`;
    return;
  }

  count.textContent = `${results.length} found`;
  list.innerHTML = results.map((r, i) => `
    <div class="result-item">
      <div class="result-rank ${i === 0 ? 'top' : ''}">${i + 1}</div>
      <div class="result-info">
        <div class="result-name">${escHtml(r.part_name)}</div>
        <div class="result-meta">
          <span class="badge">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
            ${escHtml(r.location)}
          </span>
          ${r.part_id ? `<span class="badge">ID: ${escHtml(r.part_id)}</span>` : ''}
          ${r.category ? `<span class="badge">${escHtml(r.category)}</span>` : ''}
        </div>
      </div>
      ${r.confidence !== undefined ? `<span class="confidence">${Math.round(r.confidence * 100)}%</span>` : ''}
    </div>
  `).join('');

  card.style.display = 'block';
}

function escHtml(str) {
  return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// --- Demo mode (no backend) ---
// Inject mock results for UI preview
if (window.location.search.includes('demo')) {
  setTimeout(() => {
    showResults([
      { part_name: 'Valve Stem Assembly', location: 'Shelf B-12, Bin 4', part_id: 'HP-4521-B', category: 'Hydraulics', confidence: 0.97 },
      { part_name: 'Valve Stem (Short)', location: 'Shelf B-12, Bin 5', part_id: 'HP-4521-A', category: 'Hydraulics', confidence: 0.81 },
      { part_name: 'Pressure Relief Valve', location: 'Shelf C-03, Bin 2', part_id: 'HP-7832-C', category: 'Hydraulics', confidence: 0.64 },
    ]);
  }, 800);
  showLoading();
}
