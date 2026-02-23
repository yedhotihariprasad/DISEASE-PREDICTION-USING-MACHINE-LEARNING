const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('image');
const predsDiv = document.getElementById('preds');
const imgPreview = document.getElementById('imgPreview');
const topK = document.getElementById('topK');
const thresh = document.getElementById('thresh');
const threshVal = document.getElementById('threshVal');

thresh.addEventListener('input', () => { threshVal.innerText = parseFloat(thresh.value).toFixed(2); });

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) {
    imgPreview.src = '';
    return;
  }
  const url = URL.createObjectURL(file);
  imgPreview.src = url;
  // update file label
  const label = document.getElementById('fileLabelText');
  if (label) label.innerText = file.name;
});

function renderPredictions(predictions, threshold=0.0) {
  predsDiv.innerHTML = '';
  predictions.forEach(p => {
    if (p.confidence < threshold) return;
    const row = document.createElement('div');
    row.className = 'predRow';
    const label = document.createElement('div');
    label.className = 'predLabel';
    label.innerText = p.label;
    const barWrap = document.createElement('div');
    barWrap.className = 'barWrap';
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = Math.round(p.confidence * 100) + '%';
    bar.innerText = (p.confidence * 100).toFixed(1) + '%';
    barWrap.appendChild(bar);
    row.appendChild(label);
    row.appendChild(barWrap);
    predsDiv.appendChild(row);
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return alert('Choose an image');
  const fd = new FormData();
  fd.append('image', file);
  const k = parseInt(topK.value || '3', 10);
  const threshold = parseFloat(thresh.value || '0');
  predsDiv.innerText = 'Uploading...';
  try {
    const res = await fetch('/predict?top_k=' + k, { method: 'POST', body: fd });
    const j = await res.json();
    if (res.ok) {
      renderPredictions(j.predictions || [], threshold);
      // show filename for accessibility
      const fileLabel = document.getElementById('fileLabelText');
      if (fileLabel) fileLabel.innerText = file.name;
    } else {
      predsDiv.innerText = 'Error: ' + (j.error || JSON.stringify(j));
    }
  } catch (err) {
    predsDiv.innerText = 'Error: ' + err.message;
  }
});

document.getElementById('resetBtn').addEventListener('click', () => {
  fileInput.value = '';
  imgPreview.src = '';
  document.getElementById('fileLabelText').innerText = 'Choose image';
  predsDiv.innerHTML = '';
  thresh.value = 0; document.getElementById('threshVal').innerText = '0.00';
});
