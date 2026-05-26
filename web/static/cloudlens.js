const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const drop = $('drop');
const fileInput = $('file');
const pick = $('pick');
const preview = $('preview');
const predictBtn = $('predict');
const result = $('result');

let currentFile = null;

async function checkStatus() {
  try {
    const r = await fetch('/api/cloudlens/status');
    const s = await r.json();
    if (s.ready) {
      statusEl.textContent = 'Model siap';
      statusEl.classList.add('ok');
    } else {
      statusEl.textContent = '⚠ ' + (s.hint || 'Model belum siap');
      statusEl.classList.add('warn');
    }
  } catch (e) {
    statusEl.textContent = 'Gagal cek status';
    statusEl.classList.add('err');
  }
}

function setFile(f) {
  if (!f || !f.type.startsWith('image/')) return;
  currentFile = f;
  const url = URL.createObjectURL(f);
  preview.src = url;
  preview.hidden = false;
  predictBtn.disabled = false;
}

pick.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
fileInput.addEventListener('change', (e) => setFile(e.target.files[0]));
drop.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));

predictBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  predictBtn.disabled = true;
  predictBtn.textContent = 'Menganalisis…';
  result.className = '';
  result.innerHTML = '<div class="result-empty">Memproses…</div>';

  const fd = new FormData();
  fd.append('file', currentFile);
  try {
    const r = await fetch('/api/cloudlens/predict', { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      result.innerHTML = `<div class="error">${err.detail || 'Error ' + r.status}</div>`;
      return;
    }
    const data = await r.json();
    renderResults(data);
  } catch (e) {
    result.innerHTML = `<div class="error">${e.message}</div>`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Klasifikasi';
  }
});

function renderResults({results, latency_ms}) {
  const html = results.map((r, i) => `
    <div class="result-card ${i === 0 ? 'primary' : ''}">
      <h3>${r.label} <span style="font-size:0.8em;color:var(--muted);">${(r.confidence * 100).toFixed(1)}%</span></h3>
      <div class="bar"><div class="bar-fill" style="width:${r.confidence * 100}%"></div></div>
      ${i === 0 ? `<p class="desc">${r.desc}</p><p class="meta">☁ ${r.weather}</p>` : ''}
    </div>
  `).join('');
  result.innerHTML = html + `<p class="meta" style="text-align:right;color:var(--muted);font-size:0.8em;">${latency_ms} ms</p>`;
}

checkStatus();
