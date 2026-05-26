const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const video = $('video');
const canvas = $('canvas');
const startCam = $('startCam');
const snap = $('snap');
const drop = $('drop');
const fileInput = $('file');
const pick = $('pick');
const preview = $('preview');
const predictBtn = $('predict');
const result = $('result');

let stream = null;
let currentFile = null;

// Tabs
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $('tab-' + t.dataset.tab).classList.add('active');
}));

async function checkStatus() {
  try {
    const r = await fetch('/api/faceid/status');
    const s = await r.json();
    const users = s.registered_users || [];
    statusEl.textContent = `${users.length} wajah terdaftar${users.length ? ': ' + users.join(', ') : ''}`;
    statusEl.classList.add('ok');
    renderUserList(users);
  } catch (e) {
    statusEl.textContent = 'Gagal cek status';
    statusEl.classList.add('err');
  }
}

function renderUserList(users) {
  const list = $('userList');
  if (!list) return;
  if (!users.length) {
    list.innerHTML = '<li class="meta">Belum ada wajah terdaftar</li>';
    return;
  }
  list.innerHTML = users.map(name => `
    <li>
      <span>👤 ${name}</span>
      <button class="btn-danger" data-name="${name}">Hapus</button>
    </li>
  `).join('');
  list.querySelectorAll('.btn-danger').forEach(btn => {
    btn.addEventListener('click', async () => {
      const name = btn.dataset.name;
      if (!confirm(`Hapus wajah "${name}" dari database?`)) return;
      btn.disabled = true;
      btn.textContent = 'Menghapus…';
      try {
        const r = await fetch(`/api/faceid/register/${encodeURIComponent(name)}`, {method: 'DELETE'});
        if (!r.ok) {
          const err = await r.json().catch(() => ({detail: r.statusText}));
          alert(err.detail || 'Gagal: ' + r.status);
          btn.disabled = false;
          btn.textContent = 'Hapus';
          return;
        }
        await checkStatus();
      } catch (e) {
        alert(e.message);
        btn.disabled = false;
        btn.textContent = 'Hapus';
      }
    });
  });
}

// Webcam
startCam.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: true});
    video.srcObject = stream;
    startCam.textContent = 'Kamera aktif';
    startCam.disabled = true;
    snap.disabled = false;
  } catch (e) {
    alert('Tidak bisa akses kamera: ' + e.message);
  }
});

snap.addEventListener('click', async () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  await analyze({image: dataUrl});
});

// Upload
function setFile(f) {
  if (!f || !f.type.startsWith('image/')) return;
  currentFile = f;
  preview.src = URL.createObjectURL(f);
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
  const fd = new FormData();
  fd.append('file', currentFile);
  await analyze(fd);
});

async function analyze(payload) {
  snap.disabled = true;
  predictBtn.disabled = true;
  result.innerHTML = '<div class="result-empty">Memproses…</div>';
  try {
    const opts = {method: 'POST'};
    if (payload instanceof FormData) {
      opts.body = payload;
    } else {
      const fd = new FormData();
      fd.append('image', payload.image);
      opts.body = fd;
    }
    const r = await fetch('/api/faceid/analyze', opts);
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      result.innerHTML = `<div class="error">${err.detail || 'Error ' + r.status}</div>`;
      return;
    }
    renderResult(await r.json());
  } catch (e) {
    result.innerHTML = `<div class="error">${e.message}</div>`;
  } finally {
    if (stream) snap.disabled = false;
    if (currentFile) predictBtn.disabled = false;
  }
}

function renderResult({recognition, emotion}) {
  let html = '';

  // Recognition card
  if (recognition.matched) {
    html += `
      <div class="result-card primary">
        <h3>👋 Halo, ${recognition.name}!</h3>
        <p class="meta">Jarak: ${recognition.distance} (toleransi ${recognition.tolerance})</p>
      </div>`;
  } else {
    const reason = {
      no_face_detected: 'Tidak ada wajah terdeteksi',
      no_encoding:      'Wajah tidak bisa di-encode',
      empty_db:         'Database wajah kosong',
    }[recognition.reason] || `Tidak dikenali (jarak ${recognition.distance})`;
    html += `<div class="result-card"><h3>❓ Wajah tidak dikenali</h3><p class="meta">${reason}</p></div>`;
  }

  // Emotion card
  if (emotion) {
    const sorted = Object.entries(emotion.scores).sort((a, b) => b[1] - a[1]);
    html += `
      <div class="result-card primary">
        <h3>😊 Emosi: ${emotion.emotion} <span style="font-size:0.8em;color:var(--muted);">${(emotion.confidence * 100).toFixed(1)}%</span></h3>
        <div class="bar"><div class="bar-fill" style="width:${emotion.confidence * 100}%"></div></div>
        <div class="scores">
          ${sorted.map(([k, v], i) => `
            <div class="row-score ${i === 0 ? 'top' : ''}"><span>${k}</span><span>${(v * 100).toFixed(1)}%</span></div>
          `).join('')}
        </div>
      </div>`;
  }

  result.innerHTML = html;
}

// ─── Register ─────────────────────────────────────────────────────────────
const regVideo = $('regVideo');
const regName = $('regName');
const regStart = $('regStart');
const regProgress = $('regProgress');
const regBar = $('regBar');
const regStatus = $('regStatus');
let regStream = null;

async function ensureRegCam() {
  if (regStream) return;
  regStream = await navigator.mediaDevices.getUserMedia({video: true});
  regVideo.srcObject = regStream;
  await new Promise(r => regVideo.onloadedmetadata = r);
}

function snapshotDataUrl(video) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function captureOne(name) {
  const dataUrl = snapshotDataUrl(regVideo);
  const fd = new FormData();
  fd.append('name', name);
  fd.append('image', dataUrl);
  const r = await fetch('/api/faceid/register', {method: 'POST', body: fd});
  if (!r.ok) {
    const err = await r.json().catch(() => ({detail: r.statusText}));
    throw new Error(err.detail || 'Error ' + r.status);
  }
  return r.json();
}

regStart.addEventListener('click', async () => {
  const name = regName.value.trim();
  if (!name) { alert('Masukin nama dulu'); return; }
  if (!/^[\w-]+$/.test(name)) { alert('Nama: huruf/angka/_/- only'); return; }

  regStart.disabled = true;
  regName.disabled = true;
  regProgress.hidden = false;
  regStatus.textContent = 'Mengaktifkan kamera…';
  result.innerHTML = '';

  try {
    await ensureRegCam();
    await new Promise(r => setTimeout(r, 500));  // beri waktu video stabil

    const TOTAL = 5;
    for (let i = 0; i < TOTAL; i++) {
      regStatus.textContent = `Mengambil foto ${i + 1}/${TOTAL}… (tahan wajah di tengah)`;
      // jeda 1.5 detik antar shot biar pose sedikit beda
      await new Promise(r => setTimeout(r, i === 0 ? 800 : 1500));
      try {
        const res = await captureOne(name);
        regBar.style.width = (res.encodings / TOTAL * 100) + '%';
      } catch (e) {
        regStatus.textContent = `⚠ ${e.message} — coba ulang foto ke-${i + 1}`;
        i--;  // retry
        if (i < -1) break;
      }
    }

    regStatus.textContent = '';
    result.innerHTML = `<div class="success">✓ '${name}' berhasil didaftarkan dengan 5 foto.</div>`;
    await checkStatus();  // refresh user list di status pill
  } catch (e) {
    result.innerHTML = `<div class="error">${e.message}</div>`;
  } finally {
    regStart.disabled = false;
    regName.disabled = false;
    regName.value = '';
    setTimeout(() => { regProgress.hidden = true; regBar.style.width = '0%'; }, 2000);
  }
});

// Cleanup: stop both webcam streams when switching tabs away
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  const tab = t.dataset.tab;
  if (tab !== 'register' && regStream) {
    regStream.getTracks().forEach(s => s.stop());
    regStream = null;
  }
  if (tab !== 'webcam' && stream) {
    stream.getTracks().forEach(s => s.stop());
    stream = null;
    startCam.textContent = 'Mulai kamera';
    startCam.disabled = false;
    snap.disabled = true;
  }
}));

checkStatus();
