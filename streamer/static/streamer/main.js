const goLiveBtn = document.getElementById('go-live');
const langSelect = document.getElementById('lang');
const status = document.getElementById('status');
const player = document.getElementById('player');
const recordingsContainer = document.getElementById('recordings');
const deleteAllBtn = document.getElementById('delete-all');
const refreshBtn = document.getElementById('refresh-recordings');

let audioQueue = [];
let currentIndex = 0;

/**
 * Play next audio file in queue
 */
function playNext() {
  if (currentIndex >= audioQueue.length) {
    status.textContent = 'Live session finished.';
    goLiveBtn.disabled = false;
    return;
  }

  player.src = audioQueue[currentIndex];
  player.play();

  status.textContent = `Playing segment ${currentIndex + 1} / ${audioQueue.length}`;
  currentIndex++;
}

/**
 * Fetch existing recordings and render them
 */
async function fetchRecordings() {
  if (!recordingsContainer) return;
  recordingsContainer.innerHTML = 'Loading...';
  try {
    const res = await fetch('/recordings/');
    const data = await res.json();
    if (!res.ok) {
      recordingsContainer.innerHTML = 'Failed to load recordings';
      return;
    }
    renderSessions(data.sessions || []);
  } catch (err) {
    recordingsContainer.innerHTML = 'Error loading recordings';
    console.error(err);
  }
}

function renderSessions(sessions) {
  if (!recordingsContainer) return;
  if (!sessions.length) {
    recordingsContainer.innerHTML = '<div class="empty">No recordings yet</div>';
    return;
  }

  const html = sessions.map(s => `
    <div class="recording-card">
      <div class="header"><div class="when">${s.created_at}</div><div class="count">${s.files.length} segments</div></div>
      <div class="files">
        ${s.files.map(f => `
          <div class="file-row">
            <audio controls src="${f.url}"></audio>
            <a href="${f.url}" download style="color:var(--accent);font-weight:600;">Download</a>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  recordingsContainer.innerHTML = html;
}

/**
 * Delete all recordings
 */
if (deleteAllBtn) {
  deleteAllBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to delete all recordings? This cannot be undone.')) return;
    deleteAllBtn.disabled = true;
    deleteAllBtn.textContent = 'Deleting...';

    try {
      const res = await fetch('/delete-recordings/', { method: 'POST' });
      const data = await res.json();
      status.textContent = `Deleted ${data.deleted} files`;
      await fetchRecordings();
    } catch (err) {
      status.textContent = 'Delete failed: ' + err;
      console.error(err);
    }

    deleteAllBtn.disabled = false;
    deleteAllBtn.textContent = 'Delete All';
  });
}

if (refreshBtn) {
  refreshBtn.addEventListener('click', fetchRecordings);
}

/**
 * Handle Go Live click
 */
goLiveBtn.addEventListener('click', async () => {
  goLiveBtn.disabled = true;
  status.textContent = 'Generating commentary and voice... please wait.';
  audioQueue = [];
  currentIndex = 0;

  try {
    const res = await fetch('/go-live/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        language: langSelect.value
      })
    });

    const data = await res.json();

    if (!res.ok || !data.audio_urls || data.audio_urls.length === 0) {
      status.textContent = 'Error: ' + (data.error || 'No audio generated');
      goLiveBtn.disabled = false;
      return;
    }

    audioQueue = data.audio_urls;
    status.textContent = `Starting live audio (1 / ${audioQueue.length})`;

    // refresh recordings list after generation
    fetchRecordings();

    playNext();

  } catch (err) {
    status.textContent = 'Request failed: ' + err;
    goLiveBtn.disabled = false;
  }
});

/**
 * Auto-play next chunk when current finishes
 */
player.addEventListener('ended', playNext);

// Load recordings on page load
window.addEventListener('load', fetchRecordings);
