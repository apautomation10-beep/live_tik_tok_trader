// Safe merged main.js: preserves previous go-live + recordings + delete functionality
// and adds TikTok reply polling with interruption/resume handling and SSE support.

const goLiveBtn = document.getElementById('go-live');
const langSelect = document.getElementById('lang');
const status = document.getElementById('status');
const player = document.getElementById('player');
const recordingsContainer = document.getElementById('recordings');
const deleteAllBtn = document.getElementById('delete-all');
const refreshBtn = document.getElementById('refresh-recordings');

let audioQueue = [];
let currentIndex = 0;
let sse = null;

function safeEl(el, name) {
  if (!el) console.warn(`${name} not found in DOM`);
  return el;
}

/** Play next audio in queue */
function playNext() {
  if (!player) return;
  if (currentIndex >= audioQueue.length) {
    if (status) status.textContent = 'Live session finished.';
    if (goLiveBtn) goLiveBtn.disabled = false;
    return;
  }

  try {
    player.src = audioQueue[currentIndex];
    player.play().catch(() => {});
    if (status) status.textContent = `Playing segment ${currentIndex + 1} / ${audioQueue.length}`;
    currentIndex++;
  } catch (e) {
    console.error('playNext failed', e);
  }
}

/** Fetch existing recordings and render */
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

/** Delete all recordings (commentary and replies) */
if (deleteAllBtn) {
  deleteAllBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to delete all audio files (commentary and replies)? This cannot be undone.')) return;
    deleteAllBtn.disabled = true;
    deleteAllBtn.textContent = 'Deleting...';

    try {
      const res = await fetch('/delete-recordings/', { method: 'POST' });
      const data = await res.json();
      if (status) status.textContent = `Deleted ${data.deleted} files`;
      await fetchRecordings();
    } catch (err) {
      if (status) status.textContent = 'Delete failed: ' + err;
      console.error(err);
    }

    deleteAllBtn.disabled = false;
    deleteAllBtn.textContent = 'Delete All';
  });
}

if (refreshBtn) {
  refreshBtn.addEventListener('click', fetchRecordings);
}

/** Handle Go Live click (generate commentary + start playback) */
if (goLiveBtn) {
  goLiveBtn.addEventListener('click', async () => {
    goLiveBtn.disabled = true;
    if (status) status.textContent = 'Queued live generation... will play when audio is ready.';
    audioQueue = [];
    currentIndex = 0;

    // close existing SSE if any
    if (sse) {
      try { sse.close(); } catch (e) {}
      sse = null;
    }

    try {
      const res = await fetch('/go-live/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: (langSelect && langSelect.value) || 'en' })
      });

      const data = await res.json();

      if (!res.ok || data.status !== 'queued' || !data.session_ts) {
        if (status) status.textContent = 'Error: ' + (data.error || 'Failed to queue live session');
        if (goLiveBtn) goLiveBtn.disabled = false;
        return;
      }

      const sessionId = data.session_ts;
      const total = data.total_chunks || 0;
      if (status) status.textContent = `Session queued with ${total} segments. Waiting for generation...`;

      // poll recordings for this session until we have all segments
      let pollTimer = null;
      const pollFn = async () => {
        try {
          const r = await fetch('/recordings/');
          const payload = await r.json();
          const sessions = payload.sessions || [];
          const found = sessions.find(s => String(s.session) === String(sessionId));
          const have = (found && found.files && found.files.length) || 0;

          if (total > 0) {
            if (status) status.textContent = `Generating audio ${have} / ${total}`;

            if (have >= total && total > 0) {
              // build audio queue sorted by filename
              const files = (found.files || []).sort((a,b) => a.filename.localeCompare(b.filename));
              audioQueue = files.map(f => f.url);
              if (status) status.textContent = `Starting live audio (1 / ${audioQueue.length})`;
              fetchRecordings();
              if (pollTimer) clearInterval(pollTimer);
              playNext();
            }
          } else {
            // total unknown: start playing as soon as at least one file appears
            if (have > 0 && audioQueue.length === 0) {
              const files = (found.files || []).sort((a,b) => a.filename.localeCompare(b.filename));
              audioQueue = files.map(f => f.url);
              if (status) status.textContent = `Starting live audio (1 / ${audioQueue.length})`;
              fetchRecordings();
              playNext();
            } else if (have > 0 && audioQueue.length > 0) {
              // append any newly generated files
              const files = (found.files || []).sort((a,b) => a.filename.localeCompare(b.filename));
              const urls = files.map(f => f.url);
              for (const u of urls) {
                if (!audioQueue.includes(u)) {
                  audioQueue.push(u);
                  if (status) status.textContent = `Received new audio (total ${audioQueue.length})`;
                }
              }
            }
          }
        } catch (err) {
          console.error('Polling recordings failed', err);
        }
      };

      // start polling every 2 seconds
      pollTimer = setInterval(pollFn, 2000);
      // run immediately once
      pollFn();

    } catch (err) {
      if (status) status.textContent = 'Request failed: ' + err;
      if (goLiveBtn) goLiveBtn.disabled = false;
    }
  });
}

// Auto-play next chunk when current finishes
if (player) player.addEventListener('ended', playNext);

// --- Reply polling and interruption handling ---
let replyQueue = [];
let isPlayingReply = false;
let pausedComment = null; // { index, time }

async function pollForReplies() {
  // only poll while a live session is running (goLiveBtn.disabled === true)
  if (!goLiveBtn || goLiveBtn.disabled !== true) return;
  try {
    const res = await fetch('/replies/next/');
    if (!res.ok) return;
    const data = await res.json();
    if (data.found && data.url) {
      replyQueue.push(data.url);
      if (status) status.textContent = `Queued live reply (${replyQueue.length} pending)`;
      if (!isPlayingReply) startReplyPlayback();
    }
  } catch (err) {
    console.error('Reply poll failed', err);
  }
}

async function startReplyPlayback() {
  if (!replyQueue.length) return;

  // pause commentary and remember position
  if (!isPlayingReply) {
    if (player && !player.paused) {
      const currentPlayingIndex = Math.max(0, currentIndex - 1);
      pausedComment = { index: currentPlayingIndex, time: player.currentTime || 0 };
      try { player.pause(); } catch (e) {}
    } else {
      pausedComment = null;
    }
  }

  isPlayingReply = true;

  while (replyQueue.length) {
    const url = replyQueue.shift();
    if (status) status.textContent = `Playing live reply (${replyQueue.length} remaining)`;
    await playReplyOnce(url);
  }

  await resumeCommentary();
  isPlayingReply = false;
}

function playReplyOnce(url) {
  return new Promise((resolve) => {
    try {
      const r = new Audio(url);
      r.preload = 'auto';
      r.addEventListener('ended', () => resolve());
      r.addEventListener('error', (e) => {
        console.error('Reply play error', e);
        resolve();
      });
      r.play().catch((e) => {
        console.error('Reply play failed', e);
        resolve();
      });
    } catch (e) {
      console.error('playReplyOnce failed', e);
      resolve();
    }
  });
}

function resumeCommentary() {
  return new Promise((resolve) => {
    if (status) status.textContent = 'Resuming live commentary';

    if (!pausedComment) {
      if (player && player.src && player.paused) player.play().catch(() => {});
      return resolve();
    }

    const resumeIndex = pausedComment.index;
    const resumeTime = pausedComment.time || 0;

    // validate index
    if (!audioQueue[resumeIndex]) {
      // nothing to resume, try to continue with next segment
      if (player && player.paused) player.play().catch(() => {});
      return resolve();
    }

    try {
      player.src = audioQueue[resumeIndex];
      currentIndex = resumeIndex + 1;

      const onCanSeek = () => {
        try {
          player.currentTime = Math.max(0, Math.min(resumeTime, player.duration || resumeTime));
        } catch (e) { console.warn('Failed to set resume time', e); }
        player.play().then(() => {
          if (status) status.textContent = `Playing segment ${currentIndex} / ${audioQueue.length}`;
          player.removeEventListener('canplay', onCanSeek);
          resolve();
        }).catch((e) => {
          console.error('Resume play failed', e);
          player.removeEventListener('canplay', onCanSeek);
          resolve();
        });
      };

      player.addEventListener('canplay', onCanSeek);
      setTimeout(() => { if (player.readyState >= 2) onCanSeek(); }, 200);
    } catch (e) {
      console.error('resumeCommentary failed', e);
      resolve();
    }
  });
}

setInterval(pollForReplies, 2000);

// Load recordings on page load
window.addEventListener('load', () => {
  fetchRecordings();
  fetchAllAudio();
});

// --- All Audio manager ---
const allAudioList = document.getElementById('all-audio-list');
const refreshAllBtn = document.getElementById('refresh-all-audio');
const playAllBtn = document.getElementById('play-all');
const deleteAllAudioBtn = document.getElementById('delete-all-audio');
const audioFilter = document.getElementById('audio-filter');

let allAudioFiles = [];
let playlist = [];
let playlistIndex = 0;

async function fetchAllAudio() {
  if (!allAudioList) return;
  allAudioList.innerHTML = 'Loading...';
  try {
    const res = await fetch('/all-audio/');
    const data = await res.json();
    if (!res.ok) {
      allAudioList.innerHTML = 'Failed to load audio files';
      return;
    }
    allAudioFiles = data.files || [];
    renderAllAudio();
  } catch (err) {
    allAudioList.innerHTML = 'Error loading audio files';
    console.error(err);
  }
}

function renderAllAudio() {
  if (!allAudioList) return;
  const filter = (audioFilter && audioFilter.value) || 'all';
  const files = allAudioFiles.filter(f => filter === 'all' ? true : f.type === filter);
  if (!files.length) {
    allAudioList.innerHTML = '<div class="empty">No audio files</div>';
    return;
  }

  const html = files.map(f => `
    <div class="recording-card">
      <div class="header"><div style="font-size:0.9rem">${f.filename}</div><div style="color:var(--muted);font-size:0.85rem">${f.created || ''} · ${Math.round(f.size/1024)} KB</div></div>
      <div class="files">
        <div class="file-row">
          <audio controls src="${f.url}"></audio>
          <a href="${f.url}" download style="color:var(--accent);font-weight:600;">Download</a>
          <button data-fname="${f.filename}" class="delete-file" style="background:transparent;border:1px solid var(--border);color:var(--muted);padding:8px;border-radius:8px;margin-left:6px">Delete</button>
        </div>
      </div>
    </div>
  `).join('');

  allAudioList.innerHTML = html;

  // attach delete listeners
  document.querySelectorAll('.delete-file').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const fn = btn.dataset.fname;
      if (!confirm(`Delete ${fn}?`)) return;
      btn.disabled = true;
      try {
        const res = await fetch('/delete-file/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: fn })
        });
        const data = await res.json();
        if (res.ok && data.deleted) {
          await fetchAllAudio();
          fetchRecordings();
        } else {
          alert('Delete failed: ' + (data.error || 'unknown'));
        }
      } catch (err) {
        console.error('delete single failed', err);
        alert('Delete failed');
      }
      btn.disabled = false;
    });
  });
}

if (refreshAllBtn) refreshAllBtn.addEventListener('click', fetchAllAudio);
if (audioFilter) audioFilter.addEventListener('change', renderAllAudio);

// Play All: build playlist from filter and sequentially play using audio element
if (playAllBtn) {
  playAllBtn.addEventListener('click', () => {
    const filter = (audioFilter && audioFilter.value) || 'all';
    playlist = allAudioFiles.filter(f => filter === 'all' ? true : f.type === filter).map(f => f.url);
    if (!playlist.length) return alert('Nothing to play');
    playlistIndex = 0;
    // stop any reply playback
    replyQueue = [];
    isPlayingReply = false;
    // start playback via player element
    player.src = playlist[0];
    player.play().catch(() => {});
    playlistIndex = 1;
    if (status) status.textContent = `Playing playlist 1 / ${playlist.length}`;
  });
}

// advance playlist when player ends
if (player) player.addEventListener('ended', () => {
  if (playlist && playlistIndex < (playlist.length || 0)) {
    player.src = playlist[playlistIndex];
    player.play().catch(() => {});
    playlistIndex++;
    if (status) status.textContent = `Playing playlist ${playlistIndex} / ${playlist.length}`;
  }
});

// Delete selected (bulk) - for convenience deletes all currently filtered files
if (deleteAllAudioBtn) {
  deleteAllAudioBtn.addEventListener('click', async () => {
    if (!confirm('Delete all currently listed files? This cannot be undone.')) return;
    const filter = (audioFilter && audioFilter.value) || 'all';
    const filesToDelete = allAudioFiles.filter(f => filter === 'all' ? true : f.type === filter).map(f => f.filename);
    for (const fn of filesToDelete) {
      try {
        await fetch('/delete-file/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: fn })
        });
      } catch (err) {
        console.error('bulk delete failed for', fn, err);
      }
    }
    await fetchAllAudio();
    fetchRecordings();
  });
}
