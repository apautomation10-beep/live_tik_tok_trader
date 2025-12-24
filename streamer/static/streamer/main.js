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
    if (!confirm('Are you sure you want to delete all audio files (commentary and replies)? This cannot be undone.')) return;
    deleteAllBtn.disabled = true;
    deleteAllBtn.textContent = 'Deleting...';

    try {
      const res = await fetch('/delete-recordings/', { method: 'POST' });
      const data = await res.json();
      status.textContent = `Deleted ${data.deleted} files`;
      // refresh the listing after deletion
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

// Polling and reply interruption handling
let replyQueue = [];
let isPlayingReply = false;
let pausedComment = null; // { index, time }

async function pollForReplies() {
  // only poll while a live session is running
  if (goLiveBtn.disabled !== true) return;
  try {
    const res = await fetch('/replies/next/');
    if (!res.ok) return;
    const data = await res.json();
    if (data.found && data.url) {
      // enqueue reply
      replyQueue.push(data.url);
      status.textContent = `Queued live reply (${replyQueue.length} pending)`;
      // if we are not currently playing a reply, start the reply playback flow
      if (!isPlayingReply) {
        startReplyPlayback();
      }
    }
  } catch (err) {
    console.error('Reply poll failed', err);
  }
}

async function startReplyPlayback() {
  if (!replyQueue.length) return;
  // Pause commentary and remember where we were
  if (!isPlayingReply) {
    // if the player is currently playing commentary, save index and currentTime
    if (!player.paused) {
      const currentPlayingIndex = Math.max(0, currentIndex - 1);
      pausedComment = { index: currentPlayingIndex, time: player.currentTime || 0 };
      player.pause();
    } else {
      pausedComment = null;
    }
  }

  isPlayingReply = true;

  // Play all queued replies in order
  while (replyQueue.length) {
    const url = replyQueue.shift();
    status.textContent = `Playing live reply (${replyQueue.length} remaining)`;
    await playReplyOnce(url);
  }

  // Finished replies, resume commentary exactly where we paused
  await resumeCommentary();
  isPlayingReply = false;
}

function playReplyOnce(url) {
  return new Promise((resolve, reject) => {
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
  });
}

function resumeCommentary() {
  return new Promise((resolve, reject) => {
    status.textContent = 'Resuming live commentary';

    if (!pausedComment) {
      // nothing to resume, just ensure main playback continues
      if (player.src && player.paused) player.play();
      return resolve();
    }

    const resumeIndex = pausedComment.index;
    const resumeTime = pausedComment.time || 0;

    // Set the player to the paused segment and seek to saved time
    player.src = audioQueue[resumeIndex];

    // Update currentIndex to reflect that this segment is now playing
    currentIndex = resumeIndex + 1;

    const onCanSeek = () => {
      try {
        // Some browsers require a small delay before setting currentTime
        player.currentTime = Math.max(0, Math.min(resumeTime, player.duration || resumeTime));
      } catch (e) {
        // ignore; will start from nearest possible position
        console.warn('Failed to set resume currentTime', e);
      }
      player.play().then(() => {
        status.textContent = `Playing segment ${currentIndex} / ${audioQueue.length}`;
        player.removeEventListener('canplay', onCanSeek);
        resolve();
      }).catch((e) => {
        console.error('Resume play failed', e);
        player.removeEventListener('canplay', onCanSeek);
        resolve();
      });
    };

    // Attach handler and try to trigger metadata load if needed
    player.addEventListener('canplay', onCanSeek);
    // In case canplay already fired, try to call directly after a tick
    setTimeout(() => {
      if (player.readyState >= 2) onCanSeek();
    }, 200);
  });
}

setInterval(pollForReplies, 2000);

// Load recordings on page load
window.addEventListener('load', fetchRecordings);
