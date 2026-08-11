/* ── App State ────────────────────────────────────────────── */

const state = {
    project: null,
    videoPath: null,
    keyframes: [],
    mode: 'double',       // 'single' or 'double'
    currentView: 'home',
    reviewView: 'grid',   // 'grid', 'quad', 'single'
    singleIdx: 0,
    quadStart: 0,
    labels: {},           // frame_index -> label
    scrubberFrame: 0,
};


/* ── Navigation ──────────────────────────────────────────── */

function navigate(view, opts = {}) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');
    state.currentView = view;

    const crumbs = document.getElementById('nav-breadcrumbs');
    const projLabel = document.getElementById('nav-project');
    projLabel.textContent = state.project || '';

    const crumbMap = {
        home: '',
        processing: '<span>Processing</span>',
        review: '<span>Review</span>',
        crop: '<span>Crop & Split</span>',
        export: '<span>Export</span>',
    };
    crumbs.innerHTML = crumbMap[view] || '';

    if (view === 'home') loadHome();
}


/* ── Home View ───────────────────────────────────────────── */

async function loadHome() {
    // Load recordings
    const recList = document.getElementById('recording-list');
    try {
        const recs = await api('/api/recordings');
        if (recs.length === 0) {
            recList.innerHTML = '<p class="muted">No recordings found in recordings/</p>';
        } else {
            recList.innerHTML = recs.map(r => `
                <div class="file-item" onclick="startProject('${r.path}', '${r.name}')">
                    <span class="file-name">${r.name}</span>
                    <span class="file-meta">${r.size_mb} MB</span>
                </div>
            `).join('');
        }
    } catch (e) {
        recList.innerHTML = '<p class="muted">Error loading recordings</p>';
    }

    // Load projects
    const projList = document.getElementById('project-list');
    try {
        const projects = await api('/api/projects');
        if (projects.length === 0) {
            projList.innerHTML = '<p class="muted">No projects yet</p>';
        } else {
            projList.innerHTML = projects.map(p => `
                <div class="project-card" onclick="openProject('${p.name}')">
                    <div class="project-name">${p.name}</div>
                    <div class="project-status">
                        <div class="status-dot ${p.has_motion ? 'done' : ''}" title="Motion"></div>
                        <div class="status-dot ${p.has_peaks ? 'done' : ''}" title="Peaks"></div>
                        <div class="status-dot ${p.has_keyframes ? 'done' : ''}" title="Keyframes"></div>
                        <div class="status-dot ${p.has_pages ? 'done' : ''}" title="Pages"></div>
                        <div class="status-dot ${p.has_pdf ? 'done' : ''}" title="PDF"></div>
                    </div>
                    <div class="project-meta">
                        ${p.keyframe_count ? p.keyframe_count + ' keyframes' : ''}
                        ${p.page_count ? ' · ' + p.page_count + ' pages' : ''}
                    </div>
                </div>
            `).join('');
        }
    } catch (e) {
        projList.innerHTML = '<p class="muted">Error loading projects</p>';
    }
}


/* ── Start/Open Project ──────────────────────────────────── */

async function startProject(videoPath, videoName) {
    state.videoPath = videoPath;
    state.project = videoName.replace(/\.[^.]+$/, '');
    navigate('processing');
    await runMotionAndPeaks();
}

async function openProject(name) {
    state.project = name;
    // Find the video path
    const recs = await api('/api/recordings');
    const match = recs.find(r => r.name.startsWith(name));
    state.videoPath = match ? match.path : null;

    // Determine where to resume
    const projects = await api('/api/projects');
    const proj = projects.find(p => p.name === name);

    if (proj && proj.has_keyframes) {
        state.keyframes = await api(`/api/keyframes/${name}`);
        navigate('review');
        loadReview();
    } else if (proj && proj.has_peaks) {
        navigate('processing');
        showProcessingDone('Peaks already detected. Ready to select keyframes.');
    } else {
        navigate('processing');
        if (state.videoPath) {
            await runMotionAndPeaks();
        } else {
            document.getElementById('proc-status').textContent = 'Video file not found for this project.';
        }
    }
}


/* ── Processing ──────────────────────────────────────────── */

async function runMotionAndPeaks() {
    const title = document.getElementById('proc-title');
    const bar = document.getElementById('proc-bar');
    const status = document.getElementById('proc-status');
    const plot = document.getElementById('proc-plot');
    const actions = document.getElementById('proc-actions');

    // Phase 1: Motion
    title.textContent = 'Phase 1: Computing Motion Signal...';
    status.textContent = 'Reading video frames...';
    bar.style.width = '0%';
    plot.innerHTML = '';
    actions.style.display = 'none';

    const motionResult = await api('/api/process/motion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: state.videoPath }),
    });

    // Poll progress
    await pollProgress(motionResult.task_id, bar, status);

    // Show motion plot
    plot.innerHTML = `<img src="/plots/${state.project}/motion_plot.png?t=${Date.now()}">`;

    // Phase 2: Peaks
    title.textContent = 'Phase 2: Detecting Page Turns...';
    status.textContent = 'Finding peaks...';
    bar.style.width = '0%';

    const peakResult = await api('/api/process/peaks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: state.project }),
    });

    bar.style.width = '100%';
    status.textContent = `Found ${peakResult.peaks} page turns → ${peakResult.spreads} spreads (median ${peakResult.median_duration}s)`;
    plot.innerHTML = `<img src="/plots/${state.project}/peaks_plot.png?t=${Date.now()}">`;

    showProcessingDone('Ready to select keyframes.');
}

function showProcessingDone(msg) {
    const status = document.getElementById('proc-status');
    const actions = document.getElementById('proc-actions');
    const bar = document.getElementById('proc-bar');

    bar.style.width = '100%';
    status.textContent = msg;
    actions.style.display = 'block';

    document.getElementById('proc-next').onclick = async () => {
        await runKeyframes();
    };
}

async function runKeyframes() {
    const title = document.getElementById('proc-title');
    const bar = document.getElementById('proc-bar');
    const status = document.getElementById('proc-status');
    const actions = document.getElementById('proc-actions');

    title.textContent = 'Phase 3: Selecting Keyframes...';
    status.textContent = 'Extracting best frames...';
    bar.style.width = '0%';
    actions.style.display = 'none';

    const result = await api('/api/process/keyframes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: state.project, video_path: state.videoPath }),
    });

    await pollProgress(result.task_id, bar, status);

    // Load keyframes and go to review
    state.keyframes = await api(`/api/keyframes/${state.project}`);
    navigate('review');
    loadReview();
}


/* ── Review View ─────────────────────────────────────────── */

async function loadReview() {
    // Restore saved labels from server
    try {
        const labels = await api(`/api/keyframes/${state.project}/labels`);
        state.labels = labels || {};
    } catch (e) {
        state.labels = {};
    }

    updateReviewStats();
    toggleView(state.reviewView);

    // Set mode buttons
    document.getElementById('mode-double').classList.toggle('active', state.mode === 'double');
    document.getElementById('mode-single').classList.toggle('active', state.mode === 'single');
}

function setMode(mode) {
    state.mode = mode;
    document.getElementById('mode-double').classList.toggle('active', mode === 'double');
    document.getElementById('mode-single').classList.toggle('active', mode === 'single');
}

function toggleView(view) {
    state.reviewView = view;
    document.getElementById('review-grid').style.display = view === 'grid' ? '' : 'none';
    document.getElementById('review-quad').style.display = view === 'quad' ? '' : 'none';
    document.getElementById('review-single').style.display = view === 'single' ? '' : 'none';

    if (view === 'grid') renderGrid();
    else if (view === 'quad') renderQuad();
    else if (view === 'single') renderSingle();

    // Update button states
    document.querySelectorAll('.review-actions .btn').forEach(b => b.classList.remove('active'));
}

function updateReviewStats() {
    const total = state.keyframes.length;
    const labeled = Object.keys(state.labels).length;
    const dels = Object.values(state.labels).filter(l => ['dup', 'occlusion', 'other'].includes(l)).length;
    document.getElementById('review-stats').textContent =
        `${total} keyframes · ${dels} marked for deletion · ${labeled} reviewed`;
}

// Grid
function updateGridSize(val) {
    document.getElementById('grid-size-val').textContent = val;
    const grid = document.getElementById('review-grid');
    grid.style.gridTemplateColumns = `repeat(${val}, 1fr)`;
}

function renderGrid() {
    const grid = document.getElementById('review-grid');
    grid.innerHTML = state.keyframes.map((kf, i) => {
        const label = state.labels[kf.frame_index];
        const labelHtml = label ? `<div class="grid-label" style="background:${labelColor(label)}">${label}</div>` : '';
        return `
            <div class="grid-thumb ${i === state.singleIdx ? 'selected' : ''}"
                 onclick="state.singleIdx=${i}; toggleView('single')">
                <img src="/images/${state.project}/${kf.filename}" loading="lazy">
                ${labelHtml}
            </div>`;
    }).join('');
}

// Quad
function renderQuad() {
    const container = document.getElementById('quad-images');
    const start = state.quadStart;
    const batch = state.keyframes.slice(start, start + 4);

    container.innerHTML = batch.map((kf, i) => {
        const globalIdx = start + i;
        return `
            <div class="quad-cell ${globalIdx === state.singleIdx ? 'selected' : ''}"
                 onclick="state.singleIdx=${globalIdx}; toggleView('single')">
                <img src="/images/${state.project}/${kf.filename}">
            </div>`;
    }).join('');

    document.getElementById('quad-counter').textContent =
        `${start + 1}–${Math.min(start + 4, state.keyframes.length)} of ${state.keyframes.length}`;
}

function quadPrev() { state.quadStart = Math.max(0, state.quadStart - 4); renderQuad(); }
function quadNext() {
    if (state.quadStart + 4 < state.keyframes.length) { state.quadStart += 4; renderQuad(); }
}

// Single
let gutterDragging = false;
let gutterEditing = false;

function getEffectiveGutter(idx) {
    // If this frame has its own gutter, use it
    const kf = state.keyframes[idx];
    if (kf.gutter_pct != null) return kf.gutter_pct;
    // Walk backward to find the most recent frame with a gutter set
    for (let i = idx - 1; i >= 0; i--) {
        if (state.keyframes[i].gutter_pct != null) return state.keyframes[i].gutter_pct;
    }
    return 0.5; // default
}

function renderSingle() {
    const kf = state.keyframes[state.singleIdx];
    if (!kf) return;

    const img = new Image();
    img.onload = () => {
        const canvas = document.getElementById('review-canvas');
        const container = canvas.parentElement;
        const maxW = container.clientWidth;
        const maxH = container.clientHeight;
        const scale = Math.min(maxW / img.width, maxH / img.height, 1);

        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        canvas._imgScale = scale;
        canvas._imgW = img.width;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw gutter line if double mode
        if (state.mode === 'double') {
            const gutter = getEffectiveGutter(state.singleIdx);
            const isOwn = kf.gutter_pct != null;
            const x = canvas.width * gutter;
            ctx.strokeStyle = gutterEditing ? '#00ffff' : (isOwn ? '#ff3333' : '#ff333388');
            ctx.lineWidth = gutterEditing ? 3 : 1;
            ctx.setLineDash(gutterEditing ? [] : [6, 4]);
            ctx.beginPath();
            ctx.moveTo(x, canvas.height * 0.02);
            ctx.lineTo(x, canvas.height * 0.98);
            ctx.stroke();
            ctx.setLineDash([]);

            // Gutter label
            const source = isOwn ? 'set here' : 'inherited';
            if (gutterEditing) {
                ctx.fillStyle = '#00ffff';
                ctx.font = '12px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`Gutter: ${(gutter * 100).toFixed(1)}%`, x, canvas.height * 0.02 - 4);
                ctx.fillText('Click to place · ←/→ fine-tune · Enter to confirm · Esc to cancel', canvas.width / 2, canvas.height - 8);
            } else if (state.mode === 'double') {
                ctx.fillStyle = isOwn ? '#ff3333' : '#ff333388';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`${(gutter * 100).toFixed(1)}% (${source})`, x, canvas.height * 0.02 - 4);
            }
        }
    };
    img.src = `/images/${state.project}/${kf.filename}`;

    const label = state.labels[kf.frame_index];
    document.getElementById('single-info').textContent =
        `Frame ${kf.frame_index} · ${kf.time_sec}s · Motion: ${kf.motion_value} · ` +
        (label ? label.toUpperCase() : 'unlabeled') +
        (gutterEditing ? ' · GUTTER EDIT MODE' : '');
    document.getElementById('single-counter').textContent =
        `${state.singleIdx + 1} / ${state.keyframes.length}`;
}

// Gutter click/drag on canvas
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('review-canvas');
    if (!canvas) return;

    canvas.addEventListener('click', (e) => {
        if (state.mode !== 'double' || state.reviewView !== 'single') return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = x / canvas.width;
        if (pct > 0.1 && pct < 0.9) {
            const kf = state.keyframes[state.singleIdx];
            kf.gutter_pct = Math.round(pct * 1000) / 1000;
            gutterEditing = true;
            renderSingle();
        }
    });

    canvas.addEventListener('mousedown', (e) => {
        if (state.mode !== 'double' || state.reviewView !== 'single') return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const kf = state.keyframes[state.singleIdx];
        const gutterX = canvas.width * (kf.gutter_pct || 0.5);
        if (Math.abs(x - gutterX) < 20) {
            gutterDragging = true;
            gutterEditing = true;
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!gutterDragging) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = Math.max(0.1, Math.min(0.9, x / canvas.width));
        const kf = state.keyframes[state.singleIdx];
        kf.gutter_pct = Math.round(pct * 1000) / 1000;
        renderSingle();
    });

    document.addEventListener('mouseup', () => {
        gutterDragging = false;
    });

    // Set cursor when near gutter
    canvas.addEventListener('mousemove', (e) => {
        if (state.mode !== 'double' || state.reviewView !== 'single') return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const kf = state.keyframes[state.singleIdx];
        const gutterX = canvas.width * (kf.gutter_pct || 0.5);
        canvas.style.cursor = Math.abs(x - gutterX) < 20 ? 'col-resize' : 'crosshair';
    });
});

function singlePrev() { if (state.singleIdx > 0) { state.singleIdx--; renderSingle(); } }
function singleNext() { if (state.singleIdx < state.keyframes.length - 1) { state.singleIdx++; renderSingle(); } }

function labelFrame(label) {
    const kf = state.keyframes[state.singleIdx];
    if (state.labels[kf.frame_index] === label) {
        delete state.labels[kf.frame_index];
    } else {
        state.labels[kf.frame_index] = label;
    }
    updateReviewStats();
    renderSingle();
    saveLabelsToServer();
}

async function saveLabelsToServer() {
    await api(`/api/keyframes/${state.project}/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.labels),
    });
}

async function loadLabelsFromServer() {
    try {
        const labels = await api(`/api/keyframes/${state.project}/labels`);
        state.labels = labels || {};
    } catch (e) {
        state.labels = {};
    }
}

function labelColor(label) {
    const colors = { keep: '#22c55e', dup: '#f59e0b', occlusion: '#ec4899',
                     other: '#94a3b8', cover: '#3b82f6', doc_start: '#a855f7' };
    return colors[label] || '#666';
}


/* ── Video Scrubber ──────────────────────────────────────── */

function openScrubber() {
    const kf = state.keyframes[state.singleIdx];
    state.scrubberFrame = kf.frame_index;
    document.getElementById('scrubber-modal').style.display = 'flex';
    updateScrubber();
}

function closeScrubber() {
    document.getElementById('scrubber-modal').style.display = 'none';
}

function scrubStep(delta) {
    state.scrubberFrame = Math.max(0, state.scrubberFrame + delta);
    updateScrubber();
}

function updateScrubber() {
    const img = document.getElementById('scrubber-img');
    img.src = `/api/video-frame?video=${encodeURIComponent(state.videoPath)}&frame=${state.scrubberFrame}&t=${Date.now()}`;
    document.getElementById('scrubber-info').textContent = `Frame ${state.scrubberFrame}`;
}

async function grabFrame() {
    await api(`/api/keyframes/${state.project}/insert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: state.scrubberFrame, video_path: state.videoPath }),
    });
    state.keyframes = await api(`/api/keyframes/${state.project}`);
    closeScrubber();
    updateReviewStats();
    renderSingle();
}


/* ── Save Review ─────────────────────────────────────────── */

async function saveReview() {
    // Apply deletions
    const toDelete = Object.entries(state.labels)
        .filter(([_, label]) => ['dup', 'occlusion', 'other'].includes(label));

    for (const [frameIdx, reason] of toDelete) {
        await api(`/api/keyframes/${state.project}/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_index: parseInt(frameIdx), reason }),
        });
    }

    // Apply cover/doc_start flags
    const toUpdate = Object.entries(state.labels)
        .filter(([_, label]) => ['cover', 'doc_start'].includes(label));

    for (const [frameIdx, label] of toUpdate) {
        const updates = {};
        if (label === 'cover') updates.is_cover = true;
        if (label === 'doc_start') updates.is_doc_start = true;
        await api(`/api/keyframes/${state.project}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_index: parseInt(frameIdx), ...updates }),
        });
    }

    // Reload and move to crop
    state.keyframes = await api(`/api/keyframes/${state.project}`);
    state.labels = {};
    navigate('crop');
    loadCrop();
}


/* ── Crop & Split ────────────────────────────────────────── */

let cropIdx = 0;

function loadCrop() {
    cropIdx = 0;
    renderCrop();
}

function renderCrop() {
    const kf = state.keyframes[cropIdx];
    if (!kf) return;

    document.getElementById('crop-original-img').src =
        `/images/${state.project}/${kf.filename}?t=${Date.now()}`;
    document.getElementById('crop-result-img').src =
        `/api/crop-preview/${state.project}/${kf.filename}?mode=${state.mode}&t=${Date.now()}`;
    document.getElementById('crop-counter').textContent =
        `${cropIdx + 1} / ${state.keyframes.length}`;
}

function cropPrev() { if (cropIdx > 0) { cropIdx--; renderCrop(); } }
function cropNext() { if (cropIdx < state.keyframes.length - 1) { cropIdx++; renderCrop(); } }

async function applyCropAll() {
    const result = await api('/api/process/split', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: state.project, mode: state.mode }),
    });

    navigate('export');
    document.getElementById('export-status').textContent =
        `${result.pages} pages ready.`;
    loadExportThumbs();
}


/* ── Export ───────────────────────────────────────────────── */

async function loadExportThumbs() {
    try {
        const pages = await api(`/api/pages/${state.project}`);
        const container = document.getElementById('export-thumbs');
        container.innerHTML = pages.map(pg => `
            <div class="export-thumb">
                <img src="/pages/${state.project}/${pg.filename}" loading="lazy"
                     title="Page ${pg.page_num} (${pg.type})">
            </div>
        `).join('');
    } catch (e) {
        document.getElementById('export-thumbs').innerHTML =
            '<p class="muted">Could not load page thumbnails</p>';
    }
}

async function buildPdf() {
    const bw = document.getElementById('export-bw').checked;
    const status = document.getElementById('export-status');
    status.textContent = 'Building PDF...';

    const result = await api('/api/process/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: state.project, bw }),
    });

    status.textContent = `PDF created with ${result.pages} pages! ✓`;
}


/* ── Keyboard shortcuts ──────────────────────────────────── */

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Crop view navigation
    if (state.currentView === 'crop') {
        switch (e.key) {
            case 'ArrowRight': case 'd': cropNext(); e.preventDefault(); break;
            case 'ArrowLeft': case 'a': cropPrev(); e.preventDefault(); break;
        }
        return;
    }

    // Review view
    if (state.currentView !== 'review' || state.reviewView !== 'single') return;

    const scrubberOpen = document.getElementById('scrubber-modal').style.display === 'flex';

    if (scrubberOpen) {
        switch (e.key) {
            case 'ArrowRight': scrubStep(1); e.preventDefault(); break;
            case 'ArrowLeft': scrubStep(-1); e.preventDefault(); break;
            case 'ArrowUp': scrubStep(5); e.preventDefault(); break;
            case 'ArrowDown': scrubStep(-5); e.preventDefault(); break;
            case 'Enter': grabFrame(); e.preventDefault(); break;
            case 'Escape': closeScrubber(); e.preventDefault(); break;
        }
        if (e.shiftKey && e.key === 'ArrowRight') { scrubStep(30); e.preventDefault(); }
        if (e.shiftKey && e.key === 'ArrowLeft') { scrubStep(-30); e.preventDefault(); }
        return;
    }

    // Gutter editing mode
    if (gutterEditing) {
        const kf = state.keyframes[state.singleIdx];
        const step = e.shiftKey ? 0.001 : 0.005;  // shift = 0.1%, normal = 0.5%
        switch (e.key) {
            case 'ArrowRight':
                kf.gutter_pct = Math.min(0.9, (kf.gutter_pct || 0.5) + step);
                renderSingle(); e.preventDefault(); break;
            case 'ArrowLeft':
                kf.gutter_pct = Math.max(0.1, (kf.gutter_pct || 0.5) - step);
                renderSingle(); e.preventDefault(); break;
            case 'Enter':
                // Save gutter to server
                api(`/api/keyframes/${state.project}/update`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame_index: kf.frame_index, gutter_pct: kf.gutter_pct }),
                });
                gutterEditing = false;
                renderSingle(); e.preventDefault(); break;
            case 'Escape':
                gutterEditing = false;
                renderSingle(); e.preventDefault(); break;
        }
        return;
    }

    switch (e.key) {
        case 'ArrowRight': case 'd': singleNext(); e.preventDefault(); break;
        case 'ArrowLeft': case 'a': singlePrev(); e.preventDefault(); break;
        case '1': labelFrame('keep'); break;
        case '2': labelFrame('dup'); break;
        case '3': labelFrame('occlusion'); break;
        case '4': labelFrame('other'); break;
        case '5': labelFrame('cover'); break;
        case '6': labelFrame('doc_start'); break;
        case 'i': case 'I': openScrubber(); break;
        case 'g': case 'G':
            if (state.mode === 'double') {
                gutterEditing = !gutterEditing;
                renderSingle();
            }
            break;
    }
});


/* ── Utils ───────────────────────────────────────────────── */

async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

async function pollProgress(taskId, bar, status) {
    while (true) {
        await new Promise(r => setTimeout(r, 500));
        const prog = await api(`/api/progress/${taskId}`);
        if (prog.status === 'done') {
            bar.style.width = '100%';
            status.textContent = 'Done!';
            return prog;
        } else if (prog.status === 'error') {
            status.textContent = `Error: ${prog.error}`;
            throw new Error(prog.error);
        } else {
            bar.style.width = `${prog.progress}%`;
            status.textContent = `Processing... ${prog.progress}%`;
        }
    }
}


/* ── Init ────────────────────────────────────────────────── */

navigate('home');