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
    history: [],
};

const VIEW_ORDER = ['home', 'processing', 'review', 'crop', 'export'];


/* ── Navigation ──────────────────────────────────────────── */

function navigate(view, opts = {}) {
    if (state.currentView && state.currentView !== view && !opts.isBack) {
        state.history.push(state.currentView);
    }
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');
    state.currentView = view;

    // Back button visibility
    const backBtn = document.getElementById('nav-back');
    backBtn.style.display = (view === 'home' || state.history.length === 0) ? 'none' : '';
    const anaBtn = document.getElementById('nav-analysis');
    anaBtn.style.display = (state.project && view !== 'analysis' && view !== 'home') ? '' : 'none';

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


function goBack() {
    if (state.history.length === 0) return;
    const prev = state.history.pop();
    navigate(prev, { isBack: true });

    // Reload data for the view we're returning to
    if (prev === 'review') loadReview();
    else if (prev === 'crop') loadCrop();
    else if (prev === 'export') loadExportThumbs();
    else if (prev === 'analysis') loadAnalysis();
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
    plot.innerHTML = themedPlot(state.project, 'motion_plot');

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
    plot.innerHTML = themedPlot(state.project, 'peaks_plot');

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

// Keyboard: move selection within the visible 4, page when hitting the edge
function quadSelectPrev() {
    if (state.singleIdx > 0) {
        state.singleIdx--;
        if (state.singleIdx < state.quadStart) state.quadStart = Math.max(0, state.quadStart - 4);
        renderQuad();
    }
}
function quadSelectNext() {
    if (state.singleIdx < state.keyframes.length - 1) {
        state.singleIdx++;
        if (state.singleIdx >= state.quadStart + 4) state.quadStart += 4;
        renderQuad();
    }
}

// Single
let gutterDragging = null;   // 'top' | 'bot' | 'place' while dragging
let gutterEditing = false;
let boxEditing = false;      // crop-box edit mode in the review view
let boxDrag = null;          // {kind:'corner'|'edge'|'move', i}

function setReviewQuad(q) {
    state.keyframes[state.singleIdx].crop_quad =
        q.map(p => [+p[0].toFixed(5), +p[1].toFixed(5)]);
    renderSingle();
}

async function saveReviewQuad() {
    const kf = state.keyframes[state.singleIdx];
    if (!kf.crop_quad) return;
    await api(`/api/keyframes/${state.project}/update`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: kf.frame_index,
                               crop_quad: kf.crop_quad }),
    });
    renderSingle();
}

function reviewBoxHit(mx, my, canvas) {
    const P = reviewQuad(state.singleIdx)
        .map(([x, y]) => [x * canvas.width, y * canvas.height]);
    for (let i = 0; i < 4; i++)
        if (Math.hypot(mx - P[i][0], my - P[i][1]) < 12) return { kind: 'corner', i };
    for (let i = 0; i < 4; i++) {
        const m = lerp(P[i], P[(i + 1) % 4], 0.5);
        if (Math.hypot(mx - m[0], my - m[1]) < 12) return { kind: 'edge', i };
    }
    let inside = false;
    for (let i = 0, j = 3; i < 4; j = i++)
        if ((P[i][1] > my) !== (P[j][1] > my) &&
            mx < (P[j][0] - P[i][0]) * (my - P[i][1]) / (P[j][1] - P[i][1]) + P[i][0])
            inside = !inside;
    return inside ? { kind: 'move' } : null;
}

/* The review view and the crop view now share one gutter model:
   {top, bot} — the fraction along the box's top edge and along its bottom
   edge. Two fractions carry position *and* tilt, which matters because an
   open book fans: the spine is not parallel to the outer page edges. */

const reviewAutoQuad = {};      // idx -> detector's box, fetched lazily

function reviewQuad(idx) {
    // The gutter is stored as a fraction *of this box*, so review and crop
    // must resolve the same one — otherwise the same number lands in a
    // different place once the box changes, and the work doesn't transfer.
    const kf = state.keyframes[idx];
    if (kf.crop_quad) return kf.crop_quad;
    for (let i = idx - 1; i >= 0; i--)
        if (state.keyframes[i].crop_quad) return state.keyframes[i].crop_quad;
    if (reviewAutoQuad[idx]) return reviewAutoQuad[idx];
    fetchReviewAuto(idx);
    return [[0, 0], [1, 0], [1, 1], [0, 1]];
}

function reviewQuadSource(idx) {
    const kf = state.keyframes[idx];
    if (kf.crop_quad) return 'set here';
    for (let i = idx - 1; i >= 0; i--)
        if (state.keyframes[i].crop_quad) return 'inherited';
    return reviewAutoQuad[idx] ? 'auto' : 'detecting…';
}

async function fetchReviewAuto(idx) {
    if (reviewAutoQuad[idx] === 'pending') return;
    reviewAutoQuad[idx] = 'pending';
    try {
        const kf = state.keyframes[idx];
        const r = await api(
            `/api/crop-rect/${state.project}/${kf.filename}?mode=${state.mode}`);
        reviewAutoQuad[idx] = r.quad;
        if (state.currentView === 'review' && state.singleIdx === idx) renderSingle();
    } catch (e) { delete reviewAutoQuad[idx]; }
}

function getEffectiveGutter(idx) {
    const norm = g => (g == null) ? null
        : (typeof g === 'number') ? { top: g, bot: g }
        : ('top' in g) ? g : { top: g.pos, bot: g.pos };
    const own = norm(state.keyframes[idx].gutter);
    if (own) return { g: own, own: true };
    for (let i = idx - 1; i >= 0; i--) {
        const g = norm(state.keyframes[i].gutter);
        if (g) return { g, own: false };
    }
    return { g: { top: 0.5, bot: 0.5 }, own: false, isDefault: true };
}

function renderSingle() {
    const kf = state.keyframes[state.singleIdx];
    if (!kf) return;

    const img = new Image();
    img.onload = () => {
        const canvas = document.getElementById('review-canvas');
        const container = canvas.parentElement;
        const scale = Math.min(container.clientWidth / img.width,
                               container.clientHeight / img.height, 1);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const q = reviewQuad(state.singleIdx);

        // Crop box: dimmed outside, lit inside — the same treatment the
        // crop view uses, because it is now the same box.
        if (Array.isArray(q) && !(q[0][0] === 0 && q[2][0] === 1 && q[0][1] === 0)) {
            const P = q.map(([x, y]) => [x * canvas.width, y * canvas.height]);
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, canvas.width, canvas.height);
            ctx.moveTo(P[0][0], P[0][1]);
            for (let i = 3; i >= 1; i--) ctx.lineTo(P[i][0], P[i][1]);
            ctx.closePath();
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fill('evenodd');
            ctx.restore();

            const ownBox = reviewQuadSource(state.singleIdx) === 'set here';
            ctx.strokeStyle = boxEditing ? '#00ffff' : '#4361ee';
            ctx.lineWidth = boxEditing ? 2 : 1.5;
            ctx.setLineDash(ownBox || boxEditing ? [] : [5, 4]);
            ctx.beginPath();
            ctx.moveTo(P[0][0], P[0][1]);
            for (let i = 1; i < 4; i++) ctx.lineTo(P[i][0], P[i][1]);
            ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
            if (boxEditing) {
                ctx.fillStyle = '#00ffff';
                for (const [hx, hy] of P) ctx.fillRect(hx - 4, hy - 4, 8, 8);
                for (let i = 0; i < 4; i++) {
                    const m = lerp(P[i], P[(i + 1) % 4], 0.5);
                    ctx.fillRect(m[0] - 3, m[1] - 3, 6, 6);
                }
            }
        }

        if (state.mode === 'double') {
            const { g, own, isDefault } = getEffectiveGutter(state.singleIdx);
            const [a, b] = gutterEnds(q, g);
            const A = [a[0] * canvas.width, a[1] * canvas.height];
            const B = [b[0] * canvas.width, b[1] * canvas.height];

            ctx.strokeStyle = gutterEditing ? '#00ffff'
                            : own ? '#ff3333' : '#ff333388';
            ctx.lineWidth = gutterEditing ? 3 : 1.5;
            ctx.setLineDash(gutterEditing ? [] : [6, 4]);
            ctx.beginPath();
            ctx.moveTo(A[0], A[1]);
            ctx.lineTo(B[0], B[1]);
            ctx.stroke();
            ctx.setLineDash([]);

            if (gutterEditing) {
                // Endpoint handles: dragging one sets position and tilt in a
                // single gesture — the thing a bare fraction cannot express.
                ctx.fillStyle = '#00ffff';
                for (const P of [A, B]) {
                    ctx.beginPath();
                    ctx.arc(P[0], P[1], 6, 0, Math.PI * 2);
                    ctx.fill();
                }
                const tilt = gutterTilt(q, g);
                ctx.font = '12px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`gutter ${(g.top * 100).toFixed(1)}%` +
                             (Math.abs(tilt) > 0.05 ? `  tilt ${tilt.toFixed(1)}°` : ''),
                             A[0], A[1] - 10);
                ctx.fillText('click to place · drag ends to tilt · Enter save · Esc cancel',
                             canvas.width / 2, canvas.height - 8);
            } else {
                ctx.fillStyle = own ? '#ff3333' : '#ff333388';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(
                    `${(g.top * 100).toFixed(1)}% (${own ? 'set here' : isDefault ? 'default' : 'inherited'})`,
                    A[0], A[1] - 6);
            }
        }
    };
    img.src = `/images/${state.project}/${kf.filename}`;

    const label = state.labels[kf.frame_index];
    document.getElementById('single-info').textContent =
        `Frame ${kf.frame_index} · ${kf.time_sec}s · Motion: ${kf.motion_value} · ` +
        (label ? label.toUpperCase() : 'unlabeled') +
        (gutterEditing ? ' · GUTTER EDIT (G)' : boxEditing ? ' · BOX EDIT (B)' : '') +
        (state.mode === 'double' ? ` · box ${reviewQuadSource(state.singleIdx)}` : '');
    document.getElementById('single-counter').textContent =
        `${state.singleIdx + 1} / ${state.keyframes.length}`;
}

function gutterTilt(quad, g) {
    const [a, b] = gutterEnds(quad, g);
    const [tl, tr, br, bl] = quad;
    const vb = [(bl[0] + br[0]) / 2 - (tl[0] + tr[0]) / 2,
                (bl[1] + br[1]) / 2 - (tl[1] + tr[1]) / 2];
    const vc = [b[0] - a[0], b[1] - a[1]];
    let d = (Math.atan2(vc[1], vc[0]) - Math.atan2(vb[1], vb[0])) * 180 / Math.PI;
    return ((d + 180) % 360 + 360) % 360 - 180;
}

function setReviewGutter(g) {
    const kf = state.keyframes[state.singleIdx];
    kf.gutter = { top: +g.top.toFixed(4), bot: +g.bot.toFixed(4) };
    renderSingle();
}

async function saveReviewGutter() {
    const kf = state.keyframes[state.singleIdx];
    if (!kf.gutter) return;
    await api(`/api/keyframes/${state.project}/update`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_index: kf.frame_index, gutter: kf.gutter }),
    });
    gutterEditing = false;
    renderSingle();
}

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('review-canvas');
    if (!canvas) return;

    const toFrac = (e) => {
        const r = canvas.getBoundingClientRect();
        return [(e.clientX - r.left) / canvas.width,
                (e.clientY - r.top) / canvas.height];
    };
    const endsPx = () => {
        const { g } = getEffectiveGutter(state.singleIdx);
        const [a, b] = gutterEnds(reviewQuad(state.singleIdx), g);
        return [[a[0] * canvas.width, a[1] * canvas.height],
                [b[0] * canvas.width, b[1] * canvas.height]];
    };

    canvas.addEventListener('mousedown', (e) => {
        if (state.reviewView !== 'single') return;
        const r = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;

        if (boxEditing) {
            boxDrag = reviewBoxHit(mx, my, canvas);
            if (boxDrag) boxDrag.last = [mx / canvas.width, my / canvas.height];
            return;
        }
        if (state.mode !== 'double') return;
        const [A, B] = endsPx();

        if (Math.hypot(mx - A[0], my - A[1]) < 14) { gutterDragging = 'top'; }
        else if (Math.hypot(mx - B[0], my - B[1]) < 14) { gutterDragging = 'bot'; }
        else {
            // A plain click means "the spine is here" — upright, both ends.
            const [fx] = toFrac(e);
            const q = reviewQuad(state.singleIdx);
            const t = fracOnEdge(q[0], q[1], [fx, 0]);
            const b = fracOnEdge(q[3], q[2], [fx, 0]);
            setReviewGutter({ top: t, bot: b });
            gutterDragging = 'place';
        }
        gutterEditing = true;
        renderSingle();
    });

    canvas.addEventListener('mousemove', (e) => {
        if (boxEditing) {
            const r = canvas.getBoundingClientRect();
            const mx = e.clientX - r.left, my = e.clientY - r.top;
            if (!boxDrag) {
                const h = reviewBoxHit(mx, my, canvas);
                canvas.style.cursor = !h ? 'default'
                    : h.kind === 'move' ? 'move'
                    : h.kind === 'corner' ? 'nwse-resize' : 'pointer';
                return;
            }
            const fx = mx / canvas.width, fy = my / canvas.height;
            const [lx, ly] = boxDrag.last;
            boxDrag.last = [fx, fy];
            const q = reviewQuad(state.singleIdx).map(p => [...p]);
            if (boxDrag.kind === 'move') {
                setReviewQuad(q.map(([x, y]) => [x + fx - lx, y + fy - ly]));
            } else if (boxDrag.kind === 'corner') {
                q[boxDrag.i] = [fx, fy];
                setReviewQuad(q);
            } else {
                const i = boxDrag.i, j = (i + 1) % 4;
                q[i] = [q[i][0] + fx - lx, q[i][1] + fy - ly];
                q[j] = [q[j][0] + fx - lx, q[j][1] + fy - ly];
                setReviewQuad(q);
            }
            return;
        }
        if (!gutterDragging) {
            if (state.mode === 'double' && state.reviewView === 'single') {
                const r = canvas.getBoundingClientRect();
                const mx = e.clientX - r.left, my = e.clientY - r.top;
                const [A, B] = endsPx();
                const near = Math.hypot(mx - A[0], my - A[1]) < 14 ||
                             Math.hypot(mx - B[0], my - B[1]) < 14;
                canvas.style.cursor = near ? 'col-resize' : 'crosshair';
            }
            return;
        }
        const [fx, fy] = toFrac(e);
        const q = reviewQuad(state.singleIdx);
        const { g } = getEffectiveGutter(state.singleIdx);
        if (gutterDragging === 'top') {
            setReviewGutter({ top: fracOnEdge(q[0], q[1], [fx, fy]), bot: g.bot });
        } else if (gutterDragging === 'bot') {
            setReviewGutter({ top: g.top, bot: fracOnEdge(q[3], q[2], [fx, fy]) });
        } else {
            setReviewGutter({ top: fracOnEdge(q[0], q[1], [fx, fy]),
                              bot: fracOnEdge(q[3], q[2], [fx, fy]) });
        }
    });

    document.addEventListener('mouseup', () => {
        if (gutterDragging) { gutterDragging = null; saveReviewGutter(); }
        if (boxDrag) { boxDrag = null; saveReviewQuad(); }
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
let cropEdit = false;
let cropQuad = null;          // 4 normalized corners [tl,tr,br,bl]
let cropGutter = null;        // {top, bot} fractions along the box edges
let cropAuto = null;
let cropSrc = '', gutterSrc = '';
let cropImg = new Image();
let cropDrag = null;
let cropPreviewTimer = null;

const HANDLE = 11;

function loadCrop() { cropIdx = 0; renderCrop(); }

/* ── geometry helpers (mirror engine/crop.py) ── */
const lerp = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];

function gutterEnds(quad, g) {
    const [tl, tr, br, bl] = quad;
    return [lerp(tl, tr, g.top), lerp(bl, br, g.bot)];
}

// Where along an edge does point p sit? Used to turn a drag into a fraction.
function fracOnEdge(a, b, p) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const l2 = dx * dx + dy * dy;
    if (l2 < 1e-9) return 0.5;
    const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2;
    return Math.max(0.02, Math.min(0.98, t));
}

async function renderCrop() {
    const kf = state.keyframes[cropIdx];
    if (!kf) return;
    document.getElementById('crop-counter').textContent =
        `${cropIdx + 1} / ${state.keyframes.length}`;

    try {
        cropAuto = await api(
            `/api/crop-rect/${state.project}/${kf.filename}?mode=${state.mode}`);
        cropQuad = (kf.crop_quad || cropAuto.effective_quad || cropAuto.quad)
                     .map(p => [p[0], p[1]]);
        cropGutter = { ...(kf.gutter || cropAuto.gutter || {top:0.5, bot:0.5}) };
        cropSrc = kf.crop_quad ? 'manual' : (cropAuto.quad_src || 'auto');
        gutterSrc = kf.gutter ? 'manual' : (cropAuto.gutter_src || 'auto');
        const extra = cropAuto.lines_used != null ? ` ${cropAuto.lines_used}/4` : '';
        document.getElementById('crop-source').textContent =
            state.mode === 'double'
                ? `box ${cropSrc} · gutter ${gutterSrc}`
                : `box ${cropSrc} (${cropAuto.source}${extra})`;
    } catch (e) { cropAuto = null; }

    cropImg = new Image();
    cropImg.onload = () => drawCropCanvas();
    cropImg.src = `/images/${state.project}/${kf.filename}?t=${Date.now()}`;

    updateCropPreview();
    document.getElementById('crop-hint').textContent = cropEdit
        ? (state.mode === 'double'
            ? 'drag corners/edges · click sets spine · drag spine ends to tilt · Enter save · Esc cancel'
            : 'drag corners/edges to fit the page · Enter save · Esc cancel')
        : '';
    document.getElementById('crop-edit-btn').classList.toggle('active', cropEdit);
}

function canvasPts(quad, c) {
    return quad.map(p => [p[0] * c.width, p[1] * c.height]);
}

function drawCropCanvas() {
    const c = document.getElementById('crop-canvas');
    const box = c.parentElement;
    const maxW = box.clientWidth, maxH = box.clientHeight - 24;
    if (maxW < 10 || !cropImg.width) return;

    const sc = Math.min(maxW / cropImg.width, maxH / cropImg.height, 1);
    c.width = cropImg.width * sc;
    c.height = cropImg.height * sc;
    c.style.width = c.width + 'px';
    c.style.height = c.height + 'px';

    const ctx = c.getContext('2d');
    ctx.drawImage(cropImg, 0, 0, c.width, c.height);
    if (!cropQuad) return;

    const P = canvasPts(cropQuad, c);

    // Keep-area stays lit, everything outside dims. even-odd fill handles an
    // arbitrary (tilted) quad, not just an axis-aligned rectangle.
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.beginPath();
    ctx.rect(0, 0, c.width, c.height);
    ctx.moveTo(P[0][0], P[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(P[i][0], P[i][1]);
    ctx.closePath();
    ctx.fill('evenodd');

    // box outline — solid when it's this frame's own, dashed when inherited/auto
    ctx.strokeStyle = cropEdit ? '#00ffff' : '#4361ee';
    ctx.lineWidth = cropEdit ? 2 : 1.5;
    ctx.setLineDash(cropSrc === 'manual' ? [] : (cropSrc === 'inherited' ? [8,3] : [4,4]));
    ctx.beginPath();
    ctx.moveTo(P[0][0], P[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(P[i][0], P[i][1]);
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // gutter — its own angle, independent of the box tilt
    if (state.mode === 'double' && cropGutter) {
        const [ga, gb] = gutterEnds(cropQuad, cropGutter);
        const A = [ga[0]*c.width, ga[1]*c.height], B = [gb[0]*c.width, gb[1]*c.height];
        ctx.strokeStyle = cropEdit ? '#00ffff' : '#ff3333';
        ctx.lineWidth = 2;
        ctx.setLineDash(gutterSrc === 'manual' ? [] : (gutterSrc === 'inherited' ? [8,3] : [4,4]));
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
        ctx.setLineDash([]);
        if (cropEdit) {
            ctx.fillStyle = '#00ffff';
            for (const p of [A, B]) {
                ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI*2); ctx.fill();
            }
        }
    }

    if (cropEdit) {
        ctx.fillStyle = '#0a0a0a';
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        for (const [x, y] of P) {
            ctx.beginPath(); ctx.rect(x-5, y-5, 10, 10); ctx.fill(); ctx.stroke();
        }
    }
}

function updateCropPreview() {
    clearTimeout(cropPreviewTimer);
    cropPreviewTimer = setTimeout(() => {
        const kf = state.keyframes[cropIdx];
        let url = `/api/crop-preview/${state.project}/${kf.filename}?mode=${state.mode}`;
        if (cropQuad) url += `&quad=${encodeURIComponent(JSON.stringify(cropQuad))}`;
        document.getElementById('crop-result-img').src = url + `&t=${Date.now()}`;
    }, 140);
}

function toggleCropEdit() {
    cropEdit = !cropEdit;
    renderCrop();
}

async function resetCropRect() {
    const kf = state.keyframes[cropIdx];
    delete kf.crop_quad; delete kf.gutter;
    await api(`/api/keyframes/${state.project}/update`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({frame_index: kf.frame_index,
                              crop_quad: null, gutter: null}),
    });
    renderCrop();
}

async function saveCropRect() {
    const kf = state.keyframes[cropIdx];
    kf.crop_quad = cropQuad.map(p => [+p[0].toFixed(5), +p[1].toFixed(5)]);
    const body = {frame_index: kf.frame_index, crop_quad: kf.crop_quad};
    if (state.mode === 'double' && cropGutter) {
        kf.gutter = {top:+cropGutter.top.toFixed(4), bot:+cropGutter.bot.toFixed(4)};
        body.gutter = kf.gutter;
    }
    await api(`/api/keyframes/${state.project}/update`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body),
    });
    cropEdit = false;
    renderCrop();
}

/* ── hit testing ── */
function segDist(p, a, b) {
    const dx = b[0]-a[0], dy = b[1]-a[1], l2 = dx*dx+dy*dy;
    const t = l2 === 0 ? 0 : Math.max(0, Math.min(1,
        ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/l2));
    return Math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy));
}

function pointInQuad(p, P) {
    let inside = false;
    for (let i = 0, j = 3; i < 4; j = i++) {
        const xi = P[i][0], yi = P[i][1], xj = P[j][0], yj = P[j][1];
        if ((yi > p[1]) !== (yj > p[1]) &&
            p[0] < ((xj-xi)*(p[1]-yi))/(yj-yi) + xi) inside = !inside;
    }
    return inside;
}

function cropHitTest(mx, my, c, shift) {
    const P = canvasPts(cropQuad, c);
    for (let i = 0; i < 4; i++)
        if (Math.hypot(mx-P[i][0], my-P[i][1]) < HANDLE)
            return {kind:'corner', i};

    if (state.mode === 'double' && cropGutter) {
        const [ga, gb] = gutterEnds(cropQuad, cropGutter);
        const A = [ga[0]*c.width, ga[1]*c.height], B = [gb[0]*c.width, gb[1]*c.height];
        if (Math.hypot(mx-A[0], my-A[1]) < HANDLE) return {kind:'gutterEnd', end:'top'};
        if (Math.hypot(mx-B[0], my-B[1]) < HANDLE) return {kind:'gutterEnd', end:'bot'};
        if (segDist([mx,my], A, B) < 8) return {kind:'gutterLine'};
    }

    for (let i = 0; i < 4; i++)
        if (segDist([mx,my], P[i], P[(i+1)%4]) < 9)
            return {kind:'edge', i};

    if (pointInQuad([mx,my], P)) {
        // In double mode the interior places the spine — that's the frequent
        // action. Shift moves the whole box, so a missed grab can't silently
        // drag a large translation into the stored crop.
        if (state.mode === 'double' && !shift) return {kind:'gutterPlace'};
        return {kind:'move'};
    }
    return null;
}

/* ── mouse ── */
document.addEventListener('DOMContentLoaded', () => {
    const c = document.getElementById('crop-canvas');
    if (!c) return;
    let last = null;

    const toFrac = (e) => {
        const r = c.getBoundingClientRect();
        return [(e.clientX-r.left)/c.width, (e.clientY-r.top)/c.height];
    };
    const toPx = (e) => {
        const r = c.getBoundingClientRect();
        return [e.clientX-r.left, e.clientY-r.top];
    };

    function placeGutter(p) {
        // Keep the current tilt: shift both ends by the same amount.
        const [tl,tr,br,bl] = cropQuad;
        const t = fracOnEdge(tl, tr, p), b = fracOnEdge(bl, br, p);
        const mid = (t + b) / 2;
        const half = (cropGutter.top - cropGutter.bot) / 2;
        cropGutter.top = Math.max(0.02, Math.min(0.98, mid + half));
        cropGutter.bot = Math.max(0.02, Math.min(0.98, mid - half));
        gutterSrc = 'manual';
    }

    c.addEventListener('mousedown', (e) => {
        if (!cropEdit || !cropQuad) return;
        const [mx,my] = toPx(e);
        cropDrag = cropHitTest(mx, my, c, e.shiftKey);
        last = toFrac(e);
        if (cropDrag && cropDrag.kind === 'gutterPlace') {
            placeGutter(last); drawCropCanvas(); updateCropPreview();
        }
    });

    c.addEventListener('mousemove', (e) => {
        const [mx,my] = toPx(e);
        if (!cropDrag) {
            if (cropEdit && cropQuad) {
                const h = cropHitTest(mx, my, c, e.shiftKey);
                c.style.cursor = !h ? 'crosshair'
                    : h.kind === 'corner' ? 'nwse-resize'
                    : h.kind === 'edge' ? (h.i % 2 ? 'ew-resize' : 'ns-resize')
                    : h.kind === 'gutterEnd' ? 'grab'
                    : h.kind === 'gutterLine' ? 'ew-resize'
                    : h.kind === 'move' ? 'move' : 'crosshair';
            }
            return;
        }
        const p = toFrac(e);
        const d = [p[0]-last[0], p[1]-last[1]];
        last = p;

        if (cropDrag.kind === 'corner') {
            cropQuad[cropDrag.i] = [Math.max(0,Math.min(1,p[0])),
                                     Math.max(0,Math.min(1,p[1]))];
            cropSrc = 'manual';
        } else if (cropDrag.kind === 'edge') {
            // move both endpoints of that edge together
            const i = cropDrag.i, j = (i+1)%4;
            for (const k of [i, j]) {
                cropQuad[k] = [Math.max(0,Math.min(1,cropQuad[k][0]+d[0])),
                               Math.max(0,Math.min(1,cropQuad[k][1]+d[1]))];
            }
            cropSrc = 'manual';
        } else if (cropDrag.kind === 'move') {
            cropQuad = cropQuad.map(q => [Math.max(0,Math.min(1,q[0]+d[0])),
                                          Math.max(0,Math.min(1,q[1]+d[1]))]);
            cropSrc = 'manual';
        } else if (cropDrag.kind === 'gutterEnd') {
            // This is the interaction the fraction-only model can't express:
            // drag one end to give the spine its own angle.
            const [tl,tr,br,bl] = cropQuad;
            if (cropDrag.end === 'top') cropGutter.top = fracOnEdge(tl, tr, p);
            else                        cropGutter.bot = fracOnEdge(bl, br, p);
            gutterSrc = 'manual';
        } else if (cropDrag.kind === 'gutterLine' || cropDrag.kind === 'gutterPlace') {
            placeGutter(p);
        }
        drawCropCanvas();
        updateCropPreview();
    });

    document.addEventListener('mouseup', () => { cropDrag = null; });
    window.addEventListener('resize', () => {
        if (state.currentView === 'crop') drawCropCanvas();
    });
});

function cropPrev() { if (cropIdx > 0) { cropIdx--; cropEdit=false; renderCrop(); } }
function cropNext() { if (cropIdx < state.keyframes.length-1) { cropIdx++; cropEdit=false; renderCrop(); } }

async function restoreOriginals() {
    if (!confirm('Re-extract every keyframe from the video at full resolution?\n\n' +
                 'Your crop boxes and gutters are kept — only the image files are rebuilt.')) return;
    const counter = document.getElementById('crop-counter');
    counter.textContent = 'Restoring originals…';
    try {
        const r = await api(`/api/keyframes/${state.project}/restore-originals`,
                            { method: 'POST' });
        state.keyframes = await api(`/api/keyframes/${state.project}`);
        renderCrop();
        counter.textContent = `Restored ${r.restored} frames` +
            (r.failed.length ? ` · ${r.failed.length} could not be read` : '');
    } catch (e) {
        counter.innerHTML = `<span style="color:var(--red)">Restore failed: ${e.message}</span>`;
    }
}

async function applyCropAll() {
    const counter = document.getElementById('crop-counter');

    // Poll a background task, surfacing any failure on screen rather than
    // leaving the label frozen mid-step.
    const run = async (url, label) => {
        let task;
        try {
            task = await api(url, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project: state.project, mode: state.mode }),
            });
        } catch (e) {
            throw new Error(`${label} could not start (${e.message})`);
        }
        while (true) {
            await new Promise(r => setTimeout(r, 400));
            const prog = await api(`/api/progress/${task.task_id}`);
            if (prog.status === 'done') return prog;
            if (prog.status === 'error') throw new Error(`${label} failed: ${prog.error}`);
            counter.textContent = `${label}… ${prog.progress ?? 0}%`;
        }
    };

    try {
        counter.textContent = 'Cropping…';
        await run('/api/process/crop', 'Cropping');
        counter.textContent = 'Splitting…';
        const split = await run('/api/process/split', 'Splitting');
        navigate('export');
        document.getElementById('export-status').textContent = `${split.pages} pages ready.`;
        loadExportThumbs();
    } catch (e) {
        counter.textContent = '';
        counter.innerHTML = `<span style="color:var(--red)">${e.message}</span>`;
        console.error(e);
    }
}


/* ── Live capture ─────────────────────────────────────────── */

let liveDefaults = null;
let livePoll = null;
let liveName = null;

const LIVE_FIELDS = {
    camera: 'live-camera', resolution: 'live-resolution', fps: 'live-fps',
    sound_set: 'live-sound', settle_threshold: 'live-settle',
    turn_threshold: 'live-turn', settle_time: 'live-settle-time',
    codec: 'live-codec', preview_height: 'live-preview-h',
    analysis_height: 'live-analysis-h', smoothing_window: 'live-smoothing',
    jpeg_quality: 'live-jpeg',
};

function defaultLiveName() {
    const d = new Date(), p = n => String(n).padStart(2, '0');
    return `book_${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
}

function fillLiveForm(v) {
    for (const [k, id] of Object.entries(LIVE_FIELDS)) {
        const el = document.getElementById(id);
        if (!el || v[k] == null) continue;
        if (el.tagName === 'SELECT' &&
            ![...el.options].some(o => o.value === String(v[k]))) {
            el.add(new Option(String(v[k]), String(v[k])));
        }
        el.value = v[k];
    }
    document.getElementById('live-guide').checked = v.guide !== false;
}

function readLiveForm() {
    const v = {};
    for (const [k, id] of Object.entries(LIVE_FIELDS)) {
        const el = document.getElementById(id);
        v[k] = el.type === 'number' ? parseFloat(el.value) : el.value;
    }
    v.guide = document.getElementById('live-guide').checked;
    v.name = document.getElementById('live-name').value.trim();
    return v;
}

async function openLiveSetup() {
    navigate('live');
    document.getElementById('live-error').textContent = '';
    if (!liveDefaults) liveDefaults = await api('/api/live/defaults');
    // last-used settings persist; the name is always fresh
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('liveSettings') || '{}'); } catch (e) {}
    fillLiveForm({ ...liveDefaults, ...saved });
    document.getElementById('live-name').value = defaultLiveName();
}

function resetLiveDefaults() {
    if (liveDefaults) fillLiveForm(liveDefaults);
    try { localStorage.removeItem('liveSettings'); } catch (e) {}
}

async function probeCameras() {
    const btn = document.getElementById('live-probe');
    const sel = document.getElementById('live-camera');
    btn.textContent = '…'; btn.disabled = true;
    try {
        const cams = await api('/api/live/cameras');
        const keep = sel.value;
        sel.innerHTML = '<option value="auto">Auto — best available</option>' +
            cams.map(c => `<option value="${c.index}">Camera ${c.index} · ${c.width}×${c.height}</option>`).join('');
        if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
        if (!cams.length) document.getElementById('live-error').textContent = 'No cameras found';
    } catch (e) {
        document.getElementById('live-error').textContent = 'Camera probe failed';
    }
    btn.textContent = 'Detect'; btn.disabled = false;
}

function previewSound() {
    const set = document.getElementById('live-sound').value;
    if (set === 'off') return;
    new Audio(`/live-sounds/${set}_capture.wav?t=${Date.now()}`).play().catch(() => {});
}

async function startLive(overwrite = false) {
    const v = readLiveForm();
    const err = document.getElementById('live-error');
    err.textContent = '';
    if (!v.name) { err.textContent = 'Give the recording a name'; return; }

    const { name, ...keep } = v;
    try { localStorage.setItem('liveSettings', JSON.stringify(keep)); } catch (e) {}

    const res = await fetch('/api/live/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...v, overwrite }),
    });
    const body = await res.json();
    if (res.status === 409 && body.error === 'exists') {
        if (confirm(`${body.message}. Record over it?`)) return startLive(true);
        return;
    }
    if (!res.ok) { err.textContent = body.error || 'Could not start'; return; }

    liveName = body.project;
    state.project = liveName;
    state.videoPath = body.video;
    navigate('live-run');
    document.getElementById('live-run-actions').innerHTML = '';
    clearInterval(livePoll);
    livePoll = setInterval(pollLive, 500);
    pollLive();
}

const LIVE_PHASE = {
    opening:   'Opening camera…',
    setup:     'Setting up — press Enter in the capture window to start',
    recording: 'Recording',
    saving:    'Saving…',
    done:      'Finished',
    cancelled: 'Closed before recording',
    error:     'Something went wrong',
};

async function pollLive() {
    if (!liveName) return;
    let st;
    try { st = await api(`/api/live/status/${liveName}`); } catch (e) { return; }

    const phase = st.paused && st.phase === 'recording' ? 'Paused' : (LIVE_PHASE[st.phase] || st.phase);
    const el = document.getElementById('live-phase');
    el.textContent = phase;
    el.dataset.phase = st.paused ? 'paused' : st.phase;
    document.getElementById('live-count').textContent = st.captures ?? 0;

    const meta = [];
    if (st.elapsed != null) {
        const m = Math.floor(st.elapsed / 60), s = Math.floor(st.elapsed % 60);
        meta.push(`${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    }
    if (st.phase === 'recording' && st.state) meta.push(st.state.toLowerCase());
    if (st.error) meta.push(st.error);
    document.getElementById('live-meta').textContent = meta.join('  ·  ');

    const actions = document.getElementById('live-run-actions');
    if (st.running) return;

    clearInterval(livePoll);
    document.getElementById('live-instructions').style.display = 'none';
    if (st.phase === 'done') {
        actions.innerHTML = `<button class="btn btn-primary" onclick="finishLive()">Continue to Review →</button>`;
    } else {
        actions.innerHTML = `<button class="btn" onclick="openLiveSetup()">← Back to settings</button>`;
    }
}

async function finishLive() {
    state.keyframes = await api(`/api/keyframes/${liveName}`);
    state.mode = 'double';
    document.getElementById('live-instructions').style.display = '';
    navigate('review');
    loadReview();
}


/* ── Analysis ─────────────────────────────────────────────── */

function openAnalysis() {
    navigate('analysis');
    loadAnalysis();
}

async function loadAnalysis() {
    const statsEl = document.getElementById('analysis-stats');
    const plotsEl = document.getElementById('analysis-plots');
    statsEl.innerHTML = '';
    plotsEl.innerHTML = '<p class="muted">Loading…</p>';

    let data;
    try {
        data = await api(`/api/analysis/${state.project}`);
    } catch (e) {
        plotsEl.innerHTML = '<p class="muted">Could not load analysis.</p>';
        return;
    }

    const st = data.stats || {};
    const cards = [];
    if (st.duration_sec != null)
        cards.push(['Video Length', `${Math.floor(st.duration_sec/60)}:${String(st.duration_sec%60).padStart(2,'0')}`]);
    if (st.resolution)   cards.push(['Resolution', st.resolution]);
    if (st.total_frames) cards.push(['Frames', st.total_frames.toLocaleString()]);
    if (st.keyframes)    cards.push(['Keyframes', st.keyframes]);
    if (st.pages)        cards.push(['Pages', st.pages]);
    if (st.storage) {
        const mb = v => v >= 1024 ? `${(v/1024).toFixed(1)} GB` : `${v} MB`;
        if (st.storage.images) cards.push(['Keyframe Files', mb(st.storage.images)]);
        if (st.storage.pages)  cards.push(['Page Files', mb(st.storage.pages)]);
        if (st.storage.pdf)    cards.push(['PDF', mb(st.storage.pdf)]);
    }
    statsEl.innerHTML = cards.map(([label, val]) => `
        <div class="stat-card">
            <div class="stat-value">${val}</div>
            <div class="stat-label">${label}</div>
        </div>`).join('');

    if (!data.plots.length) {
        plotsEl.innerHTML = '<p class="muted">No plots yet — run the pipeline first.</p>';
        return;
    }
    plotsEl.innerHTML = data.plots.map(p => `
        <div class="analysis-plot">
            <h3>${p.title}</h3>
            <div class="plot-desc">${p.desc}</div>
            ${themedPlot(state.project, p.name)}
        </div>`).join('');
}


/* ── Export ───────────────────────────────────────────────── */

async function loadExportThumbs() {
    try {
        const pages = await api(`/api/pages/${state.project}`);
        const container = document.getElementById('export-thumbs');
        const preview = pages.slice(0, 10);

        container.innerHTML = preview.map(pg => `
            <div class="export-thumb">
                <img src="/pages/${state.project}/${pg.filename}" loading="lazy">
                <span class="export-thumb-num">${pg.page_num}</span>
            </div>
        `).join('');

        document.getElementById('export-preview-count').textContent =
            `— first ${preview.length} of ${pages.length} pages`;

        // Stats cards
        const covers = pages.filter(p => p.type === 'cover' || p.type === 'backcover').length;
        const spreadPages = pages.filter(p => p.type === 'left' || p.type === 'right').length;
        const singlePages = pages.filter(p => p.type === 'page').length;

        document.getElementById('export-summary').innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${pages.length}</div>
                <div class="stat-label">Total Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${state.keyframes.length}</div>
                <div class="stat-label">Source Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${spreadPages || singlePages}</div>
                <div class="stat-label">${spreadPages ? 'Split Pages' : 'Single Pages'}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${covers}</div>
                <div class="stat-label">Covers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${state.mode === 'double' ? '2-Page' : '1-Page'}</div>
                <div class="stat-label">Mode</div>
            </div>
        `;
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

    status.textContent = `PDF created with ${result.pages} pages ✓`;
}


/* ── Keyboard shortcuts ──────────────────────────────────── */

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Crop view navigation
    if (state.currentView === 'crop') {
        if (cropEdit && cropQuad) {
            const step = e.shiftKey ? 0.001 : 0.004;
            const pan = (dx, dy) => {
                cropQuad = cropQuad.map(q => [Math.max(0,Math.min(1,q[0]+dx)),
                                              Math.max(0,Math.min(1,q[1]+dy))]);
                cropSrc = 'manual';
            };
            switch (e.key) {
                case 'ArrowLeft':  pan(-step, 0); break;
                case 'ArrowRight': pan(step, 0); break;
                case 'ArrowUp':    pan(0, -step); break;
                case 'ArrowDown':  pan(0, step); break;
                case '[': case ']': {
                    // tilt the SPINE, independent of the box
                    if (state.mode !== 'double' || !cropGutter) return;
                    const d = (e.key === ']' ? 1 : -1) * 0.004;
                    cropGutter.top = Math.max(0.02, Math.min(0.98, cropGutter.top + d));
                    cropGutter.bot = Math.max(0.02, Math.min(0.98, cropGutter.bot - d));
                    gutterSrc = 'manual';
                    break;
                }
                case 'Enter':  saveCropRect(); e.preventDefault(); return;
                case 'Escape': cropEdit = false; renderCrop(); e.preventDefault(); return;
                default: return;
            }
            drawCropCanvas(); updateCropPreview(); e.preventDefault();
            return;
        }
        switch (e.key) {
            case 'ArrowRight': case 'd': cropNext(); e.preventDefault(); break;
            case 'ArrowLeft': case 'a': cropPrev(); e.preventDefault(); break;
            case 'e': case 'E': toggleCropEdit(); e.preventDefault(); break;
            case 'r': case 'R': resetCropRect(); e.preventDefault(); break;
        }
        return;
    }

    // Review — quad view navigation
    if (state.currentView === 'review' && state.reviewView === 'quad') {
        switch (e.key) {
            case 'ArrowRight': case 'd': quadSelectNext(); e.preventDefault(); break;
            case 'ArrowLeft': case 'a': quadSelectPrev(); e.preventDefault(); break;
            case 'Enter': toggleView('single'); e.preventDefault(); break;
        }
        return;
    }

    // Review — grid view navigation
    if (state.currentView === 'review' && state.reviewView === 'grid') {
        switch (e.key) {
            case 'ArrowRight': case 'd':
                if (state.singleIdx < state.keyframes.length - 1) { state.singleIdx++; renderGrid(); }
                e.preventDefault(); break;
            case 'ArrowLeft': case 'a':
                if (state.singleIdx > 0) { state.singleIdx--; renderGrid(); }
                e.preventDefault(); break;
            case 'Enter': toggleView('single'); e.preventDefault(); break;
        }
        return;
    }

    // Review — single view
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

    // Gutter editing mode — arrows nudge the whole line, [ ] tilt it.
    if (gutterEditing && state.mode === 'double') {
        const step = e.shiftKey ? 0.001 : 0.005;
        const { g } = getEffectiveGutter(state.singleIdx);
        const clamp = v => Math.max(0.02, Math.min(0.98, v));
        switch (e.key) {
            case 'ArrowLeft':
                setReviewGutter({ top: clamp(g.top - step), bot: clamp(g.bot - step) });
                e.preventDefault(); return;
            case 'ArrowRight':
                setReviewGutter({ top: clamp(g.top + step), bot: clamp(g.bot + step) });
                e.preventDefault(); return;
            case '[':   // lean the spine left at the top
                setReviewGutter({ top: clamp(g.top - step), bot: clamp(g.bot + step) });
                e.preventDefault(); return;
            case ']':
                setReviewGutter({ top: clamp(g.top + step), bot: clamp(g.bot - step) });
                e.preventDefault(); return;
            case 'Enter':
                saveReviewGutter(); e.preventDefault(); return;
            case 'Escape':
                gutterEditing = false; renderSingle(); e.preventDefault(); return;
        }
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
                if (gutterEditing) boxEditing = false;
                renderSingle();
            }
            break;
        case 'b': case 'B':
            boxEditing = !boxEditing;
            if (boxEditing) gutterEditing = false;
            renderSingle();
            break;
    }
});


function themedPlot(project, name) {
    // Two files are rendered per plot; let the OS preference choose.
    const t = Date.now();
    const dark  = `/plots/${project}/${name}.png?t=${t}`;
    const light = `/plots/${project}/${name}_light.png?t=${t}`;
    return `<picture>
        <source media="(prefers-color-scheme: light)" srcset="${light}">
        <img src="${dark}" alt="${name}">
    </picture>`;
}


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
