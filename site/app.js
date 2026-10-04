/*
 * The Crewmate Project: one small world, one waypoint graph, one source of truth.
 * The characters below are project collaborators. They do not stand in for the
 * experimental agents in the simulator.
 */
const NAV_DATA = window.SKELD_NAVIGATION;
if (!NAV_DATA) throw new Error('The bundled Skeld navigation data did not load.');
const WORLD = Object.freeze({ width: 5792, height: 3168, center: { x: 2896, y: 1584 } });

// Tune travel, formation, camera, vent storytelling, and background timing here.
const EXPERIENCE = Object.freeze({
  crewSpeed: 790, crewAcceleration: 2200, crewSpacing: 118, wideRoomOffset: 24,
  walkFrameDuration: 94, cameraDamping: 4.2, cameraLookAhead: 245, cameraSettleDuration: 390,
  cameraZoomDuration: 820, cameraShipMargin: 180, dragThreshold: 6,
  ventPause: 420, ventEnterDuration: 300, ventMasaApproachSpeed: 420,
  ventTunnelDuration: 560, ventExitDuration: 310,
  shootingStarMinDelay: 8000, shootingStarMaxDelay: 15500, shootingStarDuration: 1550, shootingStarDurationJitter: .22,
  shootingStarMinLength: 78, shootingStarMaxLength: 156, shootingStarMinAngle: -36, shootingStarMaxAngle: -12,
  shootingStarMinBrightness: .36, shootingStarMaxBrightness: .66, shootingStarMinTravel: 20, shootingStarMaxTravel: 34,
  shootingStarMinDrift: 6, shootingStarMaxDrift: 18,
  reducedMotionSpeed: 1650, reducedMotionCameraDuration: 180, reducedMotionVentScale: 0.45
});

// Change each collaborator's color here. Their sprite artwork lives under assets/world/team/.
const TEAM = Object.freeze({
  hadi: Object.freeze({ name: 'Hadi', color: 'red', start: { x: NAV_DATA.anchors.crewHadi[0], y: NAV_DATA.anchors.crewHadi[1] } }),
  masa: Object.freeze({ name: 'Masa', color: 'black', start: { x: NAV_DATA.anchors.crewMasa[0], y: NAV_DATA.anchors.crewMasa[1] } })
});
const TEAM_COLORS = Object.freeze({
  red: Object.freeze({ suit: '#dd514c', trim: 'rgba(255, 126, 117, .62)', label: 'RED' }),
  black: Object.freeze({ suit: '#080b0d', trim: 'rgba(195, 218, 228, .62)', label: 'BLACK' })
});
const SPRITES = Object.freeze({
  red: Object.freeze({ base: 'assets/world/team/red', frames: { down: 18, left: 17, right: 17, up: 17 } }),
  black: Object.freeze({ base: 'assets/world/team/red', frames: { down: 18, left: 17, right: 17, up: 17 } })
});
const COLORED_SPRITES = new Map();
async function prepareSprites() {
  const jobs = [];
  for (const [direction, count] of Object.entries(SPRITES.red.frames)) {
    for (let frame = 1; frame <= count; frame++) {
      jobs.push(new Promise((resolve, reject) => {
        const image = new Image();
        image.onerror = () => reject(new Error(`Missing character frame: ${direction}/${frame}`));
        image.onload = () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(image, 0, 0);
            const original = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (const color of ['red', 'black']) {
              const decoded = new ImageData(CREW_PALETTE.recolor(original.data, color), canvas.width, canvas.height);
              ctx.putImageData(decoded, 0, 0);
              COLORED_SPRITES.set(`${color}/${direction}/${frame}`, canvas.toDataURL('image/png'));
            }
            resolve();
          } catch (error) { reject(error); }
        };
        image.src = `${SPRITES.red.base}/${direction}/${String(frame).padStart(2, '0')}.png`;
      }));
    }
  }
  await Promise.all(jobs);
}
const WAYPOINTS = Object.freeze(Object.fromEntries(Object.entries(NAV_DATA.anchors).map(([id, p]) => [id, { x: p[0], y: p[1] }])));
const VENTS = Object.freeze({
  shields: Object.freeze({ x: NAV_DATA.vents.shields[0], y: NAV_DATA.vents.shields[1] }),
  navigation: Object.freeze({ x: NAV_DATA.vents.navigation[0], y: NAV_DATA.vents.navigation[1] })
});
const VENT_APPROACHES = Object.freeze({
  shields: Object.freeze({ x: NAV_DATA.ventApproaches.shields[0], y: NAV_DATA.ventApproaches.shields[1] }),
  navigation: Object.freeze({ x: NAV_DATA.ventApproaches.navigation[0], y: NAV_DATA.ventApproaches.navigation[1] })
});
const DEBUG_NAV = new URLSearchParams(location.search).get('debugNav') === '1';
const DEBUG_NAV_GRAPH = DEBUG_NAV && new URLSearchParams(location.search).get('graph') === '1';
const REPAIRS = Object.freeze([
  { phase: 'NAVIGATION', title: 'Long routes could catch a corner', issue: 'A drawn map path did not prove that a moving character could finish it.', fix: 'Checked 4,914 physical runs; the default route set finished with no collisions or blocked steps.' },
  { phase: 'DEMO DISPLAY', title: 'The observer panel clipped on mixed-DPI screens', issue: 'The live match window could extend beyond the visible display area.', fix: 'Adjusted the fit to keep the full game and belief panel visible.' },
  { phase: 'LEGACY VIEWER', title: 'Closing the old viewer left a loop running', issue: 'A startup window could close before one remaining loop stopped.', fix: 'Added an exit guard and a focused regression check.' },
  { phase: 'DATA AUDIT', title: 'A same-tick clue looked like a time leak', issue: 'A clue and the next action can share one game tick.', fix: 'The audit now checks causal order and terminal state instead of rounded timestamps.' },
  { phase: 'DATA GUARD', title: 'A capitalized timeout escaped the first check', issue: 'Saved matches write “TIMEOUT” in capitals.', fix: 'Normalized the outcome text and rechecked all 1,400 matches.' },
  { phase: 'FAIR COMPARISON', title: 'One baseline was missing public meeting context', issue: 'The first current-clues-only ablation also dropped the live meeting roster.', fix: 'Restored that public context in 26,192 samples and retrained only the comparison.' }
]);

const MILESTONES = Object.freeze([
  {
    id: 'mission01', number: '01', originalMilestone: 'PROJECT ORIGIN', short: 'Origin', title: 'Navigation Origin', status: 'complete', waypoint: 'mission01',
    date: 'EARLY RESEARCH', location: 'CAFETERIA · THE FIRST QUESTION',
    summary: 'The project began with a simple question: could a small agent learn to move toward a goal?',
    detail: 'This is the beginning of the research story, rather than a separate benchmark result. The first experiments turned that question into a small navigation task and a staged plan.',
    metric: 'ONE QUESTION', metricLabel: 'the starting point for the experiments that followed',
    tags: ['PROJECT ORIGIN', 'THE JOURNEY STARTS HERE'], notes: 'The movement experiments were a foundation. They did not yet test suspicion, discussion, voting, or play against people.'
  },
  {
    id: 'mission02', number: '02', originalMilestone: 'STAGE 1', short: 'Basic navigation', title: 'Basic Navigation', status: 'complete', waypoint: 'mission02',
    date: 'EARLY RESEARCH', location: 'UPPER ENGINE · STAGE 1',
    summary: 'A small reinforcement-learning agent learned to move toward a goal in a simple 2D space.',
    detail: 'The first navigation experiment learned continuous movement from compact position and distance-sensor inputs. Its trained checkpoint is preserved in the project.',
    metric: '100 / 100', metricLabel: 'goal arrivals in the evaluation set · 93.86% average path efficiency',
    screenshot: 'assets/environment-overview.png', imageAlt: 'Diagram from the original project showing its early two-dimensional navigation environment.', imageCaption: 'The early project began by learning to move through a small 2D environment.',
    tags: ['PRESERVED MODEL', 'NAVIGATION ONLY'], notes: 'This showed goal-directed movement in an open arena. It did not demonstrate social reasoning, hidden-role deduction, or play against human opponents.'
  },
  {
    id: 'mission03', number: '03', originalMilestone: 'STAGE 2', short: 'Obstacle navigation', title: 'Obstacle Navigation', status: 'complete', waypoint: 'mission03',
    date: 'EARLY RESEARCH', location: 'LOWER ENGINE · STAGE 2',
    summary: 'The next environment added blocked paths, detours, and a way to recover when movement stalled.',
    detail: 'Stage 2 built on the first controller with obstacle detours, stuck detection, and recovery checks. It remains a useful navigation baseline.',
    metric: 'VALIDATED', metricLabel: 'obstacle detours and stuck-recovery behavior are implemented',
    screenshot: 'assets/environment-overview.png', imageAlt: 'Early navigation environment material from the project.', imageCaption: 'Stage 2 added obstacle detours and recovery behavior to the early navigation work.',
    tags: ['BASELINE KEPT', 'NO SOCIAL REASONING YET'], notes: 'The longer 250,000-step Stage 2 PPO run became optional and was deferred. The Stage 2 implementation remains in the repository.'
  },
  {
    id: 'mission04', number: '04', originalMilestone: 'STAGE 2.5', short: 'Skeld foundation', title: 'Skeld Foundation', status: 'complete', waypoint: 'mission04',
    date: 'STAGE 2.5', location: 'THE SKELD · MAP FOUNDATION',
    summary: 'The navigation world grew into a 14-room ship with reachable task locations.',
    detail: 'A Skeld-style research world connected the early navigation work to the social simulator that followed.',
    metric: '40 TASKS', metricLabel: '14 rooms · every listed task destination is reachable',
    screenshot: 'assets/skeld-map-overview.png', imageAlt: 'Project map overview of the Skeld layout with rooms and the ship silhouette.', imageCaption: 'The project’s Skeld map became shared ground for navigation and later matches.',
    tags: ['MAP FOUNDATION', 'NAVIGATION INFRASTRUCTURE'], notes: 'This is a project-built research simulation map. It is not a claim of complete commercial-game parity.'
  },
  {
    id: 'mission05', number: '05', originalMilestone: 'RESEARCH PIVOT', short: 'Research pivot', title: 'Research Pivot', status: 'complete', waypoint: 'mission05',
    date: 'A CHANGE IN DIRECTION', location: 'MEDBAY · THE RESEARCH PIVOT',
    summary: 'Better movement alone would not answer the harder question: how should a crewmate reason with incomplete information?',
    detail: 'The project widened from navigation into social reasoning. The early movement work stayed useful as a foundation, while the main question shifted toward observing, remembering, and acting on clues.',
    metric: 'A NEW QUESTION', metricLabel: 'from reaching a place to deciding whom to trust',
    tags: ['NAVIGATION WORK PRESERVED', 'SOCIAL REASONING BECAME THE GOAL'], notes: 'The optional long Stage 2 run was parked rather than erased. The project moved forward toward a controlled multi-player simulator.'
  },
  {
    id: 'mission06', number: '06', originalMilestone: 'PHASE 1', phaseStatusId: 'phase-1', short: 'Information boundaries', title: 'Information Boundaries', status: 'complete', waypoint: 'mission06',
    date: '02 OCT · 11:31', location: 'SECURITY · INFORMATION BOUNDARY',
    summary: 'Before a crewmate can suspect anyone, it must be kept away from the hidden answer.',
    detail: 'The simulation separates the complete hidden game state from the information available to one simulated crewmate. Public statements remain claims, not confirmed facts, and role labels stay out of the actor’s live input.',
    metric: '42', metricLabel: 'boundary checks at the first gate',
    tags: ['HIDDEN ROLES STAY HIDDEN', 'CLAIMS ARE NOT PROOF'], notes: 'This information contract is the foundation for later memory, beliefs, and policy evaluation.'
  },
  {
    id: 'mission07', number: '07', originalMilestone: 'PHASE 1.5', phaseStatusId: 'phase-1.5', short: 'Skeld fidelity', title: 'Skeld Fidelity', status: 'complete', waypoint: 'mission07',
    date: '02 OCT · 16:56', location: 'ADMIN · MAP + TASK LOCATIONS',
    summary: 'Turn the ship into a shared world of rooms, corridors, and reachable tasks.',
    detail: 'The map geometry and task locations gave navigation and later matches a common spatial foundation. This is the research map used inside the simulator.',
    metric: '14 ROOMS', metricLabel: '40 task locations on one connected ship layout',
    screenshot: 'assets/skeld-blueprint.png', imageAlt: 'The project’s Skeld map blueprint with room boundaries and corridor geometry.', imageCaption: 'Room geometry and paths are represented in the project’s map tools.',
    tags: ['MAP GEOMETRY', 'TASK LOCATIONS'], notes: 'Map reachability was validated in the project. It is not a certification of exact commercial-game behavior.'
  },
  {
    id: 'mission08', number: '08', originalMilestone: 'PHASE 2', phaseStatusId: 'phase-2', short: 'Reliable navigation', title: 'Reliable Navigation', status: 'complete', waypoint: 'mission08',
    date: '02 OCT · 19:48', location: 'ELECTRICAL · NAVIGATION CHECKS',
    summary: 'A route only counts if the moving character can finish it without snagging on a wall.',
    detail: 'The navigation service follows collision-safe routes, reaches consoles, and can recover or report failure explicitly. Testing checks actual movement, not just a path drawn on the map.',
    metric: '4,914', metricLabel: 'physical route runs passed · zero collisions or blocked steps',
    screenshot: 'assets/skeld-blueprint.png', imageAlt: 'Map geometry used for the project’s route and walkability checks.', imageCaption: 'The simulator checks routes against its map rather than a straight-line shortcut.',
    tags: ['3,906 DESTINATION PAIRS', '1,008 RANDOM STARTS'],
    repairs: [{ title: 'A drawn route was not enough', fix: 'Physical movement was checked across 4,914 routes and starts; all completed.' }]
  },
  {
    id: 'mission09', number: '09', originalMilestone: 'PHASE 3', phaseStatusId: 'phase-3', short: 'Scripted social world', title: 'Scripted Social World', status: 'complete', waypoint: 'mission09',
    date: '02 OCT · 21:52', location: 'CAFETERIA · FIVE-PLAYER MATCHES',
    summary: 'Build complete five-player matches where scripted players can move, do tasks, meet, report, and vote.',
    detail: 'The simulator can run full matches on the Skeld and replay them deterministically. Match actions are scripted today; this stage creates a controlled world for the later observer experiment.',
    metric: '1,000 + 1,000', metricLabel: 'complete scripted matches + exact independent replays',
    screenshot: 'assets/belief-meeting.png', imageAlt: 'Captured simulator meeting screen showing public discussion, player roster, and voting.', imageCaption: 'A captured scripted meeting from the simulator; it does not show a learned policy.',
    tags: ['ZERO TIMEOUTS', 'ZERO ILLEGAL MOVES'], notes: 'Five-player scripted games are a controlled test setting, not evidence of human-level social play.'
  },
  {
    id: 'mission10', number: '10', originalMilestone: 'PHASE 4', phaseStatusId: 'phase-4', short: 'Memory + belief', title: 'Memory + Belief', status: 'complete', waypoint: 'mission10',
    date: '02 OCT · 22:44 → 03 OCT · 06:54', location: 'COMMUNICATIONS · EVENT MEMORY',
    summary: 'Remember sightings and public claims, then estimate which of the other four players might be the impostor.',
    detail: 'The observer uses only allowed clues to estimate impostor probabilities. On the final scripted test sets it beat the calibrated evidence-rule baseline. This is a belief estimate, not an action or a vote.',
    metric: '67.41%', metricLabel: 'held-out patient-family accuracy · 70.54% in-distribution',
    screenshot: 'assets/belief-exploration.png', imageAlt: 'A real project capture showing the Skeld match, the observer’s belief probabilities, and its remembered events.', imageCaption: 'A project capture of the observer’s belief panel during a scripted match.',
    tags: ['1,400 SCRIPTED MATCHES', '64,097 SAMPLES', '327 CHECKS PASS'],
    notes: 'The observer uses only allowed clues and beats the calibrated evidence-rule baseline on both final scripted test sets. The patient-family score is measured against a held-out scripted opponent family; it does not establish robust deduction against people or unfamiliar strategies.',
    extraScreenshot: 'assets/learning-curves.png', extraImageAlt: 'Validation learning curves from the project’s Phase 4 belief-model experiment.', extraImageCaption: 'Validation curves from the registered Phase 4 experiment.',
    repairs: [
      { title: 'A same-tick clue looked like a leak', fix: 'Audited causal order and terminal state instead of relying on rounded timestamps.' },
      { title: 'The timeout guard missed capital letters', fix: 'Normalized “TIMEOUT” and checked all 1,400 saved matches.' },
      { title: 'A comparison lost the meeting roster', fix: 'Restored public context in 26,192 samples and retrained that baseline.' },
      { title: 'The demo panel clipped on one display setup', fix: 'Adjusted the live view to fit the full game and belief panel.' }
    ]
  },
  {
    id: 'mission11', number: '11', originalMilestone: 'PHASE 5', phaseStatusId: 'phase-5', short: 'Strategic policy', title: 'Strategic Policy', status: 'complete', waypoint: 'mission11',
    date: '03 OCT · LEARNING STUDY', location: 'NAVIGATION · LEARNED DECISIONS',
    summary: 'One crewmate learns to choose tasks, reports, meetings and votes from its own observations and remembered clues.',
    detail: 'PPO trains a new strategic network while the suspicion model stays frozen. The network selects goals; A* executes movement. Longer movement actions let a chosen task finish, while a newly seen body or witnessed kill returns control to the network. Scripted, random and idle players provide comparison points.',
    metric: 'EVALUATED', metricLabel: 'independent training seeds and whole-match outcomes',
    tags: ['ONE LEARNED CREWMATE', '9,000 FINAL MATCHES'],
    notes: 'Validation selected the policy before final testing. Independent tests show useful learned task contribution versus idle and random controls; superiority over scripted play is not established. The evidence includes own tasks, voting coverage, action choices and uncertainty. PPO loss is not accuracy.'
  },
  {
    id: 'mission12', number: '12', originalMilestone: 'PHASE 6', phaseStatusId: 'phase-6', short: 'Multi-agent future', title: 'Multi-Agent Future', status: 'future', waypoint: 'mission12',
    date: 'LATER · FUTURE WORK', location: 'SHIELDS · OPEN QUESTION',
    summary: 'Only after one learned crewmate works, explore adapting opponents and self-play.',
    detail: 'Learned impostors, multi-agent training, self-play, and richer conversations are future questions—not current capabilities or promised results.',
    metric: 'LATER', metricLabel: 'outside the current project scope',
    tags: ['AFTER PHASE 5', 'NO SELF-PLAY YET'], notes: 'First establish whether one crewmate policy is useful and whether its results hold up against unfamiliar strategies.'
  }
]);

const PARKED_BRANCH = Object.freeze({
  id: 'parked', number: '↘', originalMilestone: 'PARKED BRANCH', short: 'Long PPO run', title: 'Keep the useful baseline; park the long run', status: 'parked', waypoint: 'parked',
  date: 'OFF THE MAIN ROUTE', location: 'LOWER ENGINE · PARKED WORK',
  summary: 'More low-level navigation training was deferred when movement alone could not answer the social question.',
  detail: 'The Stage 2 code and Stage 1 model remain available. A full 250,000-step Stage 2 PPO run was made optional and moved off the project’s critical path; this work was not deleted.',
  metric: 'PARKED', metricLabel: 'extended Stage 2 training is deferred, not a completed result',
  tags: ['IMPLEMENTATION PRESERVED', 'NOT THE CURRENT QUESTION'], notes: 'The research direction changed because navigation is a mechanical sub-problem; the harder goal is reasoning with partial information and choosing useful actions.'
});
const DESTINATIONS = Object.freeze([...MILESTONES, PARKED_BRANCH]);
const STATUS_LABEL = Object.freeze({ complete: 'COMPLETE', 'in-progress': 'IN PROGRESS', parked: 'PARKED BRANCH', next: 'NEXT', future: 'FUTURE' });
const $ = (selector, root = document) => root.querySelector(selector);
const shipImage = $('#shipArt'), viewportEl = $('#worldViewport'), cameraEl = $('#shipWorld');
const markerLayer = $('#markerLayer'), nameLayer = $('#crewNameLayer'), trailScroll = $('#trailScroll');
const milestoneDialog = $('#milestoneDialog'), logDialog = $('#logDialog');
const activePath = $('#routeActive'), activeShadow = $('#routeActiveShadow');
const routeHistory = $('#routeHistory'), routeNext = $('#routeNext'), routeFuture = $('#routeFuture'), routeParked = $('#routeParked'), routeVent = $('#routeVent');
const destinationRing = $('#routeDestination'), ventMarkerA = $('#ventMarkerA'), ventMarkerB = $('#ventMarkerB'), ventOverlay = $('#ventOverlay');
const debugLayer = $('#routeDebug'), shootingStar = $('#shootingStar');
const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const MAP_PROJECTION = NAV_DATA.projection, GRID = NAV_DATA.grid, COLLISION = NAV_DATA.collision;
const NAV_NODE_BITS = decodeBase64(GRID.nodeBits), NAV_EDGE_MASKS = decodeBase64(GRID.edgeMasks), COLLISION_BITS = decodeBase64(COLLISION.bits);
const GRID_SIZE = GRID.cols * GRID.rows, NAV_DIRECTIONS = GRID.directions.map(([x, y]) => ({ x, y }));
const NAV_EDGE_COSTS = NAV_DIRECTIONS.map(({ x, y }) => {
  const gx = x * GRID.step, gy = y * GRID.step;
  return Math.hypot(MAP_PROJECTION.x[0] * gx + MAP_PROJECTION.x[1] * gy, MAP_PROJECTION.y[0] * gx + MAP_PROJECTION.y[1] * gy);
});
const state = {
  crewPosition: { ...TEAM.hadi.start }, crewCompanionPosition: { ...TEAM.masa.start },
  cameraPosition: { ...WORLD.center }, cameraZoom: 1, targetCameraZoom: 1, cameraMode: 'overview',
  targetDestination: null, activeRoute: [], movementProgress: 0, openMilestone: null,
  transitionState: 'idle', reducedMotion: reducedMotionQuery.matches, lastChronologicalMission: null, ventStoryUsed: false,
  viewport: { width: 0, height: 0, center: { x: 0, y: 0 }, baseScale: 1 },
  movingDirection: { hadi: 'down', masa: 'down' }, motion: null, lastFrame: 0
};
const crewEls = { hadi: { root: $('#crewHadi'), image: $('#crewHadi .crew-sprite') }, masa: { root: $('#crewMasa'), image: $('#crewMasa .crew-sprite') } };
const nameEls = {}, markerEls = new Map(), trailEls = new Map(), storyRouteCache = new Map();
let rafId = 0, toastTimer = 0, shootingStarTimer = 0, shootingStarAnimation = null;
let milestoneReturnTarget = null, logReturnTarget = null, pointerPan = null, hasDrawnDebug = false;

function decodeBase64(value) { return Uint8Array.from(atob(value), char => char.charCodeAt(0)); }
function bitAt(bits, index) { return index >= 0 && index < bits.length * 8 && (bits[index >> 3] & (1 << (index & 7))) !== 0; }
function roundGridCoordinate(value) {
  const lower = Math.floor(value), fraction = value - lower;
  if (Math.abs(fraction - .5) < 1e-8) return lower % 2 === 0 ? lower : lower + 1;
  return fraction < .5 ? lower : lower + 1;
}
function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function escapeHTML(value) { return String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }
function statusPlain(status) { return ({ complete: 'complete', 'in-progress': 'research and evaluation in progress', parked: 'parked off the main route', next: 'the next mission, not started', future: 'future work' })[status]; }
function mapPixelToGame(point) {
  const dx = point.x - MAP_PROJECTION.x[2], dy = point.y - MAP_PROJECTION.y[2];
  return { x: (MAP_PROJECTION.y[1] * dx - MAP_PROJECTION.x[1] * dy) / MAP_PROJECTION.inverseDet,
    y: (-MAP_PROJECTION.y[0] * dx + MAP_PROJECTION.x[0] * dy) / MAP_PROJECTION.inverseDet };
}
function gameToMap(point) {
  return { x: MAP_PROJECTION.x[0] * point.x + MAP_PROJECTION.x[1] * point.y + MAP_PROJECTION.x[2],
    y: MAP_PROJECTION.y[0] * point.x + MAP_PROJECTION.y[1] * point.y + MAP_PROJECTION.y[2] };
}
function collisionCellIndexes(value) {
  const nearest = Math.round(value);
  return Math.abs(value - nearest) < 1e-7 ? [nearest - 1, nearest] : [Math.floor(value)];
}
function safeGamePoint(point) {
  const cols = collisionCellIndexes((point.x - COLLISION.minX) / COLLISION.step);
  const rows = collisionCellIndexes((COLLISION.maxY - point.y) / COLLISION.step);
  return rows.some(row => cols.some(col => col >= 0 && row >= 0 && col < COLLISION.cols && row < COLLISION.rows &&
    bitAt(COLLISION_BITS, row * COLLISION.cols + col)));
}
function isSafeWorldPoint(point) { return safeGamePoint(mapPixelToGame(point)); }
function clearLineWorld(a, b) {
  const from = mapPixelToGame(a), to = mapPixelToGame(b), span = Math.hypot(to.x - from.x, to.y - from.y);
  const steps = Math.max(1, Math.ceil(span / (COLLISION.step * .5)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    if (!safeGamePoint({ x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t })) return false;
  }
  return true;
}
function gridPoint(index) {
  const row = Math.floor(index / GRID.cols), col = index - row * GRID.cols;
  return gameToMap({ x: GRID.minX + col * GRID.step, y: GRID.minY + row * GRID.step });
}
function navAttachCandidates(point, limit = 8) {
  const native = mapPixelToGame(point), col = roundGridCoordinate((native.x - GRID.minX) / GRID.step), row = roundGridCoordinate((native.y - GRID.minY) / GRID.step), found = [], seen = new Set();
  for (let radius = 0; radius <= 12; radius++) {
    for (let r = Math.max(0, row - radius); r <= Math.min(GRID.rows - 1, row + radius); r++) {
      for (let c = Math.max(0, col - radius); c <= Math.min(GRID.cols - 1, col + radius); c++) {
        const index = r * GRID.cols + c;
        if (!bitAt(NAV_NODE_BITS, index)) continue;
        const candidate = gridPoint(index);
        if (!seen.has(index) && clearLineWorld(point, candidate)) { seen.add(index); found.push({ index, cost: distance(point, candidate) }); }
      }
    }
    if (found.length >= limit && radius >= 4) break;
  }
  found.sort((a, b) => a.cost - b.cost);
  return found.slice(0, limit);
}
class MinHeap {
  constructor() { this.items = []; }
  push(id, score) {
    const item = { id, score }, items = this.items; items.push(item); let i = items.length - 1;
    while (i > 0) { const parent = (i - 1) >> 1; if (items[parent].score <= score) break; items[i] = items[parent]; i = parent; }
    items[i] = item;
  }
  pop() {
    const items = this.items; if (!items.length) return null;
    const root = items[0], tail = items.pop();
    if (items.length) { let i = 0; while (true) {
      const left = i * 2 + 1, right = left + 1; if (left >= items.length) break;
      const child = right < items.length && items[right].score < items[left].score ? right : left;
      if (items[child].score >= tail.score) break; items[i] = items[child]; i = child;
    } items[i] = tail; }
    return root;
  }
  get size() { return this.items.length; }
}
function findPhysicalRoute(start, goal) {
  if (!isSafeWorldPoint(start) || !isSafeWorldPoint(goal)) return null;
  if (distance(start, goal) < .01) return clearLineWorld(start, goal) ? [{ ...start }, { ...goal }] : null;
  if (clearLineWorld(start, goal)) return [{ ...start }, { ...goal }];
  const starts = navAttachCandidates(start), goals = navAttachCandidates(goal);
  if (!starts.length || !goals.length) return null;
  const targetCosts = new Map(goals.map(item => [item.index, item.cost]));
  const distances = new Float64Array(GRID_SIZE); distances.fill(Infinity);
  const previous = new Int32Array(GRID_SIZE); previous.fill(-2); const heap = new MinHeap();
  for (const item of starts) {
    if (item.cost >= distances[item.index]) continue;
    distances[item.index] = item.cost; previous[item.index] = -1; heap.push(item.index, item.cost);
  }
  let best = Infinity, end = -1;
  while (heap.size) {
    const item = heap.pop(), current = item.id;
    if (item.score !== distances[current]) continue;
    if (item.score >= best) break;
    const targetCost = targetCosts.get(current);
    if (targetCost !== undefined && item.score + targetCost < best) { best = item.score + targetCost; end = current; }
    const edgeMask = NAV_EDGE_MASKS[current]; if (!edgeMask) continue;
    const row = Math.floor(current / GRID.cols), col = current - row * GRID.cols;
    for (let direction = 0; direction < NAV_DIRECTIONS.length; direction++) {
      if (!(edgeMask & (1 << direction))) continue;
      const offset = NAV_DIRECTIONS[direction], nr = row + offset.y, nc = col + offset.x;
      if (nr < 0 || nc < 0 || nr >= GRID.rows || nc >= GRID.cols) continue;
      const next = nr * GRID.cols + nc, candidate = item.score + NAV_EDGE_COSTS[direction];
      if (candidate >= distances[next]) continue;
      distances[next] = candidate; previous[next] = current; heap.push(next, candidate);
    }
  }
  if (end < 0) return null;
  const routeIds = [];
  for (let current = end; current >= 0; current = previous[current]) routeIds.push(current);
  routeIds.reverse();
  const raw = [start, ...routeIds.map(gridPoint), goal], points = [raw[0]];
  let anchor = 0;
  while (anchor < raw.length - 1) {
    let farthest = raw.length - 1;
    while (farthest > anchor + 1 && !clearLineWorld(raw[anchor], raw[farthest])) farthest--;
    points.push(raw[farthest]); anchor = farthest;
  }
  return points.slice(1).every((point, index) => clearLineWorld(points[index], point)) ? points : null;
}
function pathLength(points) { let total = 0; for (let i = 1; i < points.length; i++) total += distance(points[i - 1], points[i]); return total; }
function pointAlong(points, travel) {
  if (!points.length) return { x: 0, y: 0 };
  if (travel <= 0 || points.length === 1) return { ...points[0] };
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i], segment = distance(a, b);
    if (travel <= segment || i === points.length - 1) {
      const t = segment < .001 ? 1 : clamp(travel / segment, 0, 1);
      return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    }
    travel -= segment;
  }
  return { ...points.at(-1) };
}
function pathToSvg(points) { return points.length ? `M ${points.map(p => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' L ')}` : ''; }
function missionPosition(mission) { return WAYPOINTS[mission.waypoint]; }
function markerPosition(mission) {
  const point = NAV_DATA.markerAnchors?.[mission.waypoint] || NAV_DATA.anchors[mission.waypoint];
  return { x: point[0], y: point[1] };
}
function routeForStoryLeg(from, to) {
  const key = `${from.id}>${to.id}`;
  if (!storyRouteCache.has(key)) storyRouteCache.set(key, findPhysicalRoute(missionPosition(from), missionPosition(to)) || []);
  return storyRouteCache.get(key);
}
function storyLine(startIndex, endIndex) {
  const pieces = [];
  for (let i = startIndex; i < endIndex; i++) {
    const route = routeForStoryLeg(MILESTONES[i], MILESTONES[i + 1]);
    if (route.length) pieces.push(...(pieces.length ? route.slice(1) : route));
  }
  return pieces;
}
function renderStoryRoutes() {
  routeHistory.setAttribute('d', pathToSvg(storyLine(0, 9)));
  routeNext.setAttribute('d', pathToSvg(storyLine(9, 10)));
  routeFuture.setAttribute('d', pathToSvg(storyLine(10, 11)));
  const branch = findPhysicalRoute(missionPosition(MILESTONES[2]), missionPosition(PARKED_BRANCH));
  routeParked.setAttribute('d', pathToSvg(branch || []));
}
function destinationLabel(destination) { return destination.id === 'parked' ? 'PARKED PPO' : `MISSION ${destination.number}`; }
function markerMarkup(destination) {
  const button = document.createElement('button'); button.className = 'mission-marker'; button.type = 'button';
  button.dataset.milestone = destination.id; button.dataset.status = destination.status;
  button.setAttribute('aria-label', `${destinationLabel(destination)}: ${destination.title}. ${statusPlain(destination.status)}. Select to travel with Hadi and Masa.`);
  button.title = `${destinationLabel(destination)} · ${destination.title} · ${statusPlain(destination.status)}`;
  const bead = document.createElement('span'); bead.className = 'marker-beacon'; bead.setAttribute('aria-hidden', 'true');
  const title = document.createElement('span'); title.className = 'marker-title'; title.textContent = destination.id === 'parked' ? 'PARKED PPO' : `MISSION ${destination.number}`;
  const status = document.createElement('span'); status.className = 'marker-state'; status.textContent = STATUS_LABEL[destination.status];
  button.append(bead, title, status); markerLayer.append(button); markerEls.set(destination.id, button);
}
function renderNavigation() {
  DESTINATIONS.forEach(markerMarkup);
  for (const key of Object.keys(TEAM)) {
    const plate = document.createElement('span'); plate.className = `crew-nameplate ${key === 'hadi' ? 'is-hadi' : 'is-masa'}`;
    plate.textContent = TEAM[key].name; nameLayer.append(plate); nameEls[key] = plate;
    plate.style.borderColor = TEAM_COLORS[TEAM[key].color].trim;
    const person = document.querySelector(`[data-person="${key}"]`);
    if (person) {
      const dot = person.querySelector('i'); dot.style.background = TEAM_COLORS[TEAM[key].color].suit;
      dot.style.border = TEAM[key].color === 'black' ? '1px solid #c9d4d8' : '0';
      person.querySelector('small').textContent = TEAM_COLORS[TEAM[key].color].label;
    }
  }
  for (const mission of MILESTONES) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'trail-stop';
    button.dataset.milestone = mission.id; button.dataset.status = mission.status;
    button.setAttribute('aria-label', `Travel to Mission ${mission.number}, ${mission.title}. ${statusPlain(mission.status)}.`);
    const bead = document.createElement('span'); bead.className = 'trail-bead'; bead.textContent = mission.number;
    const label = document.createElement('span'); label.className = 'trail-label'; label.textContent = mission.short;
    const date = document.createElement('span'); date.className = 'trail-date'; date.textContent = mission.date;
    button.append(bead, label, date); trailScroll.append(button); trailEls.set(mission.id, button);
  }
  renderStoryRoutes(); renderRepairs();
}
function renderRepairs() {
  const host = $('#repairList');
  for (const repair of REPAIRS) {
    const entry = document.createElement('article'); entry.className = 'repair-entry';
    const title = document.createElement('b'); title.textContent = repair.title;
    const problem = document.createElement('p'); problem.textContent = `${repair.phase} · ${repair.issue}`;
    const fix = document.createElement('small'); fix.textContent = `FIXED · ${repair.fix}`;
    entry.append(title, problem, fix); host.append(entry);
  }
}
function resizeCamera() {
  const rect = viewportEl.getBoundingClientRect();
  state.viewport.width = rect.width; state.viewport.height = rect.height;
  const headerHeight = $('.mission-header').getBoundingClientRect().height, dockHeight = $('.milestone-trail').getBoundingClientRect().height;
  state.viewport.center = { x: rect.width / 2, y: (headerHeight + rect.height - dockHeight) / 2 };
  state.viewport.baseScale = Math.max(.001, Math.min((rect.width - 16) / WORLD.width, (rect.height - 32) / WORLD.height));
  state.cameraPosition = clampCameraPosition(state.cameraPosition, state.cameraZoom); applyWorldTransform(); updateMarkers(); updateDebugBounds();
}
function cameraScale(zoom = state.cameraZoom) { return state.viewport.baseScale * zoom; }
function screenPoint(position) {
  const scale = cameraScale();
  return { x: state.viewport.center.x + (position.x - state.cameraPosition.x) * scale,
    y: state.viewport.center.y + (position.y - state.cameraPosition.y) * scale };
}
function clampCameraPosition(position, zoom = state.cameraZoom) {
  const scale = Math.max(.001, cameraScale(zoom));
  const halfWidth = state.viewport.width / (2 * scale), halfHeight = state.viewport.height / (2 * scale);
  const minX = Math.max(0, halfWidth - EXPERIENCE.cameraShipMargin), maxX = Math.min(WORLD.width, WORLD.width - halfWidth + EXPERIENCE.cameraShipMargin);
  const minY = Math.max(0, halfHeight - EXPERIENCE.cameraShipMargin), maxY = Math.min(WORLD.height, WORLD.height - halfHeight + EXPERIENCE.cameraShipMargin);
  return { x: minX > maxX ? WORLD.center.x : clamp(position.x, minX, maxX),
    y: minY > maxY ? WORLD.center.y : clamp(position.y, minY, maxY) };
}
function focusCameraPosition(position, zoom = focusZoom()) { return clampCameraPosition(position, zoom); }
function markerScreenOffset(id) {
  const offsets = {
    mission01: { x: -34, y: -10 }, mission02: { x: -15, y: -18 }, mission03: { x: -25, y: 17 },
    mission04: { x: -18, y: -9 }, mission05: { x: -14, y: -14 }, mission06: { x: -15, y: -8 },
    mission07: { x: 12, y: -14 }, mission08: { x: -12, y: 17 }, mission09: { x: 14, y: -12 },
    mission10: { x: 12, y: -12 }, mission11: { x: 14, y: -12 }, mission12: { x: -16, y: 16 }, parked: { x: -18, y: 15 }
  };
  return offsets[id] || { x: 0, y: -8 };
}
function applyWorldTransform() {
  const scale = cameraScale(), x = state.viewport.center.x - state.cameraPosition.x * scale, y = state.viewport.center.y - state.cameraPosition.y * scale;
  cameraEl.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0) scale(${scale.toFixed(5)})`;
  updateDebugBounds();
}
function updateMarkers() {
  for (const destination of DESTINATIONS) {
    const button = markerEls.get(destination.id); if (!button) continue;
    const p = screenPoint(markerPosition(destination)), offset = markerScreenOffset(destination.id);
    button.style.setProperty('--screen-x', `${(p.x + offset.x).toFixed(1)}px`);
    button.style.setProperty('--screen-y', `${(p.y + offset.y).toFixed(1)}px`);
    button.classList.toggle('is-selected', state.targetDestination?.id === destination.id);
    button.classList.toggle('is-next', ['next', 'in-progress'].includes(destination.status));
    button.hidden = p.x < -145 || p.x > state.viewport.width + 145 || p.y < 10 || p.y > state.viewport.height + 30;
  }
  for (const key of Object.keys(TEAM)) {
    const p = screenPoint(key === 'hadi' ? state.crewPosition : state.crewCompanionPosition), plate = nameEls[key];
    plate.style.setProperty('--screen-x', `${p.x.toFixed(1)}px`);
    plate.style.setProperty('--screen-y', `${(p.y - 152 * cameraScale() - 7).toFixed(1)}px`);
  }
}
function setSpriteDirection(key, direction, moving, dt = 0) {
  const cfg = TEAM[key], spriteSet = SPRITES[cfg.color], root = crewEls[key].root, img = crewEls[key].image;
  root.dataset.color = cfg.color;
  if (direction !== state.movingDirection[key]) {
    state.movingDirection[key] = direction; root.dataset.facing = direction; root._walkFrame = 0; root._walkClock = 0;
  }
  root.classList.toggle('is-walking', moving); root.classList.toggle('is-idle', !moving);
  if (moving) {
    root._walkClock = (root._walkClock || 0) + dt * 1000;
    if (root._walkClock >= EXPERIENCE.walkFrameDuration) {
      const steps = Math.floor(root._walkClock / EXPERIENCE.walkFrameDuration);
      root._walkClock %= EXPERIENCE.walkFrameDuration; root._walkFrame = ((root._walkFrame || 0) + steps) % spriteSet.frames[direction];
    }
  } else { root._walkFrame = 0; root._walkClock = 0; }
  const frame = moving && !state.reducedMotion ? ((root._walkFrame || 0) % spriteSet.frames[direction]) + 1 : 1;
  const src = COLORED_SPRITES.get(`${cfg.color}/${direction}/${frame}`);
  if (img.getAttribute('src') !== src) img.setAttribute('src', src);
}
function placeCrew(hadi, masa, directionHadi = 'down', directionMasa = directionHadi, moving = false, dt = 0) {
  state.crewPosition = { x: hadi.x, y: hadi.y }; state.crewCompanionPosition = { x: masa.x, y: masa.y };
  for (const [key, position, direction] of [['hadi', hadi, directionHadi], ['masa', masa, directionMasa]]) {
    setCrewPositionVisual(key, position);
    setSpriteDirection(key, direction, moving, dt);
  }
}
function setCrewPositionVisual(key, position) {
  const root = crewEls[key].root;
  root.style.setProperty('--world-x', `${position.x.toFixed(2)}px`);
  root.style.setProperty('--world-y', `${position.y.toFixed(2)}px`);
}
function facingFor(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  if (Math.abs(dx) > Math.abs(dy)) return dx < 0 ? 'left' : 'right';
  return dy < 0 ? 'up' : 'down';
}
function facingAlong(points, progress, fallback = 'down') {
  const here = pointAlong(points, progress), ahead = pointAlong(points, Math.min(pathLength(points), progress + 58));
  return distance(here, ahead) < .1 ? fallback : facingFor(here, ahead);
}
function setRouteArt(points) {
  const d = pathToSvg(points); activePath.setAttribute('d', d); activeShadow.setAttribute('d', d); activePath.classList.remove('is-arrived');
  const end = points.at(-1);
  if (end) { destinationRing.setAttribute('cx', end.x); destinationRing.setAttribute('cy', end.y); destinationRing.classList.add('is-active'); }
  activePath.classList.add('is-moving'); activeShadow.classList.add('is-moving');
}
function clearRouteArt() {
  activePath.classList.remove('is-moving', 'is-arrived'); activeShadow.classList.remove('is-moving'); destinationRing.classList.remove('is-active');
  state.activeRoute = []; routeVent.setAttribute('d', ''); ventMarkerA.classList.remove('is-active'); ventMarkerB.classList.remove('is-active');
}
function setTravelMessage(message) { $('#travelStatus').textContent = message; }
function focusZoom() { return state.viewport.width <= 560 ? 3.05 : state.viewport.width <= 900 ? 1.9 : 1.45; }
function easeInOut(value) { return value < .5 ? 4 * value * value * value : 1 - Math.pow(-2 * value + 2, 3) / 2; }
function scheduleFrame() { if (!rafId) rafId = requestAnimationFrame(animationFrame); }
function stopMotion() {
  const previous = state.motion;
  if (rafId) cancelAnimationFrame(rafId); rafId = 0; state.motion = null; state.lastFrame = 0;
  if (previous && /^vent/.test(previous.phase) && previous.entryPositions) {
    state.crewPosition = { ...previous.entryPositions.hadi }; state.crewCompanionPosition = { ...previous.entryPositions.masa };
    setCrewPositionVisual('hadi', state.crewPosition); setCrewPositionVisual('masa', state.crewCompanionPosition);
  }
  for (const key of Object.keys(crewEls)) crewEls[key].root.classList.remove('is-venting', 'is-emerging');
  ventOverlay.classList.remove('is-in-vent'); ventMarkerA.classList.remove('is-active'); ventMarkerB.classList.remove('is-active'); routeVent.setAttribute('d', '');
}
function transitionVelocity(current, desired, acceleration, dt) {
  const delta = acceleration * dt;
  return current < desired ? Math.min(desired, current + delta) : Math.max(desired, current - delta);
}
function advanceTravelDistance(motion, dt, speed, acceleration) {
  const remaining = Math.max(0, motion.routeLength - motion.distance);
  if (remaining <= .4) { motion.distance = motion.routeLength; motion.velocity = 0; return true; }
  const desired = Math.min(speed, Math.sqrt(2 * acceleration * remaining)), prior = motion.velocity;
  motion.velocity = transitionVelocity(prior, desired, acceleration, dt);
  motion.distance = Math.min(motion.routeLength, motion.distance + (prior + motion.velocity) * .5 * dt);
  if (motion.routeLength - motion.distance < .6) { motion.distance = motion.routeLength; motion.velocity = 0; return true; }
  return false;
}
function joinRoutes(first, second) {
  if (!first?.length) return second || [];
  if (!second?.length) return first;
  return [...first, ...second.slice(distance(first.at(-1), second[0]) < .001 ? 1 : 0)];
}
function prepareWalk(motion, route, phase) {
  if (!route || route.length < 2) return false;
  const connector = findPhysicalRoute(state.crewCompanionPosition, route[0]);
  if (!connector) return false;
  const sharedRoute = joinRoutes(connector, route);
  if (sharedRoute.length < 2 || !sharedRoute.slice(1).every((point, i) => clearLineWorld(sharedRoute[i], point))) return false;
  motion.phase = phase; motion.elapsed = 0; motion.route = sharedRoute;
  motion.routeStartOffset = pathLength(connector); motion.routeLength = pathLength(route); motion.totalRouteLength = pathLength(sharedRoute);
  motion.distance = 0; motion.velocity = 0; motion.formationBlend = 0; motion.formationSide = 1;
  state.activeRoute = sharedRoute; state.movementProgress = 0; setRouteArt(sharedRoute); return true;
}
function companionDistanceAlong(motion, traveled) {
  const ratio = motion.routeLength > .1 ? Math.max(0, (motion.totalRouteLength - EXPERIENCE.crewSpacing) / motion.routeLength) : 0;
  return Math.min(motion.totalRouteLength, traveled * ratio);
}
function companionPositionAlong(motion, traveled) {
  const companionDistance = companionDistanceAlong(motion, traveled);
  const base = pointAlong(motion.route, companionDistance);
  if (state.reducedMotion || EXPERIENCE.wideRoomOffset <= 0 || companionDistance <= 12) return base;
  const before = pointAlong(motion.route, Math.max(0, companionDistance - 26));
  const after = pointAlong(motion.route, Math.min(motion.totalRouteLength, companionDistance + 26));
  const length = Math.max(1, distance(before, after)), normal = { x: -(after.y - before.y) / length, y: (after.x - before.x) / length };
  let candidate = null;
  for (const side of [motion.formationSide, -motion.formationSide]) {
    const trial = { x: base.x + normal.x * EXPERIENCE.wideRoomOffset * side, y: base.y + normal.y * EXPERIENCE.wideRoomOffset * side };
    if (isSafeWorldPoint(trial) && clearLineWorld(base, trial)) { candidate = trial; motion.formationSide = side; break; }
  }
  const targetBlend = candidate ? 1 : 0, dt = motion.lastDt || .016;
  motion.formationBlend += (targetBlend - motion.formationBlend) * (1 - Math.exp(-4 * dt));
  if (!candidate || motion.formationBlend < .015) return base;
  return { x: base.x + (candidate.x - base.x) * motion.formationBlend, y: base.y + (candidate.y - base.y) * motion.formationBlend };
}
function updateCrewForWalk(motion, dt) {
  const hadiProgress = motion.routeStartOffset + motion.distance, masaProgress = companionDistanceAlong(motion, motion.distance);
  const hadi = pointAlong(motion.route, hadiProgress), masa = companionPositionAlong(motion, motion.distance);
  placeCrew(hadi, masa, facingAlong(motion.route, hadiProgress, state.movingDirection.hadi),
    facingAlong(motion.route, masaProgress, state.movingDirection.masa), true, dt);
  state.movementProgress = motion.routeLength ? motion.distance / motion.routeLength : 1;
}
function dampPoint(current, target, damping, dt) {
  const amount = 1 - Math.exp(-damping * dt);
  return clampCameraPosition({ x: current.x + (target.x - current.x) * amount, y: current.y + (target.y - current.y) * amount }, state.cameraZoom);
}
function followCrew(motion, dt, lookahead = EXPERIENCE.cameraLookAhead) {
  if (state.cameraMode === 'overview') {
    state.cameraPosition = dampPoint(state.cameraPosition, WORLD.center, EXPERIENCE.cameraDamping, dt);
    state.cameraZoom += (1 - state.cameraZoom) * (1 - Math.exp(-EXPERIENCE.cameraDamping * dt)); state.targetCameraZoom = 1; return;
  }
  state.targetCameraZoom = focusZoom();
  state.cameraZoom += (state.targetCameraZoom - state.cameraZoom) * (1 - Math.exp(-EXPERIENCE.cameraDamping * dt));
  let focus = state.crewPosition;
  if (motion.route?.length && motion.phase.startsWith('walk')) focus = pointAlong(motion.route, Math.min(motion.totalRouteLength, motion.routeStartOffset + motion.distance + lookahead));
  state.cameraPosition = dampPoint(state.cameraPosition, focusCameraPosition(focus, state.cameraZoom), EXPERIENCE.cameraDamping, dt);
}
function startVentBoarding(motion) {
  motion.entryPositions = { hadi: { ...state.crewPosition }, masa: { ...state.crewCompanionPosition } };
  state.ventStoryUsed = true; motion.elapsed = 0; motion.phase = 'ventPause';
  ventMarkerA.setAttribute('cx', VENTS.shields.x); ventMarkerA.setAttribute('cy', VENTS.shields.y);
  ventMarkerB.setAttribute('cx', VENTS.navigation.x); ventMarkerB.setAttribute('cy', VENTS.navigation.y);
  ventMarkerA.classList.add('is-active');
  setTravelMessage('A playful ship-vent shortcut in this website · simulator movement is unchanged.');
}
function entranceToVent(motion) { crewEls.hadi.root.classList.add('is-venting'); motion.elapsed = 0; motion.phase = 'ventHadiEntry'; }
function startMasaVentApproach(motion) {
  const route = findPhysicalRoute(state.crewCompanionPosition, VENT_APPROACHES.shields);
  if (!route) { failVentTravel(motion, 'The crew could not reach the vent along a clear corridor.'); return; }
  motion.masaVentRoute = route; motion.masaVentLength = pathLength(route); motion.masaVentDistance = 0; motion.masaVentVelocity = 0;
  state.activeRoute = route; setRouteArt(route);
  motion.elapsed = 0; motion.phase = 'ventMasaWalk'; setSpriteDirection('masa', facingAlong(route, 0), true);
}
function startVentTunnel(motion) {
  state.activeRoute = []; activePath.classList.remove('is-moving'); activeShadow.classList.remove('is-moving');
  crewEls.masa.root.classList.add('is-venting'); motion.phase = 'ventMasaEntry'; motion.elapsed = 0;
}
function failVentTravel(motion, message) {
  if (motion.entryPositions) placeCrew(motion.entryPositions.hadi, motion.entryPositions.masa, state.movingDirection.hadi, state.movingDirection.masa, false);
  for (const key of Object.keys(crewEls)) crewEls[key].root.classList.remove('is-venting', 'is-emerging');
  ventOverlay.classList.remove('is-in-vent'); ventMarkerA.classList.remove('is-active'); ventMarkerB.classList.remove('is-active'); routeVent.setAttribute('d', '');
  state.activeRoute = []; activePath.classList.remove('is-moving'); activeShadow.classList.remove('is-moving');
  setTravelMessage(message); motion.phase = 'failed';
}
function makeVentTransferPath() {
  const a = VENTS.shields, b = VENTS.navigation, control = { x: (a.x + b.x) / 2 + 145, y: (a.y + b.y) / 2 - 110 };
  routeVent.setAttribute('d', `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} Q ${control.x.toFixed(1)} ${control.y.toFixed(1)} ${b.x.toFixed(1)} ${b.y.toFixed(1)}`);
}
function safeLateralPoint(origin, tangent, width) {
  const mag = Math.max(1, Math.hypot(tangent.x, tangent.y)), normal = { x: -tangent.y / mag, y: tangent.x / mag };
  for (const amount of [width, width * .72, width * .45]) for (const side of [1, -1]) {
    const candidate = { x: origin.x + normal.x * amount * side, y: origin.y + normal.y * amount * side };
    if (isSafeWorldPoint(candidate) && clearLineWorld(origin, candidate)) return candidate;
  }
  return { ...origin };
}
function showVentExit(motion) {
  const exitRoute = findPhysicalRoute(VENT_APPROACHES.navigation, missionPosition(motion.mission));
  if (!exitRoute) { failVentTravel(motion, 'The crew could not reconnect to a clear route after the vent.'); return false; }
  motion.exitRoute = exitRoute;
  const next = pointAlong(exitRoute, Math.min(92, pathLength(exitRoute)));
  state.crewPosition = { ...VENT_APPROACHES.navigation };
  state.crewCompanionPosition = safeLateralPoint(VENT_APPROACHES.navigation, { x: next.x - VENT_APPROACHES.navigation.x, y: next.y - VENT_APPROACHES.navigation.y }, 74);
  crewEls.hadi.root.classList.remove('is-venting'); crewEls.hadi.root.classList.add('is-emerging');
  ventMarkerA.classList.remove('is-active'); ventMarkerB.classList.add('is-active');
  placeCrew(state.crewPosition, state.crewCompanionPosition, facingAlong(exitRoute, 0), facingAlong(exitRoute, 0), false);
  state.activeRoute = exitRoute; setRouteArt(exitRoute);
  motion.elapsed = 0; motion.phase = 'ventHadiExit'; setTravelMessage('Hadi and Masa are back on the corridor route.'); return true;
}
function startWalkingAfterVent(motion) {
  const route = findPhysicalRoute(state.crewPosition, missionPosition(motion.mission));
  if (!route || !prepareWalk(motion, route, 'walkToMission')) {
    failVentTravel(motion, 'The crew could not reconnect to a clear ship route.'); return;
  }
  crewEls.hadi.root.classList.remove('is-emerging', 'is-venting'); crewEls.masa.root.classList.remove('is-emerging', 'is-venting');
  ventOverlay.classList.remove('is-in-vent'); ventMarkerA.classList.remove('is-active'); ventMarkerB.classList.remove('is-active'); routeVent.setAttribute('d', '');
  setTravelMessage(`Mission ${motion.mission.number} · the corridor route continues.`);
}
function updateTravel(motion, dt) {
  const reducedScale = state.reducedMotion ? EXPERIENCE.reducedMotionVentScale : 1;
  if (motion.phase === 'walkToMission' || motion.phase === 'walkToVent') {
    motion.lastDt = dt;
    const speed = state.reducedMotion ? EXPERIENCE.reducedMotionSpeed : EXPERIENCE.crewSpeed;
    const done = advanceTravelDistance(motion, dt, speed, EXPERIENCE.crewAcceleration * (state.reducedMotion ? 2 : 1));
    updateCrewForWalk(motion, dt); followCrew(motion, dt);
    if (done) {
      placeCrew(pointAlong(motion.route, motion.routeStartOffset + motion.routeLength), companionPositionAlong(motion, motion.routeLength), state.movingDirection.hadi, state.movingDirection.masa, false);
      if (motion.phase === 'walkToVent') startVentBoarding(motion); else beginSettle(motion);
    }
    return;
  }
  motion.elapsed += dt * (state.reducedMotion ? 1.65 : 1);
  if (motion.phase === 'ventPause') {
    followCrew(motion, dt, 0); if (motion.elapsed >= EXPERIENCE.ventPause * reducedScale / 1000) entranceToVent(motion);
  } else if (motion.phase === 'ventHadiEntry') {
    followCrew(motion, dt, 0); if (motion.elapsed >= EXPERIENCE.ventEnterDuration * reducedScale / 1000) startMasaVentApproach(motion);
  } else if (motion.phase === 'ventMasaWalk') {
    const remaining = Math.max(0, motion.masaVentLength - motion.masaVentDistance);
    const speed = state.reducedMotion ? EXPERIENCE.reducedMotionSpeed : EXPERIENCE.ventMasaApproachSpeed;
    const prior = motion.masaVentVelocity, desired = Math.min(speed, Math.sqrt(2 * EXPERIENCE.crewAcceleration * remaining));
    motion.masaVentVelocity = transitionVelocity(prior, desired, EXPERIENCE.crewAcceleration, dt);
    motion.masaVentDistance = Math.min(motion.masaVentLength, motion.masaVentDistance + (prior + motion.masaVentVelocity) * .5 * dt);
    const masa = pointAlong(motion.masaVentRoute, motion.masaVentDistance);
    placeCrew(state.crewPosition, masa, state.movingDirection.hadi, facingAlong(motion.masaVentRoute, motion.masaVentDistance), false);
    followCrew(motion, dt, 0);
    if (motion.masaVentLength - motion.masaVentDistance < .8) {
      placeCrew(state.crewPosition, VENT_APPROACHES.shields, state.movingDirection.hadi, facingAlong(motion.masaVentRoute, motion.masaVentLength), false);
      startVentTunnel(motion);
    }
  } else if (motion.phase === 'ventMasaEntry') {
    followCrew(motion, dt, 0);
    if (motion.elapsed >= EXPERIENCE.ventEnterDuration * reducedScale / 1000) {
      motion.phase = 'ventTunnel'; motion.elapsed = 0; motion.tunnelCameraStart = { ...state.cameraPosition }; motion.tunnelZoomStart = state.cameraZoom;
      motion.tunnelCameraEnd = focusCameraPosition(VENTS.navigation, focusZoom());
      ventOverlay.classList.add('is-in-vent'); ventMarkerA.classList.remove('is-active'); ventMarkerB.classList.add('is-active');
      makeVentTransferPath(); setTravelMessage('Short website-only vent passage · the research simulator does not use vents.');
    }
  } else if (motion.phase === 'ventTunnel') {
    const duration = EXPERIENCE.ventTunnelDuration * reducedScale / 1000, t = easeInOut(clamp(motion.elapsed / duration, 0, 1));
    state.cameraPosition = clampCameraPosition({
      x: motion.tunnelCameraStart.x + (motion.tunnelCameraEnd.x - motion.tunnelCameraStart.x) * t,
      y: motion.tunnelCameraStart.y + (motion.tunnelCameraEnd.y - motion.tunnelCameraStart.y) * t
    }, state.cameraZoom);
    state.cameraZoom = motion.tunnelZoomStart + (focusZoom() - motion.tunnelZoomStart) * t;
    if (motion.elapsed >= duration) showVentExit(motion);
  } else if (motion.phase === 'ventHadiExit') {
    followCrew(motion, dt, 0); const duration = EXPERIENCE.ventExitDuration * reducedScale / 1000;
    if (motion.elapsed >= duration * .48 && crewEls.masa.root.classList.contains('is-venting')) {
      crewEls.masa.root.classList.remove('is-venting'); crewEls.masa.root.classList.add('is-emerging');
    }
    if (motion.elapsed >= duration) startWalkingAfterVent(motion);
  } else if (motion.phase === 'settle') {
    const duration = (state.reducedMotion ? EXPERIENCE.reducedMotionCameraDuration : EXPERIENCE.cameraSettleDuration) / 1000;
    const t = easeInOut(clamp(motion.elapsed / duration, 0, 1));
    state.cameraPosition = clampCameraPosition({
      x: motion.settleStart.x + (motion.settleTarget.x - motion.settleStart.x) * t,
      y: motion.settleStart.y + (motion.settleTarget.y - motion.settleStart.y) * t
    }, motion.settleZoom);
    state.cameraZoom = motion.settleZoomStart + (motion.settleZoom - motion.settleZoomStart) * t;
    if (motion.elapsed >= duration) completeArrival(motion);
  } else if (motion.phase === 'failed') { state.transitionState = 'idle'; state.motion = null; clearRouteArt(); }
}
function beginSettle(motion) {
  motion.phase = 'settle'; motion.elapsed = 0; motion.settleStart = { ...state.cameraPosition }; motion.settleZoomStart = state.cameraZoom;
  activePath.classList.remove('is-moving'); activePath.classList.add('is-arrived'); activeShadow.classList.remove('is-moving');
  setSpriteDirection('hadi', state.movingDirection.hadi, false); setSpriteDirection('masa', state.movingDirection.masa, false);
  motion.settleTarget = state.cameraMode === 'overview' ? WORLD.center : focusCameraPosition(missionPosition(motion.mission), focusZoom());
  motion.settleZoom = state.cameraMode === 'overview' ? 1 : focusZoom();
}
function completeArrival(motion) {
  if (state.motion !== motion || state.targetDestination?.id !== motion.mission.id) return;
  const mission = motion.mission;
  state.crewPosition = { ...missionPosition(mission) }; state.transitionState = 'arrived'; state.motion = null;
  state.targetDestination = mission; state.movementProgress = 1; state.lastFrame = 0;
  setSpriteDirection('hadi', state.movingDirection.hadi, false); setSpriteDirection('masa', state.movingDirection.masa, false);
  activePath.classList.remove('is-moving'); activePath.classList.add('is-arrived'); activeShadow.classList.remove('is-moving');
  if (mission.id !== 'parked') state.lastChronologicalMission = mission.id;
  setTravelMessage(`Arrived at ${mission.location.toLowerCase()}.`); showMilestone(mission, motion.trigger);
}
function updateCameraTween(motion, dt) {
  motion.elapsed += dt; const t = easeInOut(clamp(motion.elapsed / motion.duration, 0, 1));
  state.cameraPosition = clampCameraPosition({
    x: motion.startPosition.x + (motion.position.x - motion.startPosition.x) * t,
    y: motion.startPosition.y + (motion.position.y - motion.startPosition.y) * t
  }, state.cameraZoom);
  state.cameraZoom = motion.startZoom + (motion.zoom - motion.startZoom) * t;
  if (motion.elapsed >= motion.duration) {
    state.cameraPosition = clampCameraPosition(motion.position, motion.zoom); state.cameraZoom = motion.zoom;
    state.transitionState = 'idle'; state.motion = null; motion.done?.();
  }
}
function animationFrame(now) {
  rafId = 0; if (!state.motion) return;
  const dt = state.lastFrame ? Math.min(.05, Math.max(.001, (now - state.lastFrame) / 1000)) : 0;
  state.lastFrame = now; const motion = state.motion;
  if (motion.kind === 'camera') updateCameraTween(motion, dt); else updateTravel(motion, dt);
  applyWorldTransform(); updateMarkers(); if (DEBUG_NAV_GRAPH) updateDebugBounds();
  if (state.motion) rafId = requestAnimationFrame(animationFrame); else { rafId = 0; state.lastFrame = 0; }
}
function animateCameraTo(position, zoom, duration, done = () => {}) {
  stopMotion(); state.transitionState = 'camera'; state.targetCameraZoom = zoom;
  const milliseconds = state.reducedMotion ? EXPERIENCE.reducedMotionCameraDuration : duration;
  if (milliseconds < 2) {
    state.cameraPosition = clampCameraPosition(position, zoom); state.cameraZoom = zoom;
    state.transitionState = 'idle'; applyWorldTransform(); updateMarkers(); done(); return;
  }
  state.motion = { kind: 'camera', startPosition: { ...state.cameraPosition }, startZoom: state.cameraZoom,
    position: clampCameraPosition(position, zoom), zoom, duration: milliseconds / 1000, elapsed: 0, done };
  state.lastFrame = 0; scheduleFrame();
}
function showMilestone(mission, trigger) {
  if (!mission) return;
  state.openMilestone = mission.id; state.transitionState = 'arrived'; $('#panelLocation').textContent = mission.location;
  $('#panelIndex').textContent = mission.id === 'parked' ? 'PARKED BRANCH' : `MISSION ${mission.number}`;
  const media = mission.screenshot ? `<figure class="panel-media"><img src="${escapeHTML(mission.screenshot)}" alt="${escapeHTML(mission.imageAlt || '')}" loading="eager"><figcaption>${escapeHTML(mission.imageCaption || '')}</figcaption></figure>` : '';
  const extraEvidence = mission.extraScreenshot ? `<figure class="panel-media"><img src="${escapeHTML(mission.extraScreenshot)}" alt="${escapeHTML(mission.extraImageAlt || '')}" loading="lazy"><figcaption>${escapeHTML(mission.extraImageCaption || '')}</figcaption></figure>` : '';
  const charts = (mission.evidence || []).map(chart => `<details class="panel-chart"><summary>${escapeHTML(chart.title)}</summary><figure class="panel-media"><a href="${escapeHTML(chart.path)}" target="_blank" rel="noopener"><img src="${escapeHTML(chart.path)}" alt="${escapeHTML(chart.title)}" loading="lazy"></a><figcaption>${escapeHTML(chart.caption)}</figcaption></figure></details>`).join('');
  const tags = mission.tags?.length ? `<ul class="panel-note-tags">${mission.tags.map(tag => `<li>${escapeHTML(tag)}</li>`).join('')}</ul>` : '';
  const repairs = mission.repairs?.length ? mission.repairs.map(repair => `<div class="panel-repair"><b>ISSUE FOUND · ${escapeHTML(repair.title)}</b><span>FIX APPLIED · ${escapeHTML(repair.fix)}</span></div>`).join('') : '';
  const notes = mission.notes || mission.detail;
  const explanation = mission.detail && mission.detail !== notes ? `<p>${escapeHTML(mission.detail)}</p>` : '';
  const statusLine = mission.id === 'parked' ? `PARKED BRANCH · ${STATUS_LABEL.parked}` : `MISSION ${mission.number} · ${mission.originalMilestone} · ${STATUS_LABEL[mission.status]}`;
  $('#terminalContent').innerHTML = `
    <div class="panel-status" data-status="${mission.status}"><i aria-hidden="true"></i><span>${escapeHTML(statusLine)}</span><span class="panel-date">${escapeHTML(mission.date)}</span></div>
    <h2 id="panelTitle">${escapeHTML(mission.title)}</h2>
    <p class="panel-summary">${escapeHTML(mission.summary)}</p>
    <div class="panel-stat"><strong>${escapeHTML(mission.metric)}</strong><span>${escapeHTML(mission.metricLabel)}</span></div>
    ${media}<details class="panel-notes"><summary>OPEN THE MISSION LOG</summary>${explanation}<p>${escapeHTML(notes)}</p>${tags}${extraEvidence}${charts}${repairs}</details>`;
  if (!milestoneDialog.open) milestoneDialog.showModal();
  $('#closeMilestone').focus({ preventScroll: true });
  setTravelMessage(mission.id === 'parked' ? 'Parked off the main research route.' : `Mission ${mission.number} · ${STATUS_LABEL[mission.status]}.`);
  markerEls.get(mission.id)?.classList.add('is-selected'); if (trigger) trigger.setAttribute('aria-current', 'step');
}
function travelTo(missionId, trigger = null) {
  const mission = DESTINATIONS.find(item => item.id === missionId);
  if (!mission || milestoneDialog.open || logDialog.open) return;
  if (state.targetDestination?.id === mission.id && state.transitionState === 'walking') return;
  stopMotion(); milestoneReturnTarget = trigger || trailEls.get(mission.id) || markerEls.get(mission.id);
  state.targetDestination = mission; state.cameraMode = 'follow'; state.targetCameraZoom = focusZoom();
  state.transitionState = 'walking'; state.openMilestone = null; state.movementProgress = 0;
  for (const button of markerEls.values()) button.classList.remove('is-selected');
  for (const button of trailEls.values()) { button.classList.remove('is-selected'); button.removeAttribute('aria-current'); }
  markerLayer.querySelectorAll('[aria-current]').forEach(button => button.removeAttribute('aria-current'));
  markerEls.get(mission.id)?.classList.add('is-selected'); trailEls.get(mission.id)?.classList.add('is-selected');
  const useVentStory = mission.id === 'mission11' && state.lastChronologicalMission === 'mission10' && !state.ventStoryUsed;
  const route = findPhysicalRoute({ ...state.crewPosition }, useVentStory ? VENT_APPROACHES.shields : missionPosition(mission));
  if (!route || route.length < 2) { state.transitionState = 'idle'; setTravelMessage('That location could not be connected to a clear route.'); return; }
  const motion = { kind: 'travel', mission, trigger, phase: useVentStory ? 'walkToVent' : 'walkToMission', elapsed: 0 };
  state.motion = motion;
  if (!prepareWalk(motion, route, motion.phase)) {
    state.motion = null; state.transitionState = 'idle'; setTravelMessage('The crew could not connect to the ship route.'); return;
  }
  setTravelMessage(useVentStory ? 'Mission 11 · follow the corridor to one playful ship vent.' : `Mission ${mission.number} · Hadi and Masa are on their way.`);
  state.lastFrame = 0; scheduleFrame();
}
function viewMap() {
  if (milestoneDialog.open || logDialog.open) return;
  state.cameraMode = 'overview'; state.targetCameraZoom = 1;
  setTravelMessage(state.transitionState === 'walking' ? 'Ship overview · Hadi and Masa are still moving.' : 'Ship overview · choose any mission along the journey.');
  if (state.transitionState === 'walking') return;
  state.targetDestination = null;
  for (const button of markerEls.values()) button.classList.remove('is-selected');
  for (const button of trailEls.values()) button.classList.remove('is-selected');
  animateCameraTo(WORLD.center, 1, EXPERIENCE.cameraZoomDuration);
}
function openLog() {
  if (!milestoneDialog.open && !logDialog.open) { logReturnTarget = $('#logButton'); logDialog.showModal(); $('#closeLog').focus({ preventScroll: true }); }
}
function closeDialog(dialog) { if (dialog.open) dialog.close(); }
function toast(message) {
  const element = $('#toast'); element.textContent = message; element.classList.add('is-visible');
  window.clearTimeout(toastTimer); toastTimer = window.setTimeout(() => element.classList.remove('is-visible'), 2500);
}
const intro = 'We are building an Among Us-inspired social-deduction AI in a research simulator. One learned crewmate now chooses actions using its observations, memory and a frozen suspicion model. We are testing whether it contributes useful tasks and votes and improves crew outcomes against scripted opponents. Adapting opponents and self-play come only after that single-crewmate test succeeds.';
function zoomAt(clientX, clientY, factor) {
  if (state.motion || milestoneDialog.open || logDialog.open) return;
  const oldScale = cameraScale(), targetZoom = clamp(state.cameraZoom * factor, .82, 3.25);
  if (Math.abs(targetZoom - state.cameraZoom) < .01) return;
  const rect = viewportEl.getBoundingClientRect(), sx = clientX - rect.left, sy = clientY - rect.top;
  const worldPoint = { x: state.cameraPosition.x + (sx - state.viewport.center.x) / oldScale,
    y: state.cameraPosition.y + (sy - state.viewport.center.y) / oldScale };
  const newScale = cameraScale(targetZoom);
  const position = { x: worldPoint.x - (sx - state.viewport.center.x) / newScale,
    y: worldPoint.y - (sy - state.viewport.center.y) / newScale };
  state.cameraMode = 'manual'; animateCameraTo(position, targetZoom, 230);
}
function isNonMapControl(target) {
  return !!target.closest('button, a, dialog, input, textarea, select, summary, .team-key, .map-legend, .opening-caption, .ship-caption, .travel-status, .toast');
}
function addEventListeners() {
  markerLayer.addEventListener('click', event => { const button = event.target.closest('[data-milestone]'); if (button) travelTo(button.dataset.milestone, button); });
  trailScroll.addEventListener('click', event => { const button = event.target.closest('[data-milestone]'); if (button) travelTo(button.dataset.milestone, button); });
  $('#mapButton').addEventListener('click', viewMap); $('#logButton').addEventListener('click', openLog);
  $('#closeMilestone').addEventListener('click', () => closeDialog(milestoneDialog)); $('#closeLog').addEventListener('click', () => closeDialog(logDialog));
  $('#shareIntro').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(intro); toast('A short project introduction is copied and ready to share.'); }
    catch { toast(intro); }
  });
  milestoneDialog.addEventListener('close', () => {
    if (state.transitionState === 'arrived') state.transitionState = 'idle';
    state.openMilestone = null; setTravelMessage('Hadi and Masa are still here. Choose another mission.');
    const returnTarget = milestoneReturnTarget; milestoneReturnTarget = null;
    requestAnimationFrame(() => { if (returnTarget?.isConnected && !returnTarget.hidden) returnTarget.focus({ preventScroll: true }); });
  });
  milestoneDialog.addEventListener('click', event => { if (event.target === milestoneDialog) closeDialog(milestoneDialog); });
  logDialog.addEventListener('click', event => { if (event.target === logDialog) closeDialog(logDialog); });
  logDialog.addEventListener('close', () => requestAnimationFrame(() => { if (logReturnTarget?.isConnected) logReturnTarget.focus({ preventScroll: true }); }));
  milestoneDialog.addEventListener('cancel', () => { state.openMilestone = null; }); logDialog.addEventListener('cancel', () => {});
  window.addEventListener('resize', resizeCamera, { passive: true });
  reducedMotionQuery.addEventListener?.('change', event => {
    state.reducedMotion = event.matches; if (state.reducedMotion) stopShootingStars(); else startShootingStars();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopShootingStars(); else if (!state.reducedMotion) startShootingStars(); state.lastFrame = 0;
  });
  document.addEventListener('keydown', event => {
    if (event.key.toLowerCase() === 'm' && !event.altKey && !event.ctrlKey && !event.metaKey && !milestoneDialog.open && !logDialog.open) viewMap();
  });
  viewportEl.addEventListener('pointerdown', event => {
    if (state.motion || milestoneDialog.open || logDialog.open || isNonMapControl(event.target) || event.button > 0) return;
    pointerPan = { id: event.pointerId, startX: event.clientX, startY: event.clientY, lastX: event.clientX, lastY: event.clientY, dragging: false };
  });
  viewportEl.addEventListener('pointermove', event => {
    if (!pointerPan || pointerPan.id !== event.pointerId || state.motion) return;
    if (!pointerPan.dragging && Math.hypot(event.clientX - pointerPan.startX, event.clientY - pointerPan.startY) < EXPERIENCE.dragThreshold) return;
    if (!pointerPan.dragging) {
      pointerPan.dragging = true; viewportEl.classList.add('is-dragging');
      try { viewportEl.setPointerCapture(event.pointerId); } catch {}
    }
    const dx = event.clientX - pointerPan.lastX, dy = event.clientY - pointerPan.lastY;
    pointerPan.lastX = event.clientX; pointerPan.lastY = event.clientY;
    const scale = Math.max(.001, cameraScale()); state.cameraMode = 'manual';
    state.cameraPosition = clampCameraPosition({ x: state.cameraPosition.x - dx / scale, y: state.cameraPosition.y - dy / scale });
    applyWorldTransform(); updateMarkers(); event.preventDefault();
  });
  const endPan = event => {
    if (!pointerPan || pointerPan.id !== event.pointerId) return;
    const dragged = pointerPan.dragging; pointerPan = null; viewportEl.classList.remove('is-dragging');
    if (viewportEl.hasPointerCapture?.(event.pointerId)) viewportEl.releasePointerCapture(event.pointerId);
    if (dragged) event.preventDefault();
  };
  viewportEl.addEventListener('pointerup', endPan); viewportEl.addEventListener('pointercancel', endPan);
  viewportEl.addEventListener('wheel', event => {
    if ((!event.ctrlKey && !event.metaKey) || isNonMapControl(event.target)) return;
    event.preventDefault(); zoomAt(event.clientX, event.clientY, Math.exp(-event.deltaY * .002));
  }, { passive: false });
}
function startShootingStars() {
  if (state.reducedMotion || document.hidden || !shootingStar || shootingStarTimer) return;
  const wait = EXPERIENCE.shootingStarMinDelay + Math.random() * (EXPERIENCE.shootingStarMaxDelay - EXPERIENCE.shootingStarMinDelay);
  shootingStarTimer = window.setTimeout(() => {
    shootingStarTimer = 0;
    if (state.reducedMotion || document.hidden) { startShootingStars(); return; }
    const between = (min, max) => min + Math.random() * (max - min);
    const angle = between(EXPERIENCE.shootingStarMinAngle, EXPERIENCE.shootingStarMaxAngle);
    const brightness = between(EXPERIENCE.shootingStarMinBrightness, EXPERIENCE.shootingStarMaxBrightness);
    const travel = between(EXPERIENCE.shootingStarMinTravel, EXPERIENCE.shootingStarMaxTravel);
    const drift = between(EXPERIENCE.shootingStarMinDrift, EXPERIENCE.shootingStarMaxDrift);
    const length = between(EXPERIENCE.shootingStarMinLength, EXPERIENCE.shootingStarMaxLength);
    const duration = EXPERIENCE.shootingStarDuration * between(1 - EXPERIENCE.shootingStarDurationJitter, 1 + EXPERIENCE.shootingStarDurationJitter);
    shootingStar.style.left = `${between(8, 74)}vw`; shootingStar.style.top = `${between(5, 52)}vh`;
    shootingStar.style.width = `${length.toFixed(0)}px`;
    const streak = (x, y, scale) => `translate3d(${x}vw,${y}vh,0) rotate(${angle.toFixed(1)}deg) scaleX(${scale})`;
    shootingStarAnimation = shootingStar.animate([
      { opacity: 0, transform: streak(0, 0, .5) },
      { opacity: brightness, offset: .16, transform: streak(travel * .12, drift * .12, 1) },
      { opacity: 0, transform: streak(travel, drift, .74) }
    ], { duration, easing: 'cubic-bezier(.2,.6,.4,1)', fill: 'none' });
    shootingStarAnimation.onfinish = () => { shootingStarAnimation = null; startShootingStars(); };
  }, wait);
}
function stopShootingStars() {
  if (shootingStarTimer) window.clearTimeout(shootingStarTimer); shootingStarTimer = 0;
  if (shootingStarAnimation) shootingStarAnimation.cancel(); shootingStarAnimation = null;
}
function renderDebugGraph() {
  if (!DEBUG_NAV || hasDrawnDebug) return;
  hasDrawnDebug = true; debugLayer.hidden = false;
  const edgeParts = [], nodeParts = [];
  for (let index = 0; index < GRID_SIZE; index++) {
    if (!bitAt(NAV_NODE_BITS, index)) continue;
    const point = gridPoint(index); nodeParts.push(`M ${(point.x - 3).toFixed(1)} ${point.y.toFixed(1)} a 3 3 0 1 0 6 0 a 3 3 0 1 0 -6 0`);
    const bits = NAV_EDGE_MASKS[index], row = Math.floor(index / GRID.cols), col = index - row * GRID.cols;
    for (let direction = 0; direction < 8; direction++) {
      if (!(bits & (1 << direction))) continue;
      const delta = NAV_DIRECTIONS[direction], next = (row + delta.y) * GRID.cols + col + delta.x;
      if (next <= index) continue;
      const other = gridPoint(next); edgeParts.push(`M ${point.x.toFixed(1)} ${point.y.toFixed(1)} L ${other.x.toFixed(1)} ${other.y.toFixed(1)}`);
    }
  }
  debugLayer.querySelector('.route-debug-edges').setAttribute('d', edgeParts.join(' '));
  debugLayer.querySelector('.route-debug-nodes').setAttribute('d', nodeParts.join(' '));
  debugLayer.classList.add('is-visible'); updateDebugBounds();
  console.info(`Skeld route debug: ${NAV_DATA.counts.navVertices} walkable waypoints, ${NAV_DATA.counts.navEdges} verified edges.`);
}
function updateDebugBounds() {
  if (!DEBUG_NAV_GRAPH || debugLayer.hidden || !state.viewport.width) return;
  const rect = debugLayer.querySelector('.route-debug-bounds'), scale = Math.max(.001, cameraScale());
  rect.setAttribute('x', state.cameraPosition.x - state.viewport.width / (2 * scale));
  rect.setAttribute('y', state.cameraPosition.y - state.viewport.height / (2 * scale));
  rect.setAttribute('width', state.viewport.width / scale); rect.setAttribute('height', state.viewport.height / scale);
}
function setupDeveloperDebug() {
  if (!DEBUG_NAV) return;
  if (DEBUG_NAV_GRAPH) renderDebugGraph();
  window.__CREWMATE_DEBUG__ = Object.freeze({
    graph: NAV_DATA.counts, experience: EXPERIENCE, state,
    route: (from, to) => {
      const a = DESTINATIONS.find(item => item.id === from) || MILESTONES.find(item => item.number === from);
      const b = DESTINATIONS.find(item => item.id === to) || MILESTONES.find(item => item.number === to);
      if (!a || !b) return null;
      const points = findPhysicalRoute(missionPosition(a), missionPosition(b));
      return points ? { points, length: pathLength(points), clear: points.slice(1).every((p, i) => clearLineWorld(points[i], p)) } : null;
    },
    isSafe: isSafeWorldPoint, isSafeGame: safeGamePoint, clearSegment: clearLineWorld, attachments: navAttachCandidates,
    findRoute: findPhysicalRoute, gamePoint: mapPixelToGame, grid: GRID
  });
}
function setStartingSprite(key, direction = 'down') {
  const cfg = TEAM[key], spriteSet = SPRITES[cfg.color];
  crewEls[key].image.src = COLORED_SPRITES.get(`${cfg.color}/${direction}/1`);
  crewEls[key].root.dataset.color = cfg.color; crewEls[key].root.dataset.facing = direction; crewEls[key].root.classList.add('is-idle');
}
function initialize() {
  renderNavigation(); addEventListeners(); setStartingSprite('hadi'); setStartingSprite('masa');
  placeCrew(TEAM.hadi.start, TEAM.masa.start, 'down', 'down', false); resizeCamera();
  const mobile = matchMedia('(max-width: 560px)').matches; state.cameraMode = mobile ? 'follow' : 'overview';
  state.cameraPosition = mobile ? focusCameraPosition(TEAM.hadi.start, focusZoom()) : { ...WORLD.center };
  state.cameraZoom = mobile ? focusZoom() : 1; state.targetCameraZoom = state.cameraZoom;
  applyWorldTransform(); updateMarkers(); setupDeveloperDebug();
  shipImage.addEventListener('error', () => { shipImage.src = 'assets/skeld-map-overview.png'; }, { once: true });
  if (!MILESTONES.length) setTravelMessage('The mission log is empty.'); startShootingStars();
}
function applyProjectStatus(projectStatus) {
  const phaseById = new Map(projectStatus.roadmap.map(phase => [phase.id, phase]));
  if (projectStatus.project !== 'Hadisovic/social-deduction-ai' || !phaseById.has(`phase-${projectStatus.currentPhase}`) || !phaseById.has(`phase-${projectStatus.nextPhase}`)) {
    throw new Error('The public project status record is incomplete or belongs to another project.');
  }
  for (const mission of MILESTONES) {
    if (!mission.phaseStatusId) continue;
    const phase = phaseById.get(mission.phaseStatusId);
    if (!phase || !['complete', 'in-progress', 'next', 'future'].includes(phase.status)) throw new Error(`Invalid canonical status for ${mission.phaseStatusId}.`);
    mission.status = phase.status;
  }
  if (projectStatus.phase5Study) {
    const study = projectStatus.phase5Study;
    const mission = MILESTONES.find(item => item.id === 'mission11');
    mission.metric = study.metric; mission.metricLabel = study.metricLabel;
    mission.notes = study.notes; mission.tags = study.tags; mission.evidence = study.figures;
    mission.date = study.date;
  }
  const completed = projectStatus.roadmap.filter(phase => phase.status === 'complete').length;
  $('#phaseCompletion').textContent = `${completed} / ${projectStatus.roadmap.length}`;
  $('#logCurrentLabel').textContent = `RIGHT NOW · PHASE ${projectStatus.currentPhase} · ${STATUS_LABEL[phaseById.get(`phase-${projectStatus.currentPhase}`).status]}`;
  $('#nextTitle').textContent = `Future research · Phase ${projectStatus.nextPhase}`;
  $('#projectLastUpdated').dateTime = projectStatus.lastUpdated;
  $('#projectLastUpdated').textContent = projectStatus.lastUpdated;
  $('#projectVerifiedCommit').textContent = projectStatus.lastVerifiedCommit.slice(0, 7);
}
fetch('./project-status.json', { cache: 'no-store' })
  .then(response => {
    if (!response.ok) throw new Error(`Public project status could not be loaded (${response.status}).`);
    return response.json();
  })
  .then(async projectStatus => { applyProjectStatus(projectStatus); await prepareSprites(); initialize(); })
  .catch(error => {
    console.error('The project status file is required for this site.', error);
    setTravelMessage('Project status did not load. Please refresh the ship log.');
  });
