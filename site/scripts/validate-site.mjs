import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const siteDir = path.resolve(scriptDir, '..');
const repoDir = path.resolve(siteDir, '..');
const args = process.argv.slice(2);
const option = name => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const basePath = option('--base-path') ?? '/social-deduction-ai/';
const publicRoot = path.resolve(option('--root') ?? siteDir);
if (!basePath.startsWith('/') || !basePath.endsWith('/') || basePath.includes('..')) {
  throw new Error('--base-path must begin and end with / and cannot contain ..');
}

const readText = file => readFile(file, 'utf8');
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const fileExists = async file => { try { return (await stat(file)).isFile(); } catch { return false; } };
const statusCandidates = [path.join(publicRoot, 'project-status.json'), path.join(repoDir, 'project-status.json')];
let statusFile;
for (const candidate of statusCandidates) if (await fileExists(candidate)) { statusFile = candidate; break; }
assert(statusFile, 'Canonical project-status.json is missing.');
const status = JSON.parse(await readText(statusFile));
const html = await readText(path.join(publicRoot, 'index.html'));
const css = await readText(path.join(publicRoot, 'styles.css'));
const app = await readText(path.join(publicRoot, 'app.js'));
const nav = await readText(path.join(publicRoot, 'navigation-data.js'));
const navAudit = JSON.parse(await readText(path.join(publicRoot, 'navigation-audit.json')));

assert(status.schemaVersion === 1, 'Unsupported project-status.json schemaVersion.');
assert(status.project === 'Hadisovic/social-deduction-ai', 'Project status points to the wrong repository.');
assert(/^\d{4}-\d{2}-\d{2}$/.test(status.lastUpdated), 'lastUpdated must be an ISO date.');
assert(/^[0-9a-f]{40}$/i.test(status.lastVerifiedCommit), 'lastVerifiedCommit must be a full Git SHA.');
assert(status.currentPhase === 4 && status.nextPhase === 5, 'Public phase numbers must remain Phase 4 complete and Phase 5 next.');

const expectedRoadmap = [
  ['phase-1', 'complete'], ['phase-1.5', 'complete'], ['phase-2', 'complete'],
  ['phase-3', 'complete'], ['phase-4', 'complete'], ['phase-5', 'next'], ['phase-6', 'future']
];
assert(Array.isArray(status.roadmap) && status.roadmap.length === expectedRoadmap.length, 'Roadmap must contain exactly the seven tracked phases.');
const phaseIds = new Set();
for (const [index, [id, expectedStatus]] of expectedRoadmap.entries()) {
  const phase = status.roadmap[index];
  assert(phase.id === id && phase.status === expectedStatus && typeof phase.name === 'string' && phase.name.length > 0,
    `Roadmap entry ${index + 1} must be ${id} / ${expectedStatus}.`);
  assert(!phaseIds.has(phase.id), `Duplicate roadmap phase id: ${phase.id}.`);
  phaseIds.add(phase.id);
}
assert(status.roadmap.filter(phase => phase.status === 'next').length === 1, 'Exactly one phase must be NEXT.');
assert(status.roadmap.find(phase => phase.id === `phase-${status.currentPhase}`)?.status === 'complete', 'The current phase must be complete.');
assert(status.roadmap.find(phase => phase.id === `phase-${status.nextPhase}`)?.status === 'next', 'The next phase must be marked NEXT.');
assert(status.currentPhaseName === status.roadmap.find(phase => phase.id === `phase-${status.currentPhase}`).name, 'currentPhaseName must match the roadmap.');
assert(status.nextPhaseName === status.roadmap.find(phase => phase.id === `phase-${status.nextPhase}`).name, 'nextPhaseName must match the roadmap.');

const phase2 = status.headlineResults?.phase2;
const phase3 = status.headlineResults?.phase3;
const phase4 = status.headlineResults?.phase4;
assert(phase2?.directedDestinationPairsPassed === 3906 && phase2?.randomSpawnsPassed === 1008 && phase2?.combinedSuccessRatePercent === 100,
  'Phase 2 headline results do not match the verified benchmark.');
assert(phase2.collisions === 0 && phase2.blockedSteps === 0 && phase2.replans === 0 && phase2.physicalExecutionSteps === 1535435,
  'Phase 2 movement-safety figures do not match the benchmark.');
assert(phase3?.scriptedMatchesCompleted === 1000 && phase3?.exactIndependentReplays === 1000,
  'Phase 3 headline results do not match the acceptance report.');
assert(phase4?.scriptedMatches === 1400 && phase4?.samples === 64097 && phase4?.inDistributionAccuracyPercent === 70.54 &&
  phase4?.heldOutPatientFamilyAccuracyPercent === 67.41 && phase4?.testsPassed === 327,
  'Phase 4 headline results do not match the checked-in benchmark.');

const missionIds = [...app.matchAll(/\bid:\s*'(mission\d+)'/g)].map(match => match[1]);
assert(missionIds.length === 12 && new Set(missionIds).size === missionIds.length, 'Mission IDs must be unique and include all 12 story stops.');
const phaseKeys = [...app.matchAll(/\bphaseStatusId:\s*'(phase-[\d.]+)'/g)].map(match => match[1]);
assert(phaseKeys.length === expectedRoadmap.length && new Set(phaseKeys).size === phaseKeys.length && phaseKeys.every(id => phaseIds.has(id)),
  'Every canonical phase must map to exactly one website mission.');
assert(app.includes("fetch('./project-status.json'") && app.includes('mission.status = phase.status'), 'The experience must read canonical phase status at runtime.');
assert(html.includes('id="phaseCompletion"') && html.includes('id="projectVerifiedCommit"') && html.includes('VIEW RESEARCH REPOSITORY'),
  'The site must show status progress and a subtle repository link.');

const comma = value => Number(value).toLocaleString('en-US');
for (const claim of [
  `${comma(phase2.directedDestinationPairsPassed)} DESTINATION PAIRS`, `${comma(phase2.randomSpawnsPassed)} RANDOM STARTS`,
  `${comma(phase3.scriptedMatchesCompleted)} + ${comma(phase3.exactIndependentReplays)}`,
  `${comma(phase4.scriptedMatches)} SCRIPTED MATCHES`, `${comma(phase4.samples)} SAMPLES`,
  `${phase4.heldOutPatientFamilyAccuracyPercent.toFixed(2)}%`, `${phase4.inDistributionAccuracyPercent.toFixed(2)}%`,
  `${comma(phase4.testsPassed)} CHECKS PASS`
]) assert(app.includes(claim), `Public mission copy is missing verified claim: ${claim}.`);
assert(/structured information from its simulated crewmate/i.test(html) && /does not visually read the commercial game/i.test(html),
  'The landing page must keep the simulator information boundary clear.');
assert(/scripted players/i.test(app) && /not evidence of human-level social play/i.test(app), 'Phase 3 must remain clearly described as scripted.');
assert(/This is a belief estimate, not an action or a vote/i.test(app) && /This work has not started/i.test(html),
  'Phase 4 and Phase 5 limitations must remain explicit.');
assert(/hadi: Object\.freeze\(\{ name: 'Hadi', color: 'red'/.test(app) && /masa: Object\.freeze\(\{ name: 'Masa', color: 'black'/.test(app),
  'The collaborator colors must remain Hadi=red and Masa=black.');
assert(css.includes('@media (prefers-reduced-motion: reduce)') && app.includes("matchMedia('(prefers-reduced-motion: reduce)')"),
  'The site must retain both CSS and JavaScript reduced-motion handling.');
assert(navAudit && typeof navAudit === 'object' && nav.includes('SKELD_NAVIGATION'), 'Navigation data and its audit record must be present.');

const requiredFiles = ['index.html', 'styles.css', 'app.js', 'navigation-data.js', 'navigation-audit.json', '.nojekyll'];
const exactPath = async relative => {
  let current = publicRoot;
  for (const part of relative.split('/').filter(Boolean)) {
    let entries;
    try { entries = await readdir(current); } catch { return false; }
    if (!entries.includes(part)) return false;
    current = path.join(current, part);
  }
  return await fileExists(current) || relative === 'project-status.json';
};
for (const file of requiredFiles) assert(await exactPath(file), `Required site file is missing or case-mismatched: ${file}.`);

const localReferences = new Set();
for (const match of html.matchAll(/\b(?:src|href)="([^"]+)"/g)) localReferences.add(match[1]);
for (const match of css.matchAll(/url\(\s*['"]?([^)'"\s]+)['"]?\s*\)/g)) localReferences.add(match[1]);
for (const match of app.matchAll(/(?:screenshot|extraScreenshot):\s*'([^']+)'/g)) localReferences.add(match[1]);
localReferences.add('project-status.json');
for (const match of app.matchAll(/base:\s*'([^']+)'/g)) {
  const base = match[1];
  const framesMatch = app.slice(match.index, app.indexOf('}),', match.index));
  for (const frame of framesMatch.matchAll(/\b(down|left|right|up):\s*(\d+)/g)) {
    for (let number = 1; number <= Number(frame[2]); number++) localReferences.add(`${base}/${frame[1]}/${String(number).padStart(2, '0')}.png`);
  }
}
localReferences.add('assets/skeld-map-overview.png');
const checkedReferences = [];
for (const reference of localReferences) {
  if (reference.startsWith('#') || /^[a-z][a-z\d+.-]*:/i.test(reference) || reference.startsWith('//')) continue;
  assert(!reference.startsWith('/'), `Root-absolute resource path breaks project Pages: ${reference}.`);
  const resolvedUrl = new URL(reference, `https://pages.invalid${basePath}`);
  const baseUrl = new URL(basePath, 'https://pages.invalid');
  assert(resolvedUrl.pathname.startsWith(baseUrl.pathname), `Resource escapes the Pages base path: ${reference}.`);
  const relativePath = decodeURIComponent(resolvedUrl.pathname.slice(baseUrl.pathname.length));
  const exists = relativePath === 'project-status.json'
    ? await fileExists(path.join(publicRoot, relativePath)) || statusFile === path.join(repoDir, relativePath)
    : await exactPath(relativePath);
  assert(exists, `Referenced resource is missing or case-mismatched under ${basePath}: ${reference}.`);
  checkedReferences.push(reference);
}

const runtimeFiles = ['index.html', 'styles.css', 'app.js', 'navigation-data.js', 'navigation-audit.json'];
for (const relative of runtimeFiles) {
  const source = await readText(path.join(publicRoot, relative));
  assert(!/\b(?:TODO|FIXME|Lorem ipsum|PLACEHOLDER)\b/i.test(source), `Placeholder text remains in ${relative}.`);
  assert(!/https?:\/\/localhost\b/i.test(source), `Localhost reference remains in deployed file ${relative}.`);
  assert(!/(?:["'(])\/(?:assets|styles\.css|app\.js|navigation-data\.js|project-status\.json)(?:\/|[)'"\s])/i.test(source),
    `Root-absolute asset path found in ${relative}.`);
}

const filesUnder = async directory => {
  const collected = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) collected.push(...await filesUnder(full)); else collected.push(full);
  }
  return collected;
};
const publicFiles = await filesUnder(publicRoot);
const forbiddenExtensions = new Set(['.pt', '.pth', '.zip', '.pkl', '.pickle', '.pyc', '.jsonl']);
for (const file of publicFiles) {
  assert(!forbiddenExtensions.has(path.extname(file).toLowerCase()), `Research artifact must not be published: ${path.relative(publicRoot, file)}.`);
  assert(!/^\.env(?:\.|$)/i.test(path.basename(file)), `Environment file must not be published: ${path.relative(publicRoot, file)}.`);
}
const secretPatterns = [
  /\bgh[pousr]_[A-Za-z0-9]{20,}\b/,
  /\bgithub_pat_[A-Za-z0-9_]{20,}\b/,
  /\bAKIA[0-9A-Z]{16}\b/,
  /-----BEGIN (?:RSA|EC|OPENSSH|DSA) PRIVATE KEY-----/,
  /Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+\/=:-]{16,}/i,
  /https?:\/\/[^\s/@:]+:[^\s/@]+@/i
];
const textExtensions = new Set(['.html', '.css', '.js', '.mjs', '.json', '.md', '.yml', '.yaml', '.txt']);
const filesToScan = new Set([
  ...publicFiles.filter(file => textExtensions.has(path.extname(file).toLowerCase())),
  statusFile,
  path.join(repoDir, 'README.md'),
  ...await filesUnder(path.join(repoDir, '.github')).catch(() => [])
]);
for (const file of filesToScan) {
  const text = await readText(file);
  for (const pattern of secretPatterns) assert(!pattern.test(text), `Possible credential pattern found in ${path.relative(repoDir, file)}.`);
}

console.log(`PASS: project status, phase claims, ${missionIds.length} missions, and navigation audit.`);
console.log(`PASS: ${checkedReferences.length} local assets resolve with exact case under ${basePath}.`);
console.log('PASS: static-only publish set, reduced-motion rules, and obvious-secret scan.');
