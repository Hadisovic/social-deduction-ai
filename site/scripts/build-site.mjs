import { cp, copyFile, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const siteDir = path.resolve(scriptDir, '..');
const repoDir = path.resolve(siteDir, '..');
const statusFile = path.join(repoDir, 'project-status.json');
const outputArg = process.argv[2];

if (!outputArg) {
  console.error('Usage: node site/scripts/build-site.mjs <temporary-output-directory>');
  process.exit(2);
}

const outputDir = path.resolve(outputArg);
const allowedTempRoots = [process.env.RUNNER_TEMP, os.tmpdir()].filter(Boolean).map(value => path.resolve(value));
const isWithin = (parent, child) => {
  const relative = path.relative(parent, child);
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
};

if (outputDir === path.parse(outputDir).root || !allowedTempRoots.some(root => isWithin(root, outputDir)) ||
    isWithin(outputDir, repoDir) || isWithin(repoDir, outputDir)) {
  throw new Error('Build output must be a child of the system or runner temporary directory and must not contain the repository.');
}

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await cp(siteDir, outputDir, {
  recursive: true,
  filter(source) {
    const relative = path.relative(siteDir, source).split(path.sep).join('/');
    return relative !== 'README.md' && relative !== 'scripts' && !relative.startsWith('scripts/');
  }
});
await copyFile(statusFile, path.join(outputDir, 'project-status.json'));

console.log(`Built static site at ${outputDir}`);
