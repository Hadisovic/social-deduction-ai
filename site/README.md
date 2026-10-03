# Interactive project journey

`site/` contains the standalone, static Skeld-style project story. It presents the research project; it does not run Python or import the simulator. The public experience uses the map, collaborator sprites, mission data in `app.js`, and bundled navigation geometry in `navigation-data.js`.

## Preview and validate

Node.js 20 or newer is used only to assemble and check a local preview. No package install is required. From the repository root, run:

```powershell
$preview = Join-Path $env:TEMP 'social-deduction-ai-site'
node site/scripts/validate-site.mjs --base-path /social-deduction-ai/
node site/scripts/build-site.mjs $preview
node site/scripts/serve-preview.mjs $preview
```

Open `http://127.0.0.1:4173/social-deduction-ai/`. This lightweight server maps the built files beneath the real project Pages path; the validator also checks every local resource against that base path and checks exact filename case. Browser behavior can be checked with `?debugNav=1`; add `&graph=1` to show the dense walkable graph.

## Updating project facts

`project-status.json` at the repository root is the canonical source for the current phase, next phase, roadmap statuses, headline results, and the commit/date used to verify them. The build copies that file into the Pages artifact, and the site reads it at runtime. `HANDOFF.md` remains the human-readable engineering handoff; update both it and the public status when a phase closes.

The ordered mission stories, summaries, screenshots, and repairs live in the `MILESTONES` array near the top of `app.js`. Phase entries point to their status IDs in `project-status.json`; leave the IDs aligned. Add new public screenshots under `assets/` and reference them with case-exact, relative paths. Do not add checkpoints, datasets, logs, credentials, environment files, or machine-specific paths.

When Phase 5 is complete:

1. Update `HANDOFF.md` and the relevant research/benchmark documentation with verified results.
2. Update root `project-status.json`: set the current and next phase, roadmap statuses, results, `lastVerifiedCommit`, and `lastUpdated`.
3. Update Mission 11's date, summary, evidence, result, and limitations in `app.js`; update the flight-recorder text in `index.html` if it is no longer accurate.
4. Add only public screenshots needed to support the story and advance the following mission from FUTURE to NEXT when appropriate.
5. Run `node site/scripts/validate-site.mjs --base-path /social-deduction-ai/`, build the preview, and check the page at desktop and mobile sizes.
6. Open a pull request. After review and merge to `main`, the Pages workflow validates and deploys the site automatically.

Hadi's red and Masa's black character colors are configured by the `TEAM` and `TEAM_COLORS` objects in `app.js`; the matching poses are in `assets/world/team/`. They represent project collaborators, not agents used in the experiments.

## Publication flow

Pull requests run `site-checks.yml` only: status, mission, resource-path, case-sensitivity, and secret checks plus a static build. They do not publish the PR to production. A push to `main` or a manual workflow dispatch on `main` runs `pages.yml`, which builds `site/`, copies the canonical status JSON, uploads the artifact, and deploys it with GitHub's Pages actions. The root README remains the repository's research overview and links to the public journey.
