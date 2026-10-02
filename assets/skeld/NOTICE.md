# Skeld source provenance

This directory combines third-party reference data with the image supplied by the
project owner for this map-building request. It is not an official Innersloth
map SDK, and no source here establishes exact compatibility with a current client.

## Collider geometry

[SkeldJS/SkeldJS](https://github.com/SkeldJS/SkeldJS/tree/a1acf39d1751a553141cc7153e9386e37199ee2d),
revision `a1acf39d1751a553141cc7153e9386e37199ee2d`:

- `sources/skeld.json` copies `packages/pathfinding/scripts/skeld.json`.
- `sources/TheSkeld.json` copies `packages/pathfinding/data/colliders/TheSkeld.json`.
- The eight explicit repairs in our builder follow that revision's
  `packages/pathfinding/scripts/postprocess.js`.
- Raw named/layered records are the physical input. The processed file is retained
  for comparison, not blindly unioned: source layers distinguish solid objects,
  triggers, area classification, doors and visibility.
- Upstream GPL-3.0 license text is retained as `sources/LICENSE-SkeldJS.txt`.

The raw dump was introduced in September 2021. The client build is not identified.
Repository license notices do not independently license Innersloth game content.

## Tasks, vents, doors, spawns and systems

[Impostor/Impostor](https://github.com/Impostor/Impostor/tree/b09c40b7e12d35c6cae996a26a943185b341c907),
revision `b09c40b7e12d35c6cae996a26a943185b341c907`:

`sources/impostor-{tasks,vents,doors,spawn,systems}.json` copies corresponding files
under `src/Impostor.Api/Innersloth/Data/maps/Skeld/`. Upstream
`src/Impostor.Api/Innersloth/Data/info.json` identifies `gameVersion: 2026.8.18`,
`platform: StandaloneItch`, and `dumpostorVersion: 1.0.1-dev`.
The upstream GPL-3.0 license text is retained as `sources/LICENSE-Impostor.txt`.
Console lists in task definitions omit some multi-stage consoles; explicit
supplements are documented in the combined blueprint.

## Artwork and estimates

`reference_map.png` is an unchanged copy of the owner's supplied `map2.png`
(5792 by 3168 pixels). It depicts Among Us / The Skeld, owned by Innersloth LLC.
It is **not** newly created artwork or assumed to be covered by the software
repositories' licenses. This request authorized its use in the local research
sandbox; that does not establish unrestricted redistribution rights.

The owner also supplied a schematic `mini_map.PNG`; it was used as a visual
orientation reference, not imported as collision geometry. The new small player
and preview characters are drawn from Pygame primitives.

Vent landmark clicks, camera housings, missing circular utility/task console
positions and the display calibration are local estimates against the supplied
image. Existing community task-marker references in `skeld_config.py` helped
cross-check panel identity; its approximate wall geometry was not imported.
Per-record provenance and SHA-256 input hashes are included in `among_us_map.json`.

Among Us and The Skeld belong to Innersloth LLC. This project is independent and
is not affiliated with or endorsed by Innersloth. Source license texts and this
notice are retained for review; no blanket license is asserted over the combined
game-derived data or artwork.
