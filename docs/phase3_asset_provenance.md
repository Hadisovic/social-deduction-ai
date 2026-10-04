# Phase 3 asset provenance

Reviewed on 2026-10-02. The owner requested the three sources below. The visual
simulator uses selected clone artwork in an **ignored local pack**, the existing
source-backed Skeld data, and an original procedural character fallback. PNGs in
`assets/phase3_local/` are not part of this public repository. No upstream
implementation code was copied.

## AI0702/Among-Us-clone: character and task artwork

Pinned revision: `e75a1410c8bc9e82b41f2ab51deec373c8486e29`.

Inspected:

- [README.md](https://github.com/AI0702/Among-Us-clone/blob/e75a1410c8bc9e82b41f2ab51deec373c8486e29/README.md): identifies ripped game artwork and credits Innersloth.
- [LICENSE](https://github.com/AI0702/Among-Us-clone/blob/e75a1410c8bc9e82b41f2ab51deec373c8486e29/LICENSE): Unlicense for the repository software; this does not establish a separate redistribution grant for Innersloth artwork.
- `sprites.py`, `settings.py`, `game.py`: directional arrays, numbered frames,
  color handling, sprite anchoring, body selection and Pygame integration.
- The recursive Git tree and the selected PNGs listed in the manifest.

The owner's subsequent request to use the assets is implemented locally, while
preserving their earlier explicit condition not to commit ripped commercial
artwork without established redistribution rights. We do not claim that the
source software license relicenses third-party art. No independent permission
to redistribute this artwork was established in the inspected sources.

[assets/phase3_assets.json](../assets/phase3_assets.json) enumerates **every exact
upstream path and Git blob SHA-1** used. It contains 179 rendering keys referencing
119 unique PNG files:

- Red, Blue, Green, Yellow and Orange: four sampled walking frames in each of
  `left`, `right`, `up`, `down` under
  `Assets/Images/Player/<Color>/<color>_<direction>_walk/stepN.png`.
- Purple, Black, Pink, White and Brown: the one frame supplied per direction is
  reused across the four animation keys; these upstream colors have no walking
  sequence. This limitation is retained rather than inventing source frames.
- Ten body images: `Assets/Images/Player/Dead/Dead<color>.png`.
- Nine task-panel illustrations under `Assets/Images/Tasks/`:
  `Align Engine Output/engineAlign_base.png`, `Clear Asteroids/space3.png`,
  `Divert Power/electricity_Divert_Base.png`, `Empty Garbage/gb2.png`,
  `Fix Wiring/electricity_wire_base1.png`, `Fuel Engines/fuel_engines_base.png`,
  `Stabilize Steering/nav_stabilize_target.png`, `Start Reactor/reactor_base2.png`,
  `Swipe Card/admin_Wallet.png`.

The task panels illustrate matching timed tasks in the spectator sidebar. They
are not functional commercial-game minigames. Source bytes remain unchanged;
Pygame scales surfaces for display. Missing or modified images fall back to
original characters or a text task card. The normal simulator makes no network
request. The explicit local installer verifies each pinned Git blob before
writing, downloads only these 119 PNGs, and never executes upstream code:

```powershell
python phase3_assets.py --install
python run_phase3.py
```

The installer is standard-library-only. It is optional for an offline clone.
No ghost, kill-animation, sound, UI menu, full asset directory or unrelated task
image is imported. No upstream code was reused verbatim or adapted; directional
states and independent feet anchoring served as architectural references.
`phase3_renderer.character_sprite` is newly written Pygame geometry. Fonts are
system fonts through `pygame.font.SysFont`.

## darkmatter2222/Agentic-Among-Us-: inspected reference

Pinned revision: `5cc8e4ca694ae5410f42dc921e355591a5777532`.
Inspected its [README](https://github.com/darkmatter2222/Agentic-Among-Us-/blob/5cc8e4ca694ae5410f42dc921e355591a5777532/README.md),
`package.json`, recursive Git tree and `src/rendering/PlayerRenderer.ts`.
The player renderer draws procedural PIXI graphics; the tree has map/reference
screenshots, not a reusable character sprite pack. The README describes private
project licensing; no explicit redistribution grant or LICENSE file was found.
No code or image from this source was imported. Its reference map would also
replace the already validated map registration without adding useful task data.

## Impostor: requested Skeld data already integrated

The requested [Skeld directory](https://github.com/Impostor/Impostor/tree/b09c40b7e12d35c6cae996a26a943185b341c907/src/Impostor.Api/Innersloth/Data/maps/Skeld)
uses revision `b09c40b7e12d35c6cae996a26a943185b341c907`.
All five requested source files (`tasks.json`, `doors.json`, `spawn.json`,
`systems.json`, `vents.json`) were fetched and compared with
`assets/skeld/sources/impostor-*.json`: **all five are byte-for-byte identical**.
Their retained GPL-3.0 notice/license and provenance are recorded in
[assets/skeld/NOTICE.md](../assets/skeld/NOTICE.md). Phase 3 uses the existing
compiled map destinations and actual ordinary task consoles. It does not enable
vent travel, door operations, sabotage or utility tasks.

The source-backed map image `assets/skeld/reference_map.png` and validated
geometry remain unchanged. Their existing owner-supplied/game-derived status
and accuracy limits are separate from the optional character pack.

## Attribution

### Local rendering corrections (2026-10-03)

The pinned red PNGs use RGB material channels rather than final display colors.
`phase3_assets.py` decodes those channels into suit, visor and shadow colors on
copied surfaces. Alpha, neutral bone highlights and original PNG bytes are retained.
Purple, black, pink, white and brown have repeated still-image aliases for their
walk frames in the manifest. Their living animations now reuse the complete red
material-mask walk cycle with color-specific palettes. Corpse art still uses each
color's pinned body image. No new artwork download or manifest rewrite was needed.

Among Us, The Skeld and the requested game artwork belong to Innersloth LLC.
The selected local PNGs are obtained through AI0702/Among-Us-clone, whose README
credits Innersloth. This independent research simulator is not affiliated with
or endorsed by Innersloth. The local pack is not a statement of redistribution
rights and must remain outside public commits. The source manifest and this
audit provide exact attribution without bundling the artwork.
