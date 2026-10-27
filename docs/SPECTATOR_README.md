# Spectator & Replay Studio

> **Social Deduction AI — Crewmate-First Training Spectator**
>
> Branch: `feat/crewmate-first-training-spectator`

A polished desktop application for watching reinforcement-learning agents play Among Us,
inspecting their neural network decisions, following training progress, and replaying
previously recorded games.

![Spectator Screenshot](docs/spectator_screenshot.png)

---

## Quick Start

```bash
# In your repository root:
python run_training_spectator.py
```

On first launch, the spectator automatically generates a demo match and opens it.

---

## Installation

### Prerequisites

| Package | Minimum version |
|---------|----------------|
| Python  | 3.12           |
| pygame-ce | 2.5.0       |
| numpy   | 1.21           |
| shapely | 2.0            |

All of the above are already in `requirements.txt`.

```bash
pip install -r requirements.txt
```

No browser, no web server, no cloud account required.

### Windows Launcher

Double-click `launch_spectator.bat` for a one-click start.

---

## Launch Commands

| Command | Description |
|---------|-------------|
| `python run_training_spectator.py` | Auto-generate demo and open it |
| `python run_training_spectator.py --watch "C:\AI\crewmate-runs"` | Monitor a training directory |
| `python run_training_spectator.py --replay path/to/match.jsonl` | Open a specific replay |
| `python run_training_spectator.py --demo --seed 42` | Generate a specific demo seed |
| `python run_training_spectator.py --fullscreen` | Start fullscreen |
| `python run_training_spectator.py --speed 4` | Start at 4× speed |
| `python run_training_spectator.py --mode follow_latest` | Auto-follow new replays |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause |
| `RIGHT` | Step forward one simulation tick |
| `LEFT` | Step backward one tick |
| `.` (period) | Jump to next event marker |
| `,` (comma) | Jump to previous event marker |
| `N` | Next episode |
| `P` | Previous episode (seeks to start for now) |
| `R` | Restart current episode from beginning |
| `+` / `=` | Double playback speed |
| `-` | Halve playback speed |
| `TAB` | Cycle selected player |
| `1` | Toggle role reveal (shows Crewmate/Impostor) |
| `G` | Toggle route overlay |
| `V` | Toggle sight-range circles |
| `T` | Toggle task location markers |
| `L` | Toggle player name labels |
| `C` | Toggle collision geometry overlay |
| `M` | Cycle playback mode (Manual → Auto-Next → Follow Latest) |
| `S` | Toggle ★ Favourite on current replay |
| `F` | Browse for replay folder |
| `O` | Open a specific replay file |
| `D` | Generate and play a new demo match |
| `ESC` / `Q` | Quit |

### Mouse

- **Click the timeline** to seek to any point in the replay
- **Click a player row** in the right panel to select/inspect them

---

## Playback Modes

| Mode | Behaviour |
|------|-----------|
| **Manual** | Stay on the selected replay until you change it |
| **Auto-Next** | Play to end, then automatically load the next replay |
| **Follow Latest** | Whenever a new compatible replay appears in the watch directory, load it |

Switch with `M` or `--mode`.

---

## How to Monitor a Training Directory

```bash
python run_training_spectator.py --watch "C:\AI\crewmate-runs" --mode follow_latest
```

The spectator polls the directory every ~2 seconds for new `.jsonl` files.
Replays are loaded as soon as they are complete (i.e. the `.jsonl.writing` temporary
file has been renamed to `.jsonl`).

Incomplete files (still being written) are skipped automatically.

---

## Enabling Replay Recording in Training

The recorder is a **pure observer** — it never changes simulation timing, RNG,
action selection, rewards, or model gradients.

### With ScriptedMatch / Phase3Game

```python
from spectator.recorder import MatchRecorder

match = ScriptedMatch(seed=42)
recorder = MatchRecorder(
    match,
    "replays/ep-0042.jsonl",
    episode=42,
    training_session="run-20261027",
    checkpoint_path="models/crewmate_ep42.pt",
    model_architecture="PPO-MLP",
)

while not match.game.is_terminal:
    match.step()
    recorder.after_step()          # ← only line added to the loop

recorder.finish()                  # writes result + atomically renames file
```

### Recording Neural Decisions (optional)

```python
recorder.record_neural(
    player_id="p0",
    action_kind="go_to_task",
    action_target="console1",
    probabilities={"go_to_task": 0.72, "wait": 0.15, "report": 0.08, ...},
    value_estimate=0.43,
    entropy=1.24,
    legal_actions=["go_to_task", "wait", "report"],
    masked_actions=["kill", "fake_task"],
)
```

---

## How Replays Are Structured

Replays are **JSONL** files (one JSON object per line, schema `spectator-v1.0`).

```
{"record": "header",  ...}    ← always first — static metadata
{"record": "tick",    ...}    ← one per simulation tick — player positions
{"record": "event",   ...}    ← game events (kills, reports, meetings, …)
{"record": "neural",  ...}    ← optional — one per agent decision step
{"record": "reward",  ...}    ← optional — per-tick reward components
{"record": "result",  ...}    ← always last — match outcome
```

### Header fields

| Field | Description |
|-------|-------------|
| `schema` | `"spectator-v1.0"` |
| `game_schema` | `"phase3-v1"`, `"crewmate-v2"`, etc. |
| `source_version` | Git SHA of the codebase that recorded the match |
| `training_session` | Run ID (free text) |
| `episode` | Episode number |
| `scenario_seed` | Match seed |
| `agent_roles` | `{player_id: "crewmate" \| "impostor"}` |
| `controller_types` | `{player_id: "scripted" \| "ppo" \| "frozen"}` |
| `checkpoint_path` | Path to the model checkpoint used |
| `model_architecture` | Architecture identifier (e.g. `"PPO-MLP"`) |
| `config` | Full `GameConfig` dict |

### Tick fields

| Field | Description |
|-------|-------------|
| `tick` | Integer tick index |
| `sim_time` | Simulated time in seconds |
| `phase` | `"roaming"`, `"discussion"`, or `"voting"` |
| `players` | Dict of player snapshots (position, velocity, status, role, …) |
| `bodies` | Dict of body snapshots (position, victim, killer, reported) |

---

## Neural Visualization

When neural data is recorded, the right panel shows:

- **Chosen action** with target (highlighted in teal)
- **Action probabilities** as animated progress bars (top 8 actions)
- **Value estimate** from the critic network
- **Policy entropy** (lower = more decisive)
- **Reward** (per step and cumulative)

For scripted bots, the panel correctly displays *"No neural data recorded — scripted bots use rule-based decisions"* rather than fabricating values.

---

## Interesting-Game Auto-Detection

Each completed replay is automatically tagged with detected properties:

| Tag | Condition |
|-----|-----------|
| `first_impostor_win` | Impostor wins |
| `first_crew_win_ejection` | Crew wins by ejecting the impostor |
| `first_crew_win_tasks` | Crew wins by completing all tasks |
| `multiple_kills` | 2+ kills in the match |
| `witnessed_murder` | A crewmate witnessed a kill |
| `unexpected_innocent_ejection` | A crewmate was ejected |
| `very_long_game` | Match lasted >180 simulated seconds |
| `extreme_high_reward` | Total reward > 50 |
| `extreme_low_reward` | Total reward < -20 |

Tags appear in the episode library panel. Use `S` to manually bookmark interesting replays.

---

## Storage

A typical replay file for a full match is **20–150 KB** depending on duration.
The spectator does not automatically delete manually favorited replays.

---

## Switching Between Crewmate and Impostor Sources

Both collaborators can use the same application with their own local directories:

```bash
# Crewmate developer
python run_training_spectator.py --watch "C:\AI\crewmate-runs"

# Impostor developer
python run_training_spectator.py --watch "C:\AI\impostor-runs"
```

No sync of raw training logs between machines is required.

---

## Impostor Integration (for impostor developer)

The spectator reads the standard `spectator-v1.0` JSONL format.
Add these calls to your training loop (see `spectator/impostor_adapter.py` for details):

```python
from spectator.recorder import MatchRecorder

recorder = MatchRecorder(match, f"replays/impostor_ep{episode:05d}.jsonl",
                         episode=episode, training_session="impostor-run-1")
# ... inside loop:
recorder.after_step()
# ... at end:
recorder.finish()
```

The adapter (`spectator/impostor_adapter.py`) also provides impostor-specific action
labels for the neural decision panel.

---

## Historical Log Compatibility

| Log type | Position playback | Event timeline | Neural probabilities |
|----------|:-----------------:|:--------------:|:--------------------:|
| `spectator-v1.0` JSONL | ✅ | ✅ | ✅ (if recorded) |
| Phase 3 `replay_document()` JSONL | ⚠️ events only | ✅ | ❌ |
| Other old logs | ❌ | ❌ | ❌ |

For logs lacking position snapshots, the timeline and event list remain functional
but character movement cannot be shown.

---

## Running Tests

```bash
pytest tests/test_spectator.py -v
```

25 tests covering:
- Schema serialisation / deserialisation
- Malformed and incomplete file handling
- Library scanning and favourite toggle
- Playback engine seeking, stepping, event markers
- Recorder non-interference (same trajectory hash with/without recording)

---

## Troubleshooting (Windows / Pygame)

| Problem | Solution |
|---------|----------|
| `No module named 'pygame'` | `pip install pygame-ce` |
| Window appears then crashes | Ensure you have a display device connected (SDL requires a display) |
| `No module named 'shapely'` | `pip install shapely` |
| Demo takes a long time | The first launch generates the demo replay; subsequent launches are instant |
| Map artwork missing | The reference map PNG is at `assets/skeld/reference_map.png`. If missing, run `git lfs pull` or re-clone |
| Characters appear as colored circles | The optional local artwork pack is not installed. Characters use the built-in original sprite renderer instead (correct behaviour) |
| Folder browser doesn't open | Install `tkinter` (`python -m tkinter` to test). Without it, pass `--watch` on the command line |

---

## Partner Installation (Independent)

```bash
git clone https://github.com/Hadisovic/social-deduction-ai.git
cd social-deduction-ai
git checkout feat/crewmate-first-training-spectator
pip install -r requirements.txt
python run_training_spectator.py --watch "C:\path\to\your\impostor-runs"
```

No additional configuration is required.

---

## Architecture Overview

```
run_training_spectator.py     ← entry point, Pygame loop, keyboard/mouse
spectator/
  __init__.py
  replay_format.py            ← JSONL schema, recorder, library scanner
  recorder.py                 ← MatchRecorder adapter for Phase3/V2 games
  playback.py                 ← ReplayPlayer (seek, step, interpolate)
  ui_renderer.py              ← SpectatorRenderer (all Pygame drawing)
  app_state.py                ← SpectatorAppState shared between loop/renderer
  watcher.py                  ← Polling watch service (no background threads)
  demo.py                     ← Demo match generator
  impostor_adapter.py         ← Impostor replay adapter + action labels
  _smoke_test.py              ← Quick sanity check
tests/
  test_spectator.py           ← 25 automated tests
docs/
  spectator_screenshot.png    ← Real application screenshot
launch_spectator.bat          ← Windows one-click launcher
```

---

## Crewmate V2 Integration Status

The spectator is built against the **Phase 3 / ScriptedMatch** interface, which
is the base class for `CrewmateV2Game`.

To use `CrewmateV2Game` as the match source, replace `ScriptedMatch` in
`spectator/recorder.py` with `CrewmateV2Game` and supply appropriate V2 bots/policies.
The recorder calls only `match.game.players`, `match.game.bodies`,
`match.game.event_log`, and `match.game.result`, which are all present in both
Phase 3 and V2.

The neural decision panel is ready to display `V2Observation` outputs once the
PPO training pipeline calls `recorder.record_neural(...)` with the action
probabilities and value estimates from the policy network.

---

## Known Limitations

- **Live incremental playback** (following a match tick-by-tick while it's being written) is not yet implemented; the spectator loads complete replays
- **Network architecture diagram** is not yet implemented; the decision dashboard (action probabilities, value, entropy) is complete
- **Training statistics dashboard** is embedded in episode info rather than a separate charting panel; full charting requires tensorboard log parsing (future work)
- **Actor-perspective mode** (showing only observations visible to the selected player) is architecturally supported but not yet surfaced in the UI
- **V2 ghost rendering** is not yet implemented (ghosts from `crewmate_v2.ghosts` would require exposing ghost positions via the recorder)
