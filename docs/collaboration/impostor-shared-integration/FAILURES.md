# Development failures retained

1. The first `uv python list` ran in the restricted sandbox and repeatedly failed
   to persist its interpreter cache. It ended without modifying models. The
   approved unsandboxed inventory/download created a separate exact runtime.
2. The first inference-transfer candidate passed the original strict loader, but
   included `torch.__version__` as a TorchVersion subclass in its metadata. The
   new runtime's safe `weights_only=True` loader correctly rejected that object.
   The candidate remains in `impostor_shared/assets/` as failed development
   evidence. The corrected exporter uses a plain string and a new `assets-v1/`
   directory; no old artifact is overwritten. No unsafe loader or safe-global
   bypass is used. Neural tensor identity is checked independently.

3. Initial new regressions:6 passed,2 fixture errors. The setup tried assigning
   Player.position, a read-only property. Fixtures now use the existing motion.spawn
   API. No game/map or policy behavior was changed to accommodate the tests.

4. Both first learned-match constructor attempts stopped before gameplay:
   the shared callback's behavior_seed integer exceeded Torch.manual_seed's
   accepted range. The adapter now hashes the private seed plus public owner ID
   into63 bits with a declared namespace. This seed is never a neural feature.
   Added a regression using the actual handoff seed function; retries use new
   output folders. No completed match, training result or changed weights is
   attributed to these failed starts.

5. Restricted-shell broader pytest runs failed at temporary-fixture setup/cleanup
   and stalled while reporting those errors. An approved run outside the sandbox
   with a new explicit local temporary directory passed84 tests, with5 disclosed
   private-recording skips. The new checkpoint unit checks hit the same actual
   WinError5/PermissionError on temporary files and are rerun outside that sandbox.
   No simulator/viewer source was changed to make restricted fixtures pass.

Items 1–5 describe engineering failures before the shared PPO pilot.

6. The approved 29 October shared pilot completed all 16 training episodes and
   saved all checkpoints. During validation, stream 101/episode 4/seed 3610006
   exceeded the 600-second wall execution ceiling. Its next case was unstarted;
   the other three blocks completed, leaving 30/32 validation outcomes. Parent
   session 50872 exited 1 after 39.17 minutes, including waiting for other queued
   validation work. This is not a game TIMEOUT or model loss. No checkpoint was
   selected/promoted; the 64 planned paired A/B confirmation executions did not
   start. No automatic retry or deadline extension was performed. Root cause of
   the slow case remains unestablished; no simulation-step trace was persisted
   for that non-first evaluation case. An offline curator outside the frozen
   learner namespace generated partial graphs/report and strict-reloaded the
   four candidate checkpoints without altering their source locks. See
   pilot-20261029/README.md. Future evaluation should persist per-case start,
   simulation progress and timings, and distinguish a wall abort from TIMEOUT.
