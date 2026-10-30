# Jane Impostor

The new model will use **Patrick Jane as an architectural reference**, with an
independent impostor design, observation adapter, capabilities and training study.
It will not replace the 75,075-parameter historical impostor or its results.

Start with [PLAN.md](PLAN.md), [shared rules](../../docs/jane/GAME_RULES.md) and
[interface contract](../../docs/jane/INTERFACE.md). `manifest.json` is deliberately
`planned`; no new runnable network or learned checkpoint is claimed here.

Read the actual in-progress Jane implementation after the owner announces its
push, then revise compatibility for both roles. Do not copy/paste its network as
the impostor base or treat this folder structure as final.
Do not assume crewmate and impostor tensor layouts, outputs or optimizer states
are interchangeable. Shared architecture does not mean copying an unverified
checkpoint or sharing one recurrent state between players.
