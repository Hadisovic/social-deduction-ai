# Patrick Jane crewmate reference

The teammate's source was reviewed after the owner's push-ready notice.
Implementation/model pin: `d459f38674ebc537acde20647cbd07700c494533`.
Published update102: 1,031,070 parameters / 52 parameter tensors, task-pursuit-v1.
Artifact bytes/checksums were verified; no weights were loaded or copied here.

The actual model has shared encoders/attention/fusion/GRU-256 and separate actor
and critic trunks. Players are 14 OTHER rows, events 64, targets dynamic. The
vote head is 14 other-player slots plus skip at index14. Native A2 currently uses
10 neural crew + 2 scripted impostors, with lights/doors/vents and task commitment.
The update102 weights are not A2 trained; task training wins are not held-out
deduction evidence.

`manifest.json` now records reviewed source/model identity, but still says
configuration-only. No final tensor adapter or shared paired-game readiness is
claimed. Use the original checkpoint loader/runtime; preserve all historical locks.

Next: verify the real role boundary and recurrent lifecycle, decide settings
conditioning with the teammate, and implement the crew-side shared adapter. Do
not rewrite the existing 192-feature layout or copy the crew network into the
impostor project. Keep both role structures provisional and separately versioned.
