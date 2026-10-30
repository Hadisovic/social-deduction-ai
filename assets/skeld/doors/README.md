# Original Skeld door artwork

The PNG is the original extracted texture from
[AlvajoyAsante/among-us-assets, pinned commit3f6da7b](https://github.com/AlvajoyAsante/among-us-assets/blob/3f6da7bc243cc366ef6dc925938354568da023fd/Maps/Doors-sharedassets0.assets-104.png).
SHA256 is recorded in atlas.json and verified before loading. Among Us artwork
belongs to Innersloth; the source archive does not grant an Innersloth software
license. Retained here with provenance for the owner's research simulation.

The sheet is irregularly packed. atlas.json specifies separate horizontal and
vertical aperture windows, in open-to-closed order. These exclude unrelated
sprites packed into the empty space between door leaves. Vertical doors use
their own gray artwork; front-facing doors use the pale panel frames. The viewer
does not rotate front-facing doors to impersonate side doors.

The map already contains open door jambs. Only moving/closed apertures are drawn.
The texture archive does not include authoritative SpriteAnim playback rates;
the0.25-second transition is reconstructed and configurable in doors.py. Do not
describe this as a verified frame-perfect match to a specific current game build.

Additional behavior reference, without copying its implementation:
[LevelImposter DoorPatch](https://github.com/DigiWorm0/LevelImposter/blob/0aff16d6e60d97a640c7c9cb6806f4333b1e700e/LevelImposter/Core/Patches/Animations/DoorPatch.cs)
selects separate open/close clips when PlainDoor changes state;
[SabDoorBuilder](https://github.com/DigiWorm0/LevelImposter/blob/0aff16d6e60d97a640c7c9cb6806f4333b1e700e/LevelImposter/Builders/Sab/SabDoorBuilder.cs)
assigns the original clips and uses AutoOpenDoor for Skeld-style automatic doors.
