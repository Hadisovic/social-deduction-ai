"""Quick smoke-test: generate a demo replay and verify it."""
from spectator.demo import ensure_demo_replay
from spectator.replay_format import parse_replay
from pathlib import Path

p = ensure_demo_replay(Path('replays'), seed=7)
print(f'Demo replay: {p}')
data = parse_replay(p)
ticks = len(data['ticks'])
events = len(data['events'])
complete = data['complete']
print(f'Ticks: {ticks}  Events: {events}  Complete: {complete}')
result = data['result']
winner = result['winner']
reason = result['reason']
print(f'Result winner: {winner}  reason: {reason}')
tags = data['tags']
print(f'Tags: {tags}')

# Basic assertions
assert ticks > 0, "Must have at least one tick"
assert complete, "Demo match must be complete"
assert winner in ('crewmate', 'impostor', None), "Invalid winner"
print('OK - demo replay verified')
