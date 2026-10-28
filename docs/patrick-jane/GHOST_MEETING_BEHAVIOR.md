# Ghost behavior during meetings

Reviewed 2026-10-28. **No behavior or reward coefficient was changed.**

Patrick Jane inherits the historical V2 task-only ghost executor:

- Death/ejection retains the crewmate's individual task ownership. Tasks are not
  reassigned to surviving crew.
- The deterministic ghost executor receives only its owner's tasks, legal map
  task destinations, bounds, speed and simulation time. It receives no roles,
  evidence, other memories, policy, voting or communication capability.
- During roaming it moves directly toward its own next ready task, ignoring
  walls as a ghost, and completes the existing multistage task plan.
- During both discussion and voting, ghost movement and task progress freeze.
  Pending incubation deadlines advance by the meeting duration, preserving the
  same remaining delay when roaming resumes. Living task work also freezes.
- A completed ghost task contributes to the team task bar. The **approved PJ
  reward remains +1 to its owner**, once, rather than to a surviving agent. Its
  last living transition receives delayed discounted task/team credit until
  match termination. No new ghost neural decision enters the decision budget.

This is an explicit research approximation: ghosts are scripted task executors,
not learned agents or evidence-gathering teammates. Long meetings cannot provide
free task work or bypass incubation. Retail UI/animation equivalence is not
claimed. Changing this timing or reward would change the experiment and is out
of scope for this pilot.

Implementation: `crewmate_v2/ghosts.py:advance_ghost`, inherited unchanged;
`patrick_jane/environment.py:step`, `patrick_jane/worker.py:advance`,
`patrick_jane/rewards.py:RewardLedger`, `patrick_jane/collector.py:collect`.

Verification: `test_ghost_work_and_incubation_pause_in_both_meeting_phases`,
`test_ghost_executor_retains_ownership_and_freezes_meetings`,
`test_delayed_team_credit_includes_dead_actors`, and
`test_task_report_ghost_rewards_unique_and_no_vote_reward`.
