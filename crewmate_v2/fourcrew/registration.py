"""Exact prospective plan. Registration is not authorization to execute it."""
from crewmate_v2.config import LobbySettings
from crewmate_v2.learning.provenance import digest
from .config import Rewards
from .policy import source_lock
from .scenarios import PREFIX_VERSION
from pathlib import Path
import hashlib

EXPERIMENT='fourcrew-reporting-2048-seed17-v1'


def training_context(index):
    if type(index) is not int or not 0<=index<32:raise ValueError('Training schedule exhausted')
    return dict(seed=2780000+index,seed_partition='registered_train',case_index=index,
                context='ordinary' if index%2==0 else 'body',deduction_frequency=float(index%2),rotation=index%3)


def validation_contexts():
    for i in range(8):
        base=dict(seed=2790000+i,seed_partition='registered_validation',case_index=i,rotation=i%3)
        yield {**base,'context':'ordinary','deduction_frequency':0.}
        yield {**base,'context':'body' if i<4 else 'meeting_crew' if i%2==0 else 'meeting_impostor','deduction_frequency':1.}


def plan():
    return dict(experiment=EXPERIMENT,status='registered_pending_final_owner_authorization',initialization=17,
        lobby=LobbySettings().to_dict(),arms={'original':Rewards.original().to_dict(),'shaped_v2':Rewards().to_dict()},
        zero_reference='fresh initialization 17; frozen before all updates; greedy evaluation only',
        algorithm=dict(policy='FourCrewPolicy',workers=4,complete_matches_per_batch=2,central_updates_after_paired_batches=True,
            learning_rate=.0003,gamma=.99,gae_lambda=.95,ppo_epochs=4,minibatch=64,clip=.2,entropy=.02,value_coefficient=.5,max_gradient_norm=.5),
        train_schedule=[training_context(i) for i in range(32)],validation=list(validation_contexts()),
        decision_budget=dict(unit='individual non-forced actor choices',target_per_arm=2048,admissible_max_per_arm=3072,
            minimum_matched_matches_per_arm=16,maximum_matched_matches_per_arm=32,
            stopping='first completed paired two-match batch with both arms >=2048 and >=16 matches',
            overshoot='Never trim trajectories. Reject a collected paired batch before either update if either cumulative total would exceed3072; mark budget-inconclusive.',
            exposure='identical accepted seed/context/opponent sequence; actual actor decisions may differ'),
        prefix=dict(version=PREFIX_VERSION,kill_seconds=45.,handoff_seconds=75.,meeting_handoff_seconds=75.2,
            construction='Legal navigation to public static waypoints, one legal kill, all survivors at common site; optional legal pre-policy report. Actor histories come only from observations.',
            no_policy_forced_actions=True,prefix_events_excluded_from_learning=True),
        coverage=dict(training_legal_actor_body_pairs_per_arm=24,training_prefix_body_matches_min=8,
            validation_body_cases=4,validation_legal_actor_body_pairs_per_policy=12,
            validation_meeting_cases=4,crew_reporter_meetings=2,impostor_reporter_meetings=2,
            minimum_legal_actor_meeting_pairs_per_policy=12,
            realized_training_accepted_reports_for_reporting_learning_interpretation=2),
        evaluation=dict(policies=['original','shaped_v2','zero'],games_per_policy=16,total_executions=48,
            distinct_scenarios=8,action_selection='greedy',selection='last common accepted training batch satisfying the preregistered stopping rule; never validation-selected',
            metrics=['win reasons','own/ghost tasks','visibility/range/legal/choice/held report opportunities','selected/submitted/accepted reports',
                'ballot participation/accuracy/skips','claims and incorrect accusations','survival','frozen-belief top1/NLL/Brier/ECE',
                'policy vote-target ranking','reporter-only accuse/avoid/uniform','reporter-role macro strata','direct-witness strata','paired scenario bootstrap']),
        total_wall_seconds=14400,final_partitions=False,full_training_authorized=False,
        inconclusive=['wall stop or incomplete48 evaluations','budget overshoot/rejection or target not reached by32 matched games',
            'prefix legality/clock/privacy/source/checkpoint failure','coverage below registered minimum',
            'fewer than2 realized training reports in an arm: reporting-learning endpoint inconclusive',
            'reporter-role coverage absent: independent-deduction endpoint inconclusive'],
        acceptance='exploratory development study; no Phase6.1 acceptance or generalization claim from one initialization/eight clusters')


def runtime(spectator):
    root=Path(spectator)
    return dict(policy=source_lock(),spectator={name:hashlib.sha256((root/'spectator'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                for name in ('recorder.py','replay_format.py')})


def authorize(record,spectator=None):
    if record.get('approved') is not True or record.get('experiment')!=EXPERIMENT or record.get('plan_sha256')!=digest(plan()) or record.get('runtime_sha256')!=digest(runtime(spectator) if spectator else source_lock()):
        raise PermissionError('Separate final owner authorization must bind exact registered plan and source hashes')


def stop_state(decisions,matches):
    if any(n>3072 for n in decisions.values()):return 'inadmissible_overshoot'
    if matches>=16 and all(n>=2048 for n in decisions.values()):return 'complete'
    return 'inconclusive_schedule_exhausted' if matches>=32 else 'continue'
