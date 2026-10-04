"""Export measured Phase 5 figures and concise story data after final testing."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase5_research_report import build


def main():
    output=ROOT/'docs/benchmarks/phase5/expanded'
    result=json.loads((output/'final-results.json').read_text())
    if result['status']!='final_evaluation_complete':raise ValueError('Final evaluation is not complete')
    progress=build(output)
    method=result['selected_checkpoint'];id_=result['metrics'][method]['ID'];patient=result['metrics'][method]['patient']
    labels={
        'learning_curves.png':('Learning across runs','Training returns, team wins and own tasks; these are training outcomes.'),
        'interaction_budget.png':('How much game experience?','Longer actions consume more game time. Compare both decisions and simulated time.'),
        'ppo_diagnostics.png':('Neural-network optimization','Policy and value losses, entropy, KL, clipping and value prediction. Loss is not accuracy.'),
        'validation_comparison.png':('Validation and model selection','Reserved validation matches were used to select checkpoints. They are not final test results.'),
        'validation_curve.png':('Learning by checkpoint','Saved checkpoints use the same small diagnostic validation set.'),
        'evaluation.png':('Final crew wins','500 familiar-family and 500 unseen patient-family matches per method; 95% match intervals.'),
        'paired_differences.png':('Comparison with scripted play','Paired match differences and uncertainty; zero means equal crew win rates.'),
        'training_seed_comparison.png':('Independent training seeds','Three full-memory training seeds; the current-only comparison has one seed and is exploratory.'),
        'voting_accuracy.png':('Voting accuracy and participation','Correct impostor votes among cast votes, alongside incorrect votes and skips.'),
        'action_distribution.png':('What the policy chooses','Fractions of controller decisions, not fractions of game time.'),
        'seed_outcomes.png':('Shared match seeds','The same seeded scenarios across methods; green is a crew win, red is a loss or timeout.'),
        'behavior.png':('Tasks and survival','Individual contribution matters because scripted teammates can carry a passive player.')}
    target=ROOT/'site/assets/phase5';target.mkdir(parents=True,exist_ok=True)
    figures=[]
    for name in progress['figures']:
        if name not in labels:continue
        shutil.copyfile(output/'figures'/name,target/name)
        title,caption=labels[name]
        figures.append(dict(path=f'assets/phase5/{name}',title=title,caption=caption))
    status_path=ROOT/'project-status.json';status=json.loads(status_path.read_text())
    status['lastUpdated']='2026-10-04'
    for phase in status['roadmap']:
        if phase['id']=='phase-5':phase['status']='complete'
    status['phase5Study']=dict(stage='evaluated',matchesPerMethod=1000,
        metric=f"{id_['crew_win_rate']:.1%} / {patient['crew_win_rate']:.1%}",
        metricLabel='crew wins: familiar / unseen patient opponents · 500 matches each',
        date='PHASE 5 · EVALUATED',tags=['THREE TRAINING SEEDS','SELECTION LOCKED BEFORE TESTING'],
        notes=(f"The selected policy completed {id_['mean_own_tasks']:.2f} own tasks per familiar-family match and "
               f"{patient['mean_own_tasks']:.2f} against unseen patient opponents. Checkpoint selection used validation only. "
               "The graphs compare all evaluated runs with scripted, random, idle and untrained controls. "
               "The selected policy clearly improves on idle and random controls; superiority over scripted play is not established. "
               "Voting accuracy is conditional on casting a vote. The current-only ablation has one training seed, "
               "so a replicated memory benefit is not established. These are simulator results; adapting opponents and self-play remain future work."),
        figures=figures)
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({'exported_figures':len(figures),'selected_model':method,'publication':'local_files_only'}))


if __name__=='__main__':main()
