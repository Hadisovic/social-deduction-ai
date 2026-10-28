"""Display supplied budgets; preserve historical pilot defaults for old status files."""


def progress_lines(metrics):
    decisions=metrics.get('decisions',0);target=metrics.get('decision_target',512)
    if isinstance(decisions,dict):lines=[f'{name}: {count}/{target} actor choices' for name,count in decisions.items()]
    else:lines=[f'Choices {decisions}/{target}']
    lines.append(f"Validation {metrics.get('validation_games',0)}/{metrics.get('validation_target',20)}")
    return lines


def controller_label(controller,sim_time,config,ghost=False):
    if ghost:return 'AUTO GHOST (POLICY INACTIVE)'
    if sim_time<float(config.get('prefix_handoff_seconds',0)):
        return 'SCRIPTED PREFIX / NO LEARNING'
    return controller.upper()


def update_label(version):
    return 'frozen reference (0 updates)' if version==-1 else f'update {version}'
