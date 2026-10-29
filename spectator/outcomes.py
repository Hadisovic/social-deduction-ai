"""Honest replay labels: technical cuts and legacy time limits are not victories."""
def outcome_label(result):
    reason=result.get('reason','')
    if reason=='TRUNCATED':
        cause=result.get('metrics',{}).get('technical_truncation',{}).get('cause','technical safety limit')
        return 'TRUNCATED · '+cause.replace('_',' ')
    winner=result.get('winner')
    if winner in ('crewmate','impostor'):return winner.upper()+' VICTORY'
    if reason=='TIMEOUT':return 'HISTORICAL TIMEOUT · no victory'
    return 'NO VICTORY · '+str(reason).replace('_',' ')
