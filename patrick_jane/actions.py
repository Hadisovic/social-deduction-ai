"""Conditional five-head sampling; PPO includes every sampled factor exactly once."""
from dataclasses import dataclass
import numpy as np
import torch
from social_deduction.phase3_api import Intent,IntentKind
from .api import Action,Event
from .features import target_mask,goal_for


from .api import Choice
from .decoder import execute


def categorical(logits,mask,selected=None,generator=None):
    valid=torch.as_tensor(mask,dtype=torch.bool,device=logits.device)
    if len(valid)>len(logits):raise ValueError('Action mask exceeds head width')
    # Padded targets belong to the batch, not this actor. Multinomial consumes
    # width-dependent random draws even for zero-probability entries.
    logits=logits[:len(valid)]
    if not valid.any():raise ValueError('Empty conditional action mask')
    lp=torch.log_softmax(logits.masked_fill(~valid,-torch.inf),-1)
    probs=lp.exp()
    index=int(torch.multinomial(probs,1,generator=generator)) if selected is None else int(selected)
    if index<0 or index>=len(valid) or not valid[index]:raise ValueError('Stored action became invalid')
    entropy=-(probs[valid]*lp[valid]).sum()
    return index,lp[index],entropy,probs


def choose(output,packet,*,generator=None,choice=None):
    stored=choice
    values={k:-1 for k in ('interaction','navigation','communication','voting','target')}
    logp=output['value']*0;entropy=logp;probabilities={}
    def take(name,mask):
        nonlocal logp,entropy
        i,lp,en,pr=categorical(output[name],mask,getattr(stored,name) if stored is not None else None,generator)
        values[name]=i;logp=logp+lp;entropy=entropy+en
        probabilities[name]=pr.detach().cpu().tolist()
        return i
    target_kind=None;target_choice=None
    if packet.phase==0:
        interaction=take('interaction',packet.interaction_mask)
        if interaction==0:
            mask=packet.navigation_mask.copy()
            if not target_mask(packet,'navigation',1).any():mask[1]=False
            nav=take('navigation',mask)
            if nav:target_kind='navigation';target_choice=nav
        else:target_kind='interaction';target_choice=interaction
    elif packet.phase==1:
        communication=take('communication',packet.communication_mask)
        if communication:target_kind='communication';target_choice=communication
    elif packet.phase==2:take('voting',packet.vote_mask)
    else:raise ValueError('Finished actors have no living policy decision')
    if target_kind:take('target',target_mask(packet,target_kind,target_choice))
    result=Choice(**values)
    if stored is not None and result!=stored:raise ValueError('Conditional head mismatch')
    return result,logp,entropy,probabilities


def evaluate_batch(output,packets,choices):
    """Vectorized exact conditional likelihoods for contiguous recurrent sequences."""
    device=output['value'].device;size=len(packets)
    logp=torch.zeros(size,device=device);entropy=torch.zeros_like(logp)
    for name in ('interaction','navigation','communication','voting','target'):
        logits=output[name];mask=np.zeros(tuple(logits.shape),bool);indices=[];active=[]
        for i,(p,c) in enumerate(zip(packets,choices)):
            index=getattr(c,name);indices.append(max(0,index));active.append(index>=0)
            if index<0:mask[i,0]=True;continue
            if name=='target':
                kind,selected=('communication',c.communication) if p.phase==1 else ('interaction',c.interaction) if c.interaction>0 else ('navigation',c.navigation)
                valid=target_mask(p,kind,selected)
            else:
                valid=getattr(p,dict(interaction='interaction_mask',navigation='navigation_mask',communication='communication_mask',voting='vote_mask')[name]).copy()
                if name=='navigation' and not target_mask(p,'navigation',1).any():valid[1]=False
            if index>=len(valid) or not valid[index]:raise ValueError('Invalid stored conditional choice')
            mask[i,:len(valid)]=valid
        valid=torch.as_tensor(mask,device=device);weights=torch.as_tensor(active,device=device,dtype=logits.dtype)
        lp=torch.log_softmax(logits.masked_fill(~valid,-torch.inf),-1);probs=lp.exp()
        index_tensor=torch.as_tensor(indices,device=device)[:,None]
        logp=logp+lp.gather(1,index_tensor)[:,0]*weights
        entropy=entropy-(probs*lp.masked_fill(~valid,0.)).sum(-1)*weights
    return logp,entropy


