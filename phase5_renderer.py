"""Policy diagnostics alongside the existing actor-only belief panel."""
import pygame
import torch
from phase3_renderer import Phase3Renderer
from phase4_renderer import BeliefPanel
from phase5_policy import tensors


class PolicyPanel(BeliefPanel):
    def draw_policy(self,env,policy,metadata):
        surf=super().draw(env.memory,env.belief,env.observation.own.active,expanded=False)
        pygame.draw.rect(surf,(15,26,41),(0,405,360,495))
        def text(value,y,color=(221,233,242),font=None):
            rendered=(font or self.small).render(value,True,color)
            surf.blit(rendered,(16,y))
        pygame.draw.rect(surf,(15,26,41),(0,45,360,83))
        name=next(i.display_name for i in env.memory.roster if i.player_id==env.focal)
        def label(value):
            for destination in env.observation.destinations:
                if value.endswith(' / '+destination.id):
                    return value.split(' / ')[0]+' / '+Phase3Renderer.readable(destination.name)+' / '+destination.room
            for identity in env.memory.roster:
                if value.endswith(' / '+identity.player_id):
                    return value.split(' / ')[0]+' / '+identity.display_name
            return value.replace('_',' ')
        text(f'{name} / {env.focal} | learned focal crewmate',51)
        text(f'Observed through {env.memory.tick*env.memory.dt:.1f}s | {env.memory.phase.value}',77)
        text('Policy chooses actions; other players scripted' if env.observation.own.active
             else 'Focal eliminated; memory and decisions frozen',99,(136,171,195))
        text('STRATEGIC POLICY',418,font=self.title)
        text('SMOKE TRAINED / NOT BENCHMARKED' if metadata.get('quick') else 'TRAINED POLICY / see evaluation report',450,(243,180,108))
        text('Current-only ablation' if env.ablation else 'Frozen belief + learned actions',475)
        text('Last: '+label(env.last_label)[:43],503)
        with torch.no_grad():
            distribution,value=policy(tensors(env.packet));prob=distribution.probs[0].numpy()
        text(f'Value: {value.item():.3f}',530)
        text('LEGAL ACTION PROBABILITIES',564,(119,213,195))
        indices=sorted(range(len(env.options)),key=lambda i:-prob[i])[:7]
        for row,i in enumerate(indices):
            y=593+row*33
            text(f'{prob[i]:5.1%}  {label(env.options[i].label)[:34]}',y)
            pygame.draw.rect(surf,(70,131,142),(16,y+19,round(328*prob[i]),4))
        text('P: pause | Right: one decision | R: replay',850)
        text('Probabilities use actor observations only.',874,(136,171,195))
        return surf
