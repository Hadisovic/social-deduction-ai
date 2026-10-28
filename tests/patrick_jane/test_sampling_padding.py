"""Batch padding must never consume a different actor's private sampling stream."""
import unittest
import torch
from patrick_jane.actions import categorical


class SamplingPaddingTests(unittest.TestCase):
    def test_padding_does_not_change_choices_or_private_rng_state(self):
        traces=[];states=[]
        for padding in (0,13,77):
            generator=torch.Generator().manual_seed(17)
            logits=torch.tensor([.1,.2,.3]+[1000.]*padding)
            # Candidate padding is not part of this actor's legitimate action set.
            mask=[True,True,True]
            choices=[categorical(logits,mask,generator=generator)[0] for _ in range(32)]
            traces.append(choices);states.append(generator.get_state())
        self.assertEqual(traces[0],traces[1]);self.assertEqual(traces[0],traces[2])
        self.assertTrue(torch.equal(states[0],states[1]));self.assertTrue(torch.equal(states[0],states[2]))


if __name__=='__main__':unittest.main()
