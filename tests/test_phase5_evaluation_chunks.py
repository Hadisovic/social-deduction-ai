import pytest
from phase5_evaluation import evaluation_case,evaluate


def test_chunked_final_manifest_is_exactly_the_serial_manifest_without_overlap():
    whole={evaluation_case(i,1000) for i in range(1000)}
    chunks=[evaluation_case(i,100,match_offset=offset) for offset in range(0,500,50) for i in range(100)]
    assert len(chunks)==len(set(chunks))==1000
    assert set(chunks)==whole
    assert {seed for seed,heldout in chunks if heldout}==set(range(640000,640500))
    assert {seed for seed,heldout in chunks if not heldout}==set(range(630000,630500))
    assert not whole.intersection({evaluation_case(i,100,validation=True) for i in range(100)})


@pytest.mark.parametrize('kwargs',[
    {'validation':True,'matches':20,'match_offset':90},
    {'matches':100,'match_offset':451},
    {'matches':20,'match_offset':-1},
    {'quick':True,'matches':20,'match_offset':1}])
def test_out_of_partition_chunks_fail_before_creating_output(tmp_path,kwargs):
    output=tmp_path/'invalid'
    with pytest.raises(ValueError):evaluate(output,**kwargs)
    assert not output.exists()
