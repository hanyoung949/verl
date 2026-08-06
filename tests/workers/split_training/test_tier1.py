from types import SimpleNamespace
import pytest
from verl.workers.split_training.dvi.tier1 import _aggregate


def test_tier1_expected_overlap_bounds_and_coverage():
    row=SimpleNamespace(
        verifier_topk_ids=[1], verifier_topk_logprobs=[__import__('math').log(0.6)],
        verifier_residual_mass=0.4, draft_support_token_ids=[1,2],
        teacher_probs_on_draft_support=[0.6,0.1],
    )
    result=_aggregate([row],[([1,2,3],[0.2,0.3,0.5])])
    assert result.expected_first_acceptance_lower == pytest.approx(0.3)
    assert result.expected_first_acceptance_upper == pytest.approx(0.6)
    assert result.coverage == pytest.approx(0.2)
    assert result.max_teacher_unknown_mass == pytest.approx(0.3)


def test_tier1_report_has_no_advancement_metric():
    row=SimpleNamespace(
        verifier_topk_ids=[1], verifier_topk_logprobs=[0.0],
        verifier_residual_mass=0.0, draft_support_token_ids=[1],
        teacher_probs_on_draft_support=[1.0],
    )
    result=_aggregate([row],[([1],[1.0])])
    assert 'advancement' not in result.__dict__
    assert result.expected_first_acceptance_lower == result.expected_first_acceptance_upper
