import math
from pathlib import Path

import pytest
import torch

from vllm.v1.worker.gpu.split_dvi.artifact import (
    DVIArtifactError, DVIArtifactManifest, DVIArtifactWriter,
    DVIRequestRecord, DVIEvaluationRecord,
)
from verl.workers.split_training.dvi.v2_replay import (
    _group_cycles, _metrics, _proposal_reproduction, _q_distribution,
    _runtime_proposal, _triton_uniform, evaluate_v2_artifact,
)


def _manifest(*, sampling_config=None):
    return DVIArtifactManifest(
        schema_version="v2", base_checkpoint="x", base_checkpoint_hash="b",
        head_adapter_hash="h", tail_adapter_hash="t", tokenizer_revision="tok",
        split_stage_0_size=1, split_stage_2_size=1, hidden_size=2, vocab_size=100,
        draft_length=3, dtype="float32", sampling_config=(
            sampling_config
            if sampling_config is not None
            else {"draft_temperature": 1.0, "top_k": 0}
        ),
        capture_mode="evaluation", policy_version="p", num_proposals=2,
        bonus_token=True, max_response_length=10,
    )

def _rows(cycle=0, *, drop=None, bad_kind=False):
    rows=[]
    for i in range(3):
        kind="proposal" if i < 2 else "bonus"
        if bad_kind and i == 2: kind="proposal"
        proposal=i < 2
        rows.append(DVIEvaluationRecord(
            request_id="r", cycle_id=cycle, generation_id=0, row_index=i,
            absolute_position=cycle, rng_position=cycle + 8, request_seed=7,
            row_kind=kind, num_proposals=2,
            draft_token_id=1 if proposal else None,
            draft_support_token_ids=[1,2] if proposal else [],
            draft_support_logits=[0.0,0.0] if proposal else [],
            stage_0_hidden=torch.zeros(2), verifier_topk_ids=[1],
            verifier_topk_logprobs=[0.0] if False else [-0.6931471805599453],
            verifier_residual_mass=0.5, teacher_probs_on_draft_support=[0.2,0.3] if proposal else [],
            accepted=(i == 0) if proposal else None, accepted_count=1,
            committed_token_ids=[1,9] if i == 0 else [], correction_token_id=9 if i == 0 else None,
            terminal_token_id=9 if i == 0 else None, advancement=2 if i == 0 else 0,
        ))
    return [r for r in rows if r.row_index != drop]

def _artifact(tmp_path: Path, rows, *, sampling_config=None):
    out=tmp_path / "artifact"
    with DVIArtifactWriter(out, _manifest(sampling_config=sampling_config)) as w:
        w.write_requests([DVIRequestRecord(request_id="r", prompt_token_ids=[0], response_token_ids=[1,9,1])])
        w.write_evaluation_records(rows)
    return out

def test_cpu_philox_matches_runtime_triton_vectors():
    vectors = [
        (0, 0, 0.7980929017066956),
        (1, 1, 0.6560034155845642),
        (7, 17, 0.8913067579269409),
        (31051, 1024, 0.6707912087440491),
        (-1, 2147483647, 0.323047935962677),
        (-(1 << 63), 2147483648, 0.7443950176239014),
        ((1 << 63) - 1, (1 << 40) + 3, 0.34124240279197693),
        (0x123456789ABCDEF, (1 << 63) - 1, 0.19478052854537964),
    ]
    assert [
        _triton_uniform(seed, offset) for seed, offset, _ in vectors
    ] == [expected for _, _, expected in vectors]


def test_production_rng_reproduces_recorded_proposals():
    rows = _rows()
    for row in rows[:-1]:
        row.draft_token_id = _runtime_proposal(
            row, _q_distribution(row, temperature=1.0, top_k=0)
        )
    reproduced, total, mismatches = _proposal_reproduction(
        [rows], temperature=1.0, top_k=0
    )
    assert (reproduced, total, mismatches) == (2, 2, ())

    rows[0].draft_token_id = 2 if rows[0].draft_token_id == 1 else 1
    reproduced, total, mismatches = _proposal_reproduction(
        [rows], temperature=1.0, top_k=0
    )
    assert reproduced == 1
    assert total == 2
    assert len(mismatches) == 1


def test_production_rng_requires_recorded_keys():
    rows = _rows()
    rows[0].rng_position = None
    with pytest.raises(DVIArtifactError, match="request_seed and rng_position"):
        _proposal_reproduction([rows], temperature=1.0, top_k=0)


def test_metrics_use_full_support_for_expected_and_coverage():
    m=_metrics([_rows()], temperature=1.0, top_k=0)
    assert m.expected_first_acceptance == pytest.approx(0.5)
    assert m.coverage == pytest.approx(0.5)

def test_missing_row_rejected():
    with pytest.raises(DVIArtifactError, match="not dense"):
        _group_cycles(_rows(drop=1))

def test_kind_count_mismatch_rejected():
    with pytest.raises(DVIArtifactError, match="kind/count"):
        _group_cycles(_rows(bad_kind=True))

def test_exact_reproduction_and_first_block_selection(tmp_path):
    report=evaluate_v2_artifact(_artifact(tmp_path, _rows()))
    assert report.exact_reproduction
    assert report.exact_cycles == 1
    assert report.without_first_block.cycles == 0

def test_resimulation_is_deterministic_and_stops_after_reject(tmp_path):
    path=_artifact(tmp_path, _rows())
    a=evaluate_v2_artifact(path, seed=7).resimulation
    b=evaluate_v2_artifact(path, seed=7).resimulation
    assert a == b
    assert a.prefix_histogram["3"] == 0
    assert a.prefix_histogram["4"] == 0


def test_replay_does_not_apply_rollout_temperature_to_raw_draft_logits(tmp_path):
    rows = _rows()
    for row in rows[:-1]:
        row.draft_support_logits = [0.0, 1.0]
        row.teacher_probs_on_draft_support = [0.2, 0.3]
    path = _artifact(
        tmp_path,
        rows,
        sampling_config={"rollout_temperature": 0.7, "top_k": 0},
    )

    report = evaluate_v2_artifact(path)
    q0 = 1.0 / (1.0 + math.exp(1.0))
    expected = min(q0, 0.2) + min(1.0 - q0, 0.3)
    assert report.all_cycles.expected_first_acceptance == pytest.approx(expected)
