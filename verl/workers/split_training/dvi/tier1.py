"""Tier-1 candidate evaluation on a frozen schema-v2 capture."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch
from safetensors.torch import load_file as load_safetensors
from vllm.v1.worker.gpu.split_dvi.artifact import DVIArtifactReader, DVIEvaluationRecord

from .draft_head import SplitDVIDraftHead
from .offline import DVITrainResult, _sha256_tensor, load_train_result


@dataclass(frozen=True)
class Tier1Metrics:
    rows: int
    expected_first_acceptance_lower: float
    expected_first_acceptance_upper: float
    coverage: float
    max_teacher_unknown_mass: float


@dataclass(frozen=True)
class Tier1CandidateReport:
    schema_version: str
    capture_id: str
    candidate_draft_sha256: str
    marginal_all_proposals: Tier1Metrics
    cycle_start_all: Tier1Metrics
    cycle_start_without_first_block: Tier1Metrics

    def to_dict(self):
        return asdict(self)

    def write_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _known_teacher(row: DVIEvaluationRecord):
    known = {int(t): float(p) for t, p in zip(row.verifier_topk_ids, map(torch.exp, torch.tensor(row.verifier_topk_logprobs)))}
    for token, prob in zip(row.draft_support_token_ids, row.teacher_probs_on_draft_support):
        known[int(token)] = float(prob)
    topk = set(row.verifier_topk_ids)
    accounted_residual = sum(prob for token, prob in known.items() if token not in topk)
    unknown = max(0.0, float(row.verifier_residual_mass) - accounted_residual)
    return known, unknown


def _aggregate(rows, q_rows):
    if not rows:
        return Tier1Metrics(0, 0.0, 0.0, 0.0, 0.0)
    lower = upper = coverage = max_unknown = 0.0
    for row, (ids, probs) in zip(rows, q_rows):
        known, unknown_p = _known_teacher(row)
        known_overlap = sum(min(float(q), known.get(int(token), 0.0)) for token, q in zip(ids, probs) if int(token) in known)
        unknown_q = sum(float(q) for token, q in zip(ids, probs) if int(token) not in known)
        lower += known_overlap
        upper += known_overlap + min(unknown_q, unknown_p)
        teacher_topk = set(row.verifier_topk_ids)
        coverage += sum(float(q) for token, q in zip(ids, probs) if int(token) in teacher_topk)
        max_unknown = max(max_unknown, unknown_p)
    n = len(rows)
    return Tier1Metrics(n, lower/n, upper/n, coverage/n, max_unknown)


def evaluate_v2_candidate_tier1(artifact_dir, result_dir, base_projection, *, batch_size=64, device='cuda', capture_id='capture_dev_v2_014'):
    reader = DVIArtifactReader(artifact_dir)
    if reader.manifest.schema_version != 'v2':
        raise ValueError('Tier 1 requires a schema-v2 frozen capture')
    result = load_train_result(result_dir)
    if result.source_base_checkpoint_hash != reader.manifest.base_checkpoint_hash:
        raise ValueError('Candidate and frozen capture base checkpoint mismatch')
    base_projection = base_projection.float().cpu().contiguous()
    if _sha256_tensor(base_projection) != result.base_projection_sha256:
        raise ValueError('Candidate base projection checksum mismatch')
    target = torch.device(device)
    model = SplitDVIDraftHead(base_projection, rank=result.rank, alpha=result.alpha, norm=result.norm, seed=result.seed).to(target)
    state = load_safetensors(str(Path(result_dir)/'draft_head.safetensors'))
    with torch.no_grad():
        model.lora_a.copy_(state['lora_A.weight']); model.lora_b.copy_(state['lora_B.weight'])
        if model.norm_weight is not None: model.norm_weight.copy_(state['norm.weight'])
    rows=[row for row in reader.iter_evaluation_records() if row.row_kind=='proposal']
    q_rows=[]; model.eval()
    with torch.no_grad():
        for off in range(0, len(rows), batch_size):
            batch=rows[off:off+batch_size]
            hidden=torch.stack([row.stage_0_hidden.float() for row in batch]).to(target)
            logits=model(hidden).float()
            values, ids=torch.topk(logits, 16, dim=-1)
            probs=torch.softmax(values, dim=-1)
            q_rows.extend(zip(ids.cpu().tolist(), probs.cpu().tolist()))
    first_by_cycle={}
    for i,row in enumerate(rows):
        key=(row.request_id,row.cycle_id)
        if row.row_index==0: first_by_cycle[key]=i
    cycle_indices=list(first_by_cycle.values())
    min_cycle={}
    for i in cycle_indices:
        row=rows[i]; min_cycle[row.request_id]=min(min_cycle.get(row.request_id,row.cycle_id),row.cycle_id)
    remainder=[i for i in cycle_indices if rows[i].cycle_id!=min_cycle[rows[i].request_id]]
    select=lambda idx: ([rows[i] for i in idx],[q_rows[i] for i in idx])
    marginal=_aggregate(rows,q_rows)
    cr,cq=select(cycle_indices); rr,rq=select(remainder)
    return Tier1CandidateReport('dvi-tier1-candidate-v1',capture_id,result.draft_checkpoint_sha256,marginal,_aggregate(cr,cq),_aggregate(rr,rq))
