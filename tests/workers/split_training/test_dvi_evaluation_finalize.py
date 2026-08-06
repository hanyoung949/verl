from types import SimpleNamespace

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class _Producer:
    evaluation_mode = True
    session_key = ("run", "rollout", "policy")

    def __init__(self):
        self._buffers = {}
        self._evaluation_request_ids = {"21-85ce6315"}
        self.calls = []

    def finalize_request(self, session_key, request_id, prompt, response):
        self.calls.append((session_key, request_id, prompt, response))
        self._evaluation_request_ids.discard(request_id)


def test_evaluation_finalize_resolves_randomized_internal_request_id():
    extension = object.__new__(vLLMColocateWorkerExtension)
    producer = _Producer()
    extension._dvi_stage2_producer = producer
    extension._dvi_finalize_no_match_count = 0
    extension._dvi_finalize_ambiguous_count = 0

    extension.finalize_dvi_request("21", [1], [2])

    assert producer.calls == [(producer.session_key, "21-85ce6315", [1], [2])]
    assert not producer._evaluation_request_ids
    assert extension._dvi_finalize_no_match_count == 0
