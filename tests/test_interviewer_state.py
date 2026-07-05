from sentinel.agents import interviewer as iv
from sentinel.agents.state import InterviewConfig


class OneShotProvider:
    """Fake provider returning a scripted JSON string per complete() call."""

    def __init__(self, replies):
        self._it = iter(replies)
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        return next(self._it)


def test_gate_accept_fills_all_defaults_in_one_turn():
    p = OneShotProvider(['{"all_defaults": true}'])
    prog = iv.advance(iv.start_progress(), "yes just use defaults", p)
    assert prog["phase"] == "done"
    assert isinstance(prog["config"], InterviewConfig)
    assert prog["config"].failure_threshold == iv.DEFAULTS["failure_threshold"]
    assert p.calls == 1  # exactly one classifier call this turn
    assert any("default" in n.lower() for n in prog["notices"])


def test_gate_decline_then_clear_answer_advances_one_field():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no let's go through it", OneShotProvider(['{"all_defaults": false}']))
    assert prog["phase"] == "field"
    assert prog["active_index"] == 0
    clear = '{"classification":"CLEAR","reply":"Got it.","value":"turbofan RUL","deduced":[]}'
    prog = iv.advance(prog, "predict turbofan RUL", OneShotProvider([clear]))
    assert prog["values"]["framing"] == "turbofan RUL"
    assert prog["active_index"] == 1  # moved to failure_threshold


def test_all_defaults_midway_fills_the_rest():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", OneShotProvider(['{"all_defaults": false}']))
    ad = '{"classification":"ALL_DEFAULTS","reply":"ok defaults","value":null,"deduced":[]}'
    prog = iv.advance(prog, "just use defaults for the rest", OneShotProvider([ad]))
    assert prog["phase"] == "done"
    assert prog["config"].success_metric == iv.DEFAULTS["success_metric"]
