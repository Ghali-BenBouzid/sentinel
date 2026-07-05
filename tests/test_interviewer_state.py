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


def test_advance_does_not_mutate_caller_snapshot():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", OneShotProvider(['{"all_defaults": false}']))
    clear = '{"classification":"CLEAR","reply":"Got it.","value":"turbofan RUL","deduced":[]}'
    snap1 = iv.advance(prog, "predict turbofan RUL", OneShotProvider([clear]))
    # A second turn off the same snapshot must not reach back and mutate snap1.
    clear2 = '{"classification":"CLEAR","reply":"ok","value":"42","deduced":[]}'
    iv.advance(snap1, "42", OneShotProvider([clear2]))
    assert snap1["values"] == {"framing": "turbofan RUL"}  # unchanged by the later turn
    assert "failure_threshold" not in snap1["values"]


def test_question_does_not_advance_or_consume():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", OneShotProvider(['{"all_defaults": false}']))
    q = '{"classification":"QUESTION","reply":"RMSE is your typical error. So - what are we predicting?","value":null,"deduced":[]}'
    prog = iv.advance(prog, "what does RMSE mean?", OneShotProvider([q]))
    assert prog["phase"] == "field"
    assert prog["active_index"] == 0  # did not advance
    assert prog["nonanswers"] == 0  # a question is not a non-answer
    assert "framing" not in prog["values"]  # not consumed
    assert prog["next_prompt"] == "RMSE is your typical error. So - what are we predicting?"


def test_unclear_builds_to_max_nonanswers_then_defaults():
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", OneShotProvider(['{"all_defaults": false}']))
    unclear = '{"classification":"UNCLEAR","reply":"I need a clearer answer.","value":null,"deduced":[]}'
    for _ in range(iv.MAX_NONANSWERS):
        prog = iv.advance(prog, "dunno", OneShotProvider([unclear]))
    assert prog["values"]["framing"] == iv.DEFAULTS["framing"]  # fell back to default
    assert prog["active_index"] == 1  # advanced past the stuck field
