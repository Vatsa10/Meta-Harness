from meta_harness.cc_harness import AgentConfig
from meta_harness.harness_store import Artifact
from meta_harness.learn import config_with, decide_retention


def _artifact(type="doctrine", payload="Read before you edit."):
    return Artifact(id="a1", type=type, origin={"signature": "s"}, payload=payload, replay={})


def test_doctrine_lands_in_the_system_prompt():
    config = config_with(_artifact(), AgentConfig(append_system_prompt="base"))
    assert "base" in config.append_system_prompt
    assert "Read before you edit." in config.append_system_prompt


def test_skill_lands_in_the_skills_map():
    config = config_with(_artifact(type="skill", payload="Body of the skill."))
    assert "Body of the skill." in "".join(config.skills.values())


def test_injection_is_appended_to_the_framing():
    config = config_with(_artifact(type="injection", payload="Remember the encoding."),
                         AgentConfig(prompt_template="{instruction}"))
    assert "{instruction}" in config.prompt_template
    assert "Remember the encoding." in config.prompt_template


def test_a_rule_does_not_change_the_agent_config():
    # Rules act through tool.check, not through the prompt; applying one must be a no-op here.
    base = AgentConfig(append_system_prompt="base")
    assert config_with(_artifact(type="rule"), base).append_system_prompt == "base"


def test_retention_requires_the_origin_to_be_fixed():
    decision = decide_retention(origin_fixed=False, candidate_score=1.0, baseline_score=1.0,
                                candidate_context=100.0, baseline_context=200.0)
    assert decision["kept"] is False
    assert "origin" in decision["reason"]


def test_retention_rejects_a_regression():
    decision = decide_retention(origin_fixed=True, candidate_score=0.8, baseline_score=1.0,
                                candidate_context=100.0, baseline_context=200.0)
    assert decision["kept"] is False
    assert "regress" in decision["reason"]


def test_retention_keeps_a_fix_that_holds_score():
    decision = decide_retention(origin_fixed=True, candidate_score=1.0, baseline_score=1.0,
                                candidate_context=210.0, baseline_context=200.0)
    # Context is the tiebreak, not a veto: fixing the failure is the point.
    assert decision["kept"] is True


def test_retention_notes_a_context_win():
    decision = decide_retention(origin_fixed=True, candidate_score=1.0, baseline_score=1.0,
                                candidate_context=150.0, baseline_context=200.0)
    assert decision["kept"] is True
    assert decision["context_delta"] == -50.0
