from pathlib import Path
import yaml
from ruleshift.permissions import find_forbidden_fields, scan_package
from ruleshift.public_api.grid_rules import GRID_HYPOTHESES
from ruleshift.public_api.grid_schemas import GRID_PUBLIC_OBSERVATION_FIELDS
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN


def test_agent_and_public_api_packages_have_no_forbidden_imports():
    for package in ("src/ruleshift/agents", "src/ruleshift/public_api"):
        assert scan_package(Path(package)) == []


def test_grid_rules_are_public_descriptions_only():
    for rule in GRID_HYPOTHESES:
        payload = {"rule_id": rule.rule_id, "description": rule.description}
        assert find_forbidden_fields(payload) == []


def test_grid_observation_fields_cover_the_required_public_fields():
    required = {
        "observation_id", "public_obs_hash", "public_history_hash",
        "public_history", "public_text", "available_actions",
        "candidate_rules", "public_costs", "image_hash",
    }
    assert required <= set(GRID_PUBLIC_OBSERVATION_FIELDS)


def test_grid_actions_do_not_widen_the_macro_set():
    from ruleshift.public_api.grid_schemas import GRID_ACTIONS
    assert not (set(GRID_ACTIONS) & MACRO_ACTIONS_FROZEN)


def test_preregistered_config_is_still_a_draft():
    data = yaml.safe_load(
        Path("configs/preregistered_test.yaml").read_text(encoding="utf-8")
    )
    assert data["status"] == "DRAFT_NOT_FROZEN"
