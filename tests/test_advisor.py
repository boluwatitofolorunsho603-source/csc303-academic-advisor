"""Unit and integration tests for the Academic Advisor (run with: pytest)."""
import pytest

from advisor.engine import (
    AcademicAdvisor,
    ValidationError,
    build_recommendations,
    class_of_degree,
    required_gpa,
    validate_profile,
)
from app import app

GOOD = {
    "cgpa": 4.6, "last_gpa": 4.7, "attendance": 95, "study_hours": 22,
    "assignment_rate": 100, "carryovers": 0, "course_units": 22,
}
WEAK = {
    "cgpa": 1.4, "last_gpa": 1.0, "attendance": 45, "study_hours": 3,
    "assignment_rate": 40, "carryovers": 4, "course_units": 28,
}


@pytest.fixture(scope="module")
def advisor():
    return AcademicAdvisor()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---- validation -------------------------------------------------------------
def test_validate_converts_strings():
    clean = validate_profile({k: str(v) for k, v in GOOD.items()})
    assert clean["cgpa"] == 4.6 and clean["carryovers"] == 0


def test_validate_rejects_out_of_range_and_missing():
    bad = dict(GOOD, cgpa=5.8)
    bad.pop("attendance")
    with pytest.raises(ValidationError) as exc:
        validate_profile(bad)
    assert set(exc.value.errors) == {"cgpa", "attendance"}


def test_validate_rejects_text():
    with pytest.raises(ValidationError):
        validate_profile(dict(GOOD, study_hours="many"))


# ---- rules ------------------------------------------------------------------
@pytest.mark.parametrize("cgpa,expected", [
    (4.5, "First Class"), (3.5, "Second Class Upper"), (2.4, "Second Class Lower"),
    (1.5, "Third Class"), (1.0, "Pass"), (0.5, "Fail / Probation"),
])
def test_class_of_degree_boundaries(cgpa, expected):
    assert class_of_degree(cgpa) == expected


def test_required_gpa():
    # 3.0 over 100 units, want 3.5 over 150 units -> need 4.5 in remaining 50
    assert required_gpa(3.0, 100, 50, 3.5) == pytest.approx(4.5)
    assert required_gpa(3.0, 100, 0, 3.5) is None


def test_rules_flag_weak_habits():
    recs = " ".join(build_recommendations(validate_profile(WEAK), "At Risk"))
    for keyword in ("carry-over", "Attendance", "study", "assignments", "adviser"):
        assert keyword.lower() in recs.lower()


def test_rules_always_return_something():
    assert build_recommendations(validate_profile(GOOD), "On Track")


# ---- model (.pkl) -----------------------------------------------------------
def test_model_loads_with_metadata(advisor):
    assert advisor.features == list(GOOD)
    assert advisor.meta["test_accuracy"] > 0.8


def test_model_separates_strong_and_weak(advisor):
    assert advisor.advise(GOOD).standing == "Excellent"
    assert advisor.advise(WEAK).standing == "At Risk"


def test_probabilities_sum_to_one(advisor):
    assert sum(advisor.advise(GOOD).probabilities.values()) == pytest.approx(1.0, abs=0.01)


def test_target_calculation(advisor):
    result = advisor.advise(dict(GOOD, cgpa=3.0), units_done=100, units_left=50, target_cgpa=3.5)
    assert result.target["required_gpa"] == pytest.approx(4.5)
    assert result.target["achievable"]


# ---- web app ----------------------------------------------------------------
def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Get my advice" in r.data


def test_form_submission(client):
    r = client.post("/advise", data=WEAK)
    assert r.status_code == 200 and b"At Risk" in r.data


def test_form_shows_errors(client):
    r = client.post("/advise", data=dict(WEAK, cgpa="9"))
    assert r.status_code == 400 and b"between 0 and 5" in r.data


def test_json_api(client):
    r = client.post("/api/advise", json=GOOD)
    body = r.get_json()
    assert r.status_code == 200 and body["standing"] == "Excellent"
    assert body["recommendations"]


def test_json_api_errors(client):
    r = client.post("/api/advise", json={})
    assert r.status_code == 400 and "cgpa" in r.get_json()["errors"]


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"
