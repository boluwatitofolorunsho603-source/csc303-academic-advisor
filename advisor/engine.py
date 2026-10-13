"""Core logic of the Academic Advisor.

Hybrid design:
  * AI part      -> a scikit-learn model (trained on Google Colab, saved as a .pkl)
                    predicts the student's academic standing.
  * Rule part    -> explicit, testable rules (standard software engineering)
                    turn the inputs into concrete, explainable advice.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "advisor_model.pkl"

# name: (min, max, label shown to the user)
FIELD_RULES = {
    "cgpa": (0.0, 5.0, "Current CGPA"),
    "last_gpa": (0.0, 5.0, "Last semester GPA"),
    "attendance": (0.0, 100.0, "Class attendance (%)"),
    "study_hours": (0.0, 80.0, "Study hours per week"),
    "assignment_rate": (0.0, 100.0, "Assignments submitted (%)"),
    "carryovers": (0, 20, "Carry-over courses"),
    "course_units": (1, 40, "Units registered this semester"),
}

CLASS_OF_DEGREE = [  # Nigerian 5.0 scale
    (4.50, "First Class"),
    (3.50, "Second Class Upper"),
    (2.40, "Second Class Lower"),
    (1.50, "Third Class"),
    (1.00, "Pass"),
    (0.00, "Fail / Probation"),
]


class ValidationError(ValueError):
    """Raised when the submitted student profile is invalid."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass
class Advice:
    standing: str
    confidence: float
    probabilities: dict[str, float]
    class_of_degree: str
    recommendations: list[str] = field(default_factory=list)
    target: dict | None = None

    def to_dict(self) -> dict:
        return {
            "standing": self.standing,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "class_of_degree": self.class_of_degree,
            "recommendations": self.recommendations,
            "target": self.target,
        }


def validate_profile(raw: dict) -> dict:
    """Convert form/JSON values to numbers and check ranges."""
    clean, errors = {}, {}
    for name, (lo, hi, label) in FIELD_RULES.items():
        value = raw.get(name)
        if value is None or str(value).strip() == "":
            errors[name] = f"{label} is required."
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue
        if not lo <= number <= hi:
            errors[name] = f"{label} must be between {lo:g} and {hi:g}."
            continue
        clean[name] = int(number) if isinstance(lo, int) else number
    if errors:
        raise ValidationError(errors)
    return clean


def class_of_degree(cgpa: float) -> str:
    for cutoff, name in CLASS_OF_DEGREE:
        if cgpa >= cutoff:
            return name
    return CLASS_OF_DEGREE[-1][1]


def required_gpa(cgpa: float, units_done: float, units_left: float, target: float) -> float | None:
    """GPA needed in the remaining units to finish with `target` CGPA."""
    if units_left <= 0:
        return None
    return (target * (units_done + units_left) - cgpa * units_done) / units_left


def build_recommendations(p: dict, standing: str) -> list[str]:
    """Rule-based advice. Each rule is simple, explainable and unit-tested."""
    recs: list[str] = []

    if p["carryovers"] > 0:
        recs.append(
            f"Clear your {p['carryovers']} carry-over course(s) first: register them before "
            "new electives and ask the lecturers for past questions early in the semester."
        )
    if p["attendance"] < 75:
        recs.append(
            f"Attendance is {p['attendance']:.0f}%. Many departments require 75% to sit exams; "
            "aim for at least 85% and sit near the front."
        )
    if p["study_hours"] < 10:
        recs.append(
            f"You study about {p['study_hours']:.0f} hours a week. Build up to 15+ hours with a "
            "fixed timetable (e.g. 2 hours each weekday evening plus a weekend revision block)."
        )
    if p["assignment_rate"] < 80:
        recs.append(
            f"You submit {p['assignment_rate']:.0f}% of assignments. Continuous assessment is often "
            "30-40% of the grade; track every deadline in a calendar and submit everything."
        )
    if p["last_gpa"] < p["cgpa"] - 0.3:
        recs.append(
            "Your last semester GPA dropped below your CGPA. Review which courses pulled it down "
            "and meet your level adviser this week."
        )
    if p["course_units"] > 24 and standing in ("At Risk", "Needs Support"):
        recs.append(
            f"{p['course_units']} units is a heavy load for your current standing. Consider "
            "dropping an elective so you can focus on core courses."
        )
    if standing == "At Risk":
        recs.append(
            "Book a meeting with your academic adviser and join a study group or tutorial class now; "
            "don't wait for exam week."
        )
    if standing == "Excellent":
        recs.append(
            "You're performing at a high level. Protect it: keep your routine, help peers "
            "(teaching consolidates learning) and look out for scholarships and research opportunities."
        )
    if not recs:
        recs.append(
            "Your habits look solid. Push for the next class of degree by adding one extra "
            "revision session per week for your hardest course."
        )
    return recs


class AcademicAdvisor:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.meta = {k: v for k, v in bundle.items() if k != "model"}

        import sklearn
        trained_with = self.meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            import warnings
            warnings.warn(
                f"Model was trained with scikit-learn {trained_with} but {sklearn.__version__} "
                f"is installed. Run: pip install scikit-learn=={trained_with}",
                stacklevel=2,
            )

    def advise(self, raw: dict, units_done=None, units_left=None, target_cgpa=None) -> Advice:
        profile = validate_profile(raw)
        X = pd.DataFrame([profile])[self.features]

        probs = self.model.predict_proba(X)[0]
        prob_map = {c: round(float(p), 3) for c, p in zip(self.model.classes_, probs)}
        standing = max(prob_map, key=prob_map.get)

        target = None
        if units_done and units_left and target_cgpa:
            try:
                done, left, tgt = float(units_done), float(units_left), float(target_cgpa)
            except (TypeError, ValueError):
                done = left = tgt = 0
            if done >= 0 and left > 0 and 0 < tgt <= 5:
                need = required_gpa(profile["cgpa"], done, left, tgt)
                target = {
                    "target_cgpa": tgt,
                    "target_class": class_of_degree(tgt),
                    "required_gpa": round(need, 2),
                    "achievable": need <= 5.0,
                }

        return Advice(
            standing=standing,
            confidence=prob_map[standing],
            probabilities=prob_map,
            class_of_degree=class_of_degree(profile["cgpa"]),
            recommendations=build_recommendations(profile, standing),
            target=target,
        )
