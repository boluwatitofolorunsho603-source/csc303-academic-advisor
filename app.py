"""Flask web application that serves the Academic Advisor model (.pkl)."""
from flask import Flask, jsonify, render_template, request

from advisor import FIELD_RULES, AcademicAdvisor, ValidationError

app = Flask(__name__)
advisor = AcademicAdvisor()

TARGET_FIELDS = ("units_done", "units_left", "target_cgpa")


@app.get("/")
def index():
    return render_template("index.html", fields=FIELD_RULES, values={}, errors={}, meta=advisor.meta)


@app.post("/advise")
def advise():
    form = request.form.to_dict()
    try:
        result = advisor.advise(form, *(form.get(k) for k in TARGET_FIELDS))
    except ValidationError as exc:
        return render_template(
            "index.html", fields=FIELD_RULES, values=form, errors=exc.errors, meta=advisor.meta
        ), 400
    return render_template("result.html", result=result, values=form, fields=FIELD_RULES, meta=advisor.meta)


@app.post("/api/advise")
def api_advise():
    data = request.get_json(silent=True) or {}
    try:
        result = advisor.advise(data, *(data.get(k) for k in TARGET_FIELDS))
    except ValidationError as exc:
        return jsonify({"errors": exc.errors}), 400
    return jsonify(result.to_dict())


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": advisor.meta.get("model_name"),
                    "test_accuracy": advisor.meta.get("test_accuracy")})


if __name__ == "__main__":
    app.run(debug=True)
