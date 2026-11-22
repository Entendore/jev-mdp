PY    ?= .venv/bin/python
PIP   ?= .venv/bin/pip
VENV  ?= .venv

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -e ".[server,dev]"

demo:
	$(PY) examples/ex01_end_to_end.py

examples: demo
	$(PY) examples/ex02_serving_ladder.py
	$(PY) examples/ex03_feedback_loop.py
	$(PY) examples/ex04_cost_transfer.py
	$(PY) examples/ex05_compile_verify.py
	$(PY) examples/ex06_incontext_zeroshot.py
	$(PY) examples/ex07_ope_gates.py

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check jev

serve:
	$(PY) -m jev.cli serve --demo-engine --port 8000

clean:
	rm -rf $(VENV) .pytest_cache artifacts/* **/__pycache__

.PHONY: venv install demo examples test lint serve clean
