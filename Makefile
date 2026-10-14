.PHONY: install test test-data lint compiler train export eval bench serve

PY ?= .venv/bin/python
REEF ?= .venv/bin/reef
RUN ?= runs/reef-tiny
ART ?= artifacts/reef-tiny

install:
	uv venv -p 3.12 && uv pip install -e ".[train]" --group dev

test:
	$(PY) -m pytest -q

test-data:
	REEF_TEST_DATA=1 $(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

# edgetpu_compiler is x86-64 Linux only; this image runs it anywhere Docker does.
compiler:
	docker build --platform linux/amd64 -t reef-edgetpu-compiler -f docker/compiler.Dockerfile docker

train:
	$(REEF) train --out $(RUN)

export:
	$(REEF) export $(RUN) $(ART)

eval:
	$(REEF) eval $(ART) --device cpu --json $(ART)/eval_cpu.json
	$(REEF) eval $(ART) --device edgetpu --json $(ART)/eval_edgetpu.json

bench:
	$(REEF) bench $(ART) --json $(ART)/bench.json

serve:
	$(REEF) serve $(ART)
