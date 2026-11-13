# Common tasks. Inference targets need only `pip install -e .`;
# training targets need `pip install -e ".[train]"`.
DATA ?= data/v2
APOS ?= data/v2_apostrophe
RUN  ?= runs/base
CKPT ?= runs/apostrophe/last.pt

.PHONY: install install-train test lint serve data train finetune evaluate export summary check-converter docker

install:
	pip install -e ".[dev,serve]"
install-train:
	pip install -e ".[dev,serve,train]"
test:
	pytest -q
lint:
	ruff check .
serve:
	uvicorn braillelm.serving.api:app --port 7861
check-converter:
	python -m braillelm.pipelines.check_converter
data:
	python -m braillelm.pipelines.build_data --out $(DATA)
	python -m braillelm.pipelines.build_apostrophe_data --base $(DATA) --out $(APOS)
train:
	python -m braillelm.pipelines.train --config configs/train.yaml --data $(DATA) --out $(RUN)
finetune:
	python -m braillelm.pipelines.train --config configs/finetune_apostrophe.yaml --data $(APOS) --out runs/apostrophe --resume $(RUN)/best.pt
evaluate:
	python -m braillelm.pipelines.evaluate --ckpt $(CKPT) --data $(DATA) --files final_test pilot_plain pilot_extended digits_test $(APOS)/apostrophe_test.jsonl --out reports --train $(DATA)/train.jsonl
export:
	python -m braillelm.pipelines.export_onnx --ckpt $(CKPT)
summary:
	python -m braillelm.pipelines.make_summary
docker:
	docker build -t braillelm:latest .
