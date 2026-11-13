# Common tasks. Inference targets need only `pip install -e ".[serve]"`;
# training targets need `pip install -e ".[train]"`, liblouis and ideally a GPU.
DATA ?= data/v3
RUN  ?= runs/v3
CKPT ?= $(RUN)/best.pt

.PHONY: install install-train test lint serve data train evaluate export summary check-converter docker kaggle

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
	python -m braillelm.pipelines.build_data_g2 --out $(DATA)
train:
	python -m braillelm.pipelines.train --config configs/train_g2.yaml --data $(DATA) --out $(RUN)
evaluate:
	python -m braillelm.pipelines.evaluate --ckpt $(CKPT) --data $(DATA) --files final_test pilot_plain pilot_extended digits_test apostrophe_test unusual_test tricky_test everyday_test --out reports --train $(DATA)/train.jsonl
export:
	python -m braillelm.pipelines.export_onnx --ckpt $(CKPT)
summary:
	python -m braillelm.pipelines.make_summary
kaggle:
	kaggle kernels push -p kaggle
docker:
	docker build -t braillelm:latest .
