"""Optional experiment tracking.

Training and export call this thin wrapper. With no experiment name it does
nothing, so the model code never depends on a tracking tool. MLflow is the
backend today; swapping in Weights & Biases, Comet or Neptune means changing
this one file.

    python -m braillelm.pipelines.train --config configs/train.yaml ... --mlflow braillelm
    mlflow ui          # then open http://127.0.0.1:5000

Set MLFLOW_TRACKING_URI to log to a shared server instead of ./mlruns.
"""
from contextlib import contextmanager


class _Off:
    def params(self, d): pass
    def metrics(self, d, step=None): pass
    def artifact(self, path): pass
    def register_onnx(self, onnx_path, name): pass


class _MLflow:
    def __init__(self, mlflow):
        self.m = mlflow

    def params(self, d):
        self.m.log_params({k: v for k, v in d.items() if v is not None})

    def metrics(self, d, step=None):
        self.m.log_metrics({k: float(v) for k, v in d.items() if isinstance(v, (int, float))}, step=step)

    def artifact(self, path):
        self.m.log_artifact(str(path))

    def register_onnx(self, onnx_path, name):
        import onnx
        self.m.onnx.log_model(onnx.load(str(onnx_path)), name="model", registered_model_name=name)


@contextmanager
def run(experiment=None, run_name=None):
    """Yield a tracker. `experiment=None` (the default) disables tracking."""
    if not experiment:
        yield _Off()
        return
    import mlflow
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield _MLflow(mlflow)
