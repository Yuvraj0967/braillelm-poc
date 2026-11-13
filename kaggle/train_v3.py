"""Kaggle GPU job: build Grade 1 + Grade 2 data, train BrailleLM v3, evaluate, export.

Push and run from the repository root:

    kaggle kernels push -p kaggle
    kaggle kernels status <user>/braillelm-v3-1-train
    kaggle kernels output <user>/braillelm-v3-1-train -p kaggle_out

Everything worth keeping is written to /kaggle/working/release.
Settings can be changed with the constants below.
"""
import os
import shutil
import subprocess
import sys
import time

BRANCH = "tricky-text"
REPO_URL = "https://github.com/Yuvraj0967/braillelm-poc.git"
N_SENTENCES = 800000
TRAIN_ARGS = []   # extra flags for braillelm.pipelines.train
WORK = "/kaggle/temp/braillelm"      # not saved as output
RELEASE = "/kaggle/working/release"
T0 = time.time()


def sh(cmd, **kw):
    print(f"\n[{(time.time() - T0) / 60:6.1f} min] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


os.makedirs("/kaggle/temp", exist_ok=True)
os.makedirs(RELEASE, exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'NO GPU'")
sh("apt-get install -y -qq liblouis-bin liblouis-data > /dev/null && lou_translate --version | head -1")
sh(f"rm -rf {WORK} && git clone -q -b {BRANCH} {REPO_URL} {WORK} && git -C {WORK} log --oneline | head -1")
sh(f"{sys.executable} -m pip install -q -e {WORK} sacrebleu pyyaml onnx onnxruntime datasets")

py = f"cd {WORK} && {sys.executable} -m braillelm.pipelines"
sh(f"{py}.build_data_g2 --out data/v3 --n {N_SENTENCES}")
sh(f"cp {WORK}/data/v3/meta.json {RELEASE}/data_meta.json")
sh(f"{py}.train --config configs/train_g2.yaml --data data/v3 --out runs/v3 " + " ".join(TRAIN_ARGS))
sh(f"cp {WORK}/runs/v3/train_log.json {RELEASE}/")
sh(f"{py}.evaluate --ckpt runs/v3/best.pt --data data/v3 "
   f"--files final_test pilot_plain pilot_extended digits_test apostrophe_test unusual_test tricky_test everyday_test "
   f"--out {RELEASE}/reports --train data/v3/train.jsonl")
sh(f"{py}.export_onnx --ckpt runs/v3/best.pt --out {RELEASE}/braillelm_v3")
os.makedirs(f"{RELEASE}/test_sets", exist_ok=True)
for name in ["validation", "pilot_plain", "pilot_extended", "final_test", "digits_test", "apostrophe_test",
             "unusual_test", "tricky_test", "everyday_test"]:
    shutil.copy(f"{WORK}/data/v3/{name}.jsonl", f"{RELEASE}/test_sets/{name}.jsonl")
sh(f"ls -la {RELEASE} {RELEASE}/braillelm_v3 && du -sh {RELEASE}")
print(f"finished in {(time.time() - T0) / 60:.1f} min", flush=True)
