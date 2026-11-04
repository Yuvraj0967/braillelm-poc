# Training and evaluation logs

The `.log` files are copied from the local Mac experiment without editing.
They contain progress steps, measured validation losses, elapsed times, and
warnings from the Python dependencies. The selected Qwen model's training
chain is:

1. `qwen_bkft_scaled_epoch1.log` — fresh BKFT initialized Qwen run,
   7,943 microbatches, validation loss about 0.2354.
2. `qwen_bkft_scaled_epoch2.log` — continuation; selected interim
   `step4000` saved at validation loss about 0.225.
3. `qwen_bkft_continue4000.log` — second continuation; selected final
   `step4000` saved at validation loss about 0.2155.

The four-beam plain and extended development logs are
`eval_qwen_continue4000_beam4_plain.log` and
`eval_qwen_continue4000_beam4_extended_pilot.log`. The exact metric JSON
and hypotheses are in `../results/`.
