# Scalar AtomWise training implementation

The archived Fig. 2(d) run uses scalar band-gap regression. The archived config records criterion=mse and scheduler=onecycle, but compatible official AtomWise implementations override the criterion with batch-mean L1 (MAE) for backpropagation. The auxiliary losses were inactive. The epoch history accumulates batch losses; it is not directly the sample-weighted epoch MAE.

AdamW is recreated after OneCycleLR is constructed. The scheduler remains attached to the first optimizer, while the second optimizer performs the actual updates at a constant effective learning rate of 5e-4. The recorded warmup and early-stopping fields do not implement effective warmup or early stopping in this branch. Fifty epochs were recorded, with the minimum summed validation batch loss at epoch 50. Weight decay is 1e-4 for the decay group and zero for parameter names containing bias, bn or norm; this naming rule is not a universal identification of all normalization weights.

These are implementation-based historical reconstructions, supported by the archived artifacts, the author's confirmation of unmodified official code, matching checkpoint/configuration structure and compatible upstream implementations. Stepwise learning-rate measurements are not available. Compatible releases include v2024.12.12 and v2025.4.1; the exact installed release/commit, full historical Python/PyTorch/DGL environment and stepwise LR log were not retained. No unique release or end-to-end numerical replay is asserted.

Official reference: https://github.com/usnistgov/alignn/blob/v2024.12.12/alignn/train.py and https://github.com/usnistgov/alignn/blob/v2024.12.12/alignn/utils.py .

Original config, best/current/last checkpoints, train/validation histories, predictions and fixed splits are in data/historical_training/. All are unchanged archived files. The public source is a reference implementation, not proof of a bitwise replay under a newly installed environment.
