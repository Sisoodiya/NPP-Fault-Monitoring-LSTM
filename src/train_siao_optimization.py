import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from models import CNN_LSTM_Model
from siao_optimizer import SIAO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 100
BATCH_SIZE = 16

# 1. Load windows & labels (same as before)
windows = np.load("../windows.npy")   # (N, 1000, F)
labels = np.load("../labels.npy")     # (N,)

num_windows, ws, num_features = windows.shape
assert ws == WINDOW_SIZE

# 2. Compute stats (mean,median,std,var,entropy,wks with omega=1)
#    You could reuse code from train_backprop or features.py
from features import compute_basic_stats, compute_wks, compute_kurtosis_skewness

stat_feats_list = []
for i in range(num_windows):
    w = windows[i, :, :]
    stats_basic = compute_basic_stats(w)   # (5, F)
    # We can recompute Kt, Sk if we want to refine omega later
    wks = compute_wks(w, omega=1.0)
    combined = np.concatenate([stats_basic.reshape(-1), wks])
    stat_feats_list.append(combined)
stat_feats = np.vstack(stat_feats_list)  # (N, 6*F)

# 3. Build a small validation loader (e.g. 10% of the data)
indices = np.arange(num_windows)
np.random.seed(42)
np.random.shuffle(indices)
val_subset = indices[: int(0.1 * num_windows)]  # 10% for SIAO fitness
val_sampler = SubsetRandomSampler(val_subset)
val_ds = torch.utils.data.TensorDataset(torch.tensor(windows, dtype=torch.float32),
                                         torch.tensor(stat_feats, dtype=torch.float32),
                                         torch.tensor(labels, dtype=torch.long))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler)

# 4. Initialize model and load pretrained weights
model = CNN_LSTM_Model(num_features=num_features,
                       window_size=WINDOW_SIZE,
                       stat_feature_dim=stat_feats.shape[1],
                       num_classes=len(np.unique(labels)))
model = model.to(DEVICE)
model.load_state_dict(torch.load("../models/best_cnn_lstm.pth", map_location=DEVICE))

# 5. Freeze CNN parameters; only allow LSTM + FC to be changed
for name, param in model.named_parameters():
    if "cnn_encoder" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 6. Extract dimension of LSTM + FC weights
flat_init = []
for name, p in model.named_parameters():
    if p.requires_grad and (("lstm" in name) or ("classifier.fc" in name)):
        flat_init.append(p.detach().cpu().numpy().ravel())
flat_init_vector = np.concatenate(flat_init)
dim = flat_init_vector.shape[0]

# 7. Instantiate SIAO
from siao_optimizer import SIAO

siao_optimizer = SIAO(model=model,
                      val_loader=val_loader,
                      device=DEVICE,
                      population_size=20,
                      max_iters=50,
                      dim=dim,
                      lower_bound=-0.05 * np.ones(dim),  # ±5% around pretrained weights
                      upper_bound= 0.05 * np.ones(dim))

# 8. Run optimization and track progress
print("Starting SIAO optimization...")
import matplotlib.pyplot as plt

# Modify SIAO to track iteration history
iteration_history = []
rmse_history = []

# Store original optimize method
original_optimize = siao_optimizer.optimize

def optimize_with_tracking():
    """Modified optimize method that tracks progress"""
    # 1. Evaluate initial fitness for each particle
    for i in range(siao_optimizer.N):
        siao_optimizer.fitness[i] = siao_optimizer._compute_fitness(siao_optimizer.X[i, :])
        if siao_optimizer.fitness[i] < siao_optimizer.best_fitness:
            siao_optimizer.best_fitness = siao_optimizer.fitness[i]
            siao_optimizer.best_solution = siao_optimizer.X[i, :].copy()
            siao_optimizer.best_index = i

    # 2. Main loop
    for t in range(1, siao_optimizer.T + 1):
        # Compute mean position X_M
        X_M = siao_optimizer.X.mean(axis=0)

        for i in range(siao_optimizer.N):
            Xi = siao_optimizer.X[i].copy()       # current particle
            Xb = siao_optimizer.best_solution     # best solution so far
            r = np.random.rand()
            rand = np.random.rand()

            # Decide which step to take based on t/T (divide T into 4 phases)
            if t <= 0.2 * siao_optimizer.T:
                # Step 1: Expanded Exploration (Eq 6)
                new_Xi = (Xb * (1 - t/siao_optimizer.T)
                          + (X_M - Xb) * rand)
            elif t <= 0.4 * siao_optimizer.T:
                # Step 2: Narrowed Exploration (Eq 8)
                levy = siao_optimizer._levy_flight(siao_optimizer.D, beta=1.5)
                X_R = siao_optimizer.X[np.random.randint(0, siao_optimizer.N)]  # random hawk
                new_Xi = Xb * levy + X_R + (r - rand) * rand
            elif t <= 0.7 * siao_optimizer.T:
                # Step 3: Expanded Exploitation (Eq 12)
                alpha = 0.1
                Ub = siao_optimizer.ub
                Lb = siao_optimizer.lb
                RAND = np.random.rand(siao_optimizer.D)
                delta = 0.1
                new_Xi = (Xb - X_M) * alpha - rand + ((Ub - Lb) * RAND + Lb) * delta
            else:
                # Step 4 & 5: Narrowed Exploitation + Chaotic QF (Eq 13 & 14)
                H1 = 2 * np.random.rand() - 1
                H2 = 2 * (1 - t/siao_optimizer.T)
                # QF(t) from chaotic map
                QF_t = siao_optimizer._chaotic_map_qf(t)
                X_P = siao_optimizer.X[np.random.randint(0, siao_optimizer.N)]  # hawk's random pos
                levy = siao_optimizer._levy_flight(siao_optimizer.D, beta=1.5)
                new_Xi = (QF_t * Xb
                          - (H1 * X_P * rand)
                          - H2 * levy
                          + rand * H1)

            # 3. Boundary handling (clip to [lb, ub])
            new_Xi = np.clip(new_Xi, siao_optimizer.lb, siao_optimizer.ub)

            # 4. Compute new fitness
            new_fit = siao_optimizer._compute_fitness(new_Xi)

            # 5. Greedy selection
            if new_fit < siao_optimizer.fitness[i]:
                siao_optimizer.X[i, :] = new_Xi
                siao_optimizer.fitness[i] = new_fit
                # Update global best
                if new_fit < siao_optimizer.best_fitness:
                    siao_optimizer.best_fitness = new_fit
                    siao_optimizer.best_solution = new_Xi.copy()
                    siao_optimizer.best_index = i

        # Track progress
        iteration_history.append(t)
        rmse_history.append(siao_optimizer.best_fitness)
        print(f"[Iter {t}/{siao_optimizer.T}] Best RMSE: {siao_optimizer.best_fitness:.6f}")

    # After loop, best_solution holds the optimized weight vector
    return siao_optimizer.best_solution, siao_optimizer.best_fitness

# Run optimization with tracking
best_vector, best_rmse = optimize_with_tracking()
print("Finished SIAO. Best RMSE:", best_rmse)

# 9. Write best_vector back into model’s parameters
siao_optimizer._set_flat_weights(best_vector)

# 10. Save final SIAO-tuned model
torch.save(model.state_dict(), "../models/cnn_lstm_siao_tuned.pth")
print("Saved SIAO-tuned model to ../models/cnn_lstm_siao_tuned.pth")

# 11. Plot SIAO optimization progress
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(iteration_history, rmse_history, 'b-', linewidth=2, marker='o', markersize=4)
plt.title('SIAO Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Best RMSE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/siao_optimization_progress.png", dpi=200, bbox_inches='tight')
print("SIAO optimization progress saved to ../results/siao_optimization_progress.png")
plt.show()
