"""
Self-Improved Aquila Optimization (SIAO) Algorithm Implementation
Based on: "Self-improved Aquila Optimization Algorithm" paper

This implementation provides both standard SIAO and an enhanced version with
additional features for neural network optimization.
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
from copy import deepcopy
from scipy.stats import levy_stable
from scipy.special import gamma
import math

class SIAO:
    def __init__(self, model, val_loader, device,
                 population_size=20,
                 max_iters=50,
                 dim=None,
                 lower_bound=-0.1,  # relative to pretrained weights
                 upper_bound=0.1):
        """
        model: the pretrained PyTorch model (CNN+LSTM) instance
        val_loader: DataLoader over a small subset of validation data
        device: torch.device
        """
        self.model = model
        self.device = device
        self.val_loader = val_loader
        self.pop_size = population_size
        self.max_iters = max_iters

        # 1. Extract current LSTM & FC weights as a single flat vector
        self.init_weight_vector = self._get_flat_weights()
        self.dim = dim if dim is not None else self.init_weight_vector.shape[0]

        # 2. Set search bounds around pretrained weights
        self.lb = self.init_weight_vector + lower_bound  # vectorized
        self.ub = self.init_weight_vector + upper_bound

        # 3. Initialize population of positions X (shape: pop_size × dim)
        #    We sample uniformly between [lb, ub] on each dimension
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.dim))
        # 4. Initialize velocities or other arrays as needed
        self.fitness = np.full((self.pop_size,), np.inf)
        self.best_index = None
        self.best_solution = None
        self.best_fitness = np.inf

        # 5. Precompute some parameters
        self.N = self.pop_size
        self.D = self.dim
        self.T = self.max_iters

    def _get_flat_weights(self):
        """
        Extract all LSTM + FC parameters into a single 1D numpy array.
        """
        params = []
        for name, p in self.model.named_parameters():
            # Only optimize LSTM + FC layers (exclude CNN & batchnorm, for speed)
            if "lstm" in name or "classifier.fc" in name:
                params.append(p.detach().cpu().numpy().ravel())
        flat = np.concatenate(params)
        return flat

    def _set_flat_weights(self, flat_vector):
        """
        Writes a flat_vector back into the model’s LSTM + FC parameters.
        Must match the order used in _get_flat_weights.
        """
        index = 0
        for name, p in self.model.named_parameters():
            if "lstm" in name or "classifier.fc" in name:
                shape = p.data.shape
                numel = p.data.numel()
                new_vals = flat_vector[index : index + numel]
                new_tensor = torch.from_numpy(new_vals.reshape(shape)).to(self.device)
                p.data.copy_(new_tensor)
                index += numel
        assert index == flat_vector.shape[0]

    def _compute_fitness(self, flat_vector):
        """
        Compute fitness using RMSE as per equation (5) in the SIAO paper:
        RMSE = √[(1/n) * Σ(y_i - ŷ_i)²]
        
        Args:
            flat_vector: Weight vector to evaluate
            
        Returns:
            float: RMSE fitness value (lower is better)
        """
        # 1. Set weights
        self._set_flat_weights(flat_vector)
        self.model.eval()
        
        total_squared_error = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i, (X_win, X_stat, y_true) in enumerate(self.val_loader):
                X_win = X_win.to(self.device)
                X_stat = X_stat.to(self.device)
                y_true = y_true.to(self.device)
                
                # Forward pass
                logits = self.model(X_win, X_stat)  # (B, C)
                probs = F.softmax(logits, dim=1)    # (B, C)
                
                # Convert to one-hot for RMSE calculation
                y_onehot = F.one_hot(y_true, num_classes=probs.shape[1]).float()
                
                # Calculate squared error: (y_i - ŷ_i)²
                squared_error = (y_onehot - probs).pow(2).sum().item()
                total_squared_error += squared_error
                total_samples += y_true.size(0) * probs.shape[1]  # n samples × classes
                
                # Limit evaluation for speed (only first 5 batches)
                if i >= 5:
                    break
        
        # Calculate RMSE as per equation (5)
        rmse = np.sqrt(total_squared_error / total_samples)
        return rmse

    def _levy_flight(self, size, beta=1.5):
        """
        Generate a Lévy flight step vector based on equation (9) and (10) from the SIAO paper.
        
        Equation (9): Levy(D) = s × (u/|v|^(1/β))
        Equation (10): σ = [(Γ(1+β) × sin(πβ/2)) / (Γ((1+β)/2) × β × 2^((β-1)/2))]^(1/β)
        
        Args:
            size: Dimension of the step vector
            beta: Lévy distribution parameter (typically 1.5)
        
        Returns:
            numpy array: Lévy flight step vector
        """
        # Calculate σ using equation (10)
        numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (numerator / denominator) ** (1 / beta)
        
        # Generate random variables u and v
        u = np.random.normal(0, sigma, size)  # u ~ N(0, σ²)
        v = np.random.normal(0, 1, size)      # v ~ N(0, 1)
        
        # Avoid division by zero
        v = np.where(np.abs(v) < 1e-10, 1e-10, v)
        
        # Calculate Lévy step using equation (9): s × (u/|v|^(1/β))
        # s is typically set to 0.01 for step size control
        s = 0.01
        step = s * (u / (np.abs(v) ** (1 / beta)))
        
        # Clip extreme values to prevent numerical issues
        step = np.clip(step, -10, 10)
        return step

    def _chaotic_map_qf(self, t):
        """
        Implement chaotic quality function QF(t) based on SIAO paper.
        Uses the three chaotic-based mathematical formulas from equations (17), (18), (19).
        
        Args:
            t: Current iteration
            
        Returns:
            float: Quality function value
        """
        # Three chaotic formulas from the SIAO paper
        import random
        choice = random.choice([17, 18, 19])
        
        if choice == 17:
            # Equation (17): QF = 1.95 - 2^(t/T^(1/4)) / (t/T^(1/3))
            t_ratio = t / self.T
            if t_ratio == 0:
                return 1.95
            numerator = 2 ** (t_ratio ** (1/4))
            denominator = t_ratio ** (1/3)
            return 1.95 - numerator / (denominator + 1e-8)
            
        elif choice == 18:
            # Equation (18): QF = 1.95 - 2^(t/T^(1/3)) / (t/T^(1/4))
            t_ratio = t / self.T
            if t_ratio == 0:
                return 1.95
            numerator = 2 ** (t_ratio ** (1/3))
            denominator = t_ratio ** (1/4)
            return 1.95 - numerator / (denominator + 1e-8)
            
        else:  # choice == 19
            # Equation (19): QF = 0.5 + 2*exp(-(4*t/T)^2)
            t_ratio = t / self.T
            return 0.5 + 2 * np.exp(-((4 * t_ratio) ** 2))

    def _self_improvement_mechanism(self, Xi, Xb, t):
        """
        Self-improvement mechanism for SIAO based on the paper's principles.
        Applies adaptive improvements to enhance convergence.
        
        Args:
            Xi: Current solution
            Xb: Best solution  
            t: Current iteration
            
        Returns:
            numpy array: Improved solution
        """
        # Calculate improvement factor based on iteration progress
        improvement_factor = np.exp(-2 * t / self.T)  # Exponential decay
        
        # Only apply improvement if current solution is not the best
        if not np.array_equal(Xi, Xb):
            # Calculate direction toward best solution with adaptive step
            direction = Xb - Xi
            
            # Apply self-improvement with decreasing intensity
            step_size = improvement_factor * np.random.uniform(0.05, 0.15)
            improved_Xi = Xi + step_size * direction
            
            # Add small chaotic perturbation for diversity maintenance
            chaos_factor = self._chaotic_map_qf(t) * 0.01 * improvement_factor
            perturbation = np.random.uniform(-chaos_factor, chaos_factor, self.D)
            improved_Xi += perturbation
            
            # Ensure solution stays within bounds
            improved_Xi = np.clip(improved_Xi, self.lb, self.ub)
            
            return improved_Xi
        
        return Xi

    def optimize(self):
        """
        Main SIAO optimization loop implementing the complete algorithm.
        
        SIAO Phases:
        1. Expanded Exploration (0-20% of iterations)
        2. Narrowed Exploration (20-50% of iterations)  
        3. Expanded Exploitation (50-80% of iterations)
        4. Narrowed Exploitation (80-100% of iterations)
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        print(f"Starting SIAO optimization with {self.N} agents, {self.T} iterations, {self.D} dimensions")
        
        # 1. Evaluate initial population fitness
        for i in range(self.N):
            self.fitness[i] = self._compute_fitness(self.X[i, :])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.X[i, :].copy()
                self.best_index = i
        
        print(f"Initial best fitness: {self.best_fitness:.6f}")

        # 2. Main optimization loop
        for t in range(1, self.T + 1):
            # Calculate mean position using equation (7): X_m(t) = (1/N) * Σ(X_i(t)) for i=1 to D
            X_M = self.X.mean(axis=0)  # Mean position of all Aquilas
            
            # Calculate iteration ratio
            t_ratio = t / self.T
            
            for i in range(self.N):
                Xi = self.X[i].copy()
                Xb = self.best_solution
                
                # Generate random values
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                r4 = np.random.rand()

                # SIAO Phase Selection based on the paper equations
                if t_ratio <= 0.2:
                    # Phase 1: Expanded Exploration (High-soar with vertical stoop)
                    # Equation (6): X_i(t+1) = X_best(t) * (1 - t/T_max) + (X_m(t) - X_best(t)) * rand
                    new_Xi = Xb * (1 - t / self.T) + (X_M - Xb) * r1
                    
                elif t_ratio <= 0.5:
                    # Phase 2: Narrowed Exploration (Contour flight with short glide)
                    # Equation (8): X_i(t+1) = X_best(t) * Levy(D) + X_R(t) + (r-1) * rand
                    levy = self._levy_flight(self.D, beta=1.5)
                    X_R = self.X[np.random.randint(0, self.N)]  # Random hawk's position
                    new_Xi = Xb * levy + X_R + (r1 - 1) * r2
                    
                elif t_ratio <= 0.8:
                    # Phase 3: Expanded Exploitation (Low flight with slow descent)
                    # Equation (12): X_i(t+1) = (X_best(t) - X_m(t)) * α - rand + ((U_i - L_i) * RAND + L_i) * δ
                    alpha = 0.1  # Exploitation modification parameter (set to 0.1 as per paper)
                    delta = 0.1  # Parameters for exploitation modification (set to 0.1 as per paper)
                    U_i = self.ub  # Upper bounds
                    L_i = self.lb  # Lower bounds
                    RAND = np.random.rand(self.D)
                    
                    new_Xi = (Xb - X_M) * alpha - r1 + ((U_i - L_i) * RAND + L_i) * delta
                    
                else:
                    # Phase 4: Narrowed Exploitation (Attack and catch with walk and grab)
                    # Equation (13): X_i(t+1) = QF(t) * X_best(t) - (H_1 * X_P(t) * rand) - H_2 * Levy(D) + rand * H_1
                    
                    # Calculate movement coefficients as per equations (15) and (16)
                    H1 = 2 * r1 - 1  # H_1 = 2 × rand - 1, equation (15)
                    H2 = 2 * (1 - t / self.T)  # H_2 = 2 × (1 - t/T_max), equation (16)
                    
                    # Get chaotic quality function QF(t)
                    QF_t = self._chaotic_map_qf(t)
                    
                    # Select random position X_P(t)
                    X_P = self.X[np.random.randint(0, self.N)]
                    
                    # Generate Lévy flight for fine-tuning
                    levy = self._levy_flight(self.D, beta=1.5)
                    
                    # Apply equation (13)
                    new_Xi = QF_t * Xb - (H1 * X_P * r2) - H2 * levy + r3 * H1

                # Apply self-improvement mechanism
                new_Xi = self._self_improvement_mechanism(new_Xi, Xb, t)

                # Boundary handling - ensure solution stays within bounds
                new_Xi = np.clip(new_Xi, self.lb, self.ub)

                # Evaluate new solution
                new_fitness = self._compute_fitness(new_Xi)

                # Selection: Accept if better (greedy selection)
                if new_fitness < self.fitness[i]:
                    self.X[i, :] = new_Xi
                    self.fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_Xi.copy()
                        self.best_index = i

            # Progress reporting
            if t % 10 == 0 or t == self.T:
                avg_fitness = np.mean(self.fitness)
                print(f"[Iter {t:3d}/{self.T}] Best: {self.best_fitness:.6f}, "
                      f"Avg: {avg_fitness:.6f}, Phase: {self._get_phase_name(t_ratio)}")

        print(f"SIAO optimization completed. Final best fitness: {self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness
    
    def _get_phase_name(self, t_ratio):
        """Get current optimization phase name."""
        if t_ratio <= 0.2:
            return "Expanded Exploration"
        elif t_ratio <= 0.5:
            return "Narrowed Exploration"
        elif t_ratio <= 0.8:
            return "Expanded Exploitation"
        else:
            return "Narrowed Exploitation"


class AdvancedSIAO:
    """
    Enhanced SIAO with adaptive parameters and advanced optimization features.
    """
    def __init__(self, model, val_loader, device,
                 population_size=20,
                 max_iters=50,
                 dim=None,
                 lower_bound=-0.1,
                 upper_bound=0.1,
                 adaptive_params=True,
                 elite_preservation=True,
                 diversity_control=True):
        """
        Enhanced SIAO with additional features.
        """
        self.model = model
        self.device = device
        self.val_loader = val_loader
        self.pop_size = population_size
        self.max_iters = max_iters
        self.adaptive_params = adaptive_params
        self.elite_preservation = elite_preservation
        self.diversity_control = diversity_control

        # Extract current weights
        self.init_weight_vector = self._get_flat_weights()
        self.dim = dim if dim is not None else self.init_weight_vector.shape[0]

        # Set search bounds
        self.lb = self.init_weight_vector + lower_bound
        self.ub = self.init_weight_vector + upper_bound

        # Initialize population
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.full((self.pop_size,), np.inf)
        self.best_index = None
        self.best_solution = None
        self.best_fitness = np.inf
        
        # Elite preservation
        self.elite_size = max(1, self.pop_size // 10) if self.elite_preservation else 0
        self.elite_solutions = []
        self.elite_fitness = []
        
        # Diversity tracking
        self.diversity_history = []
        self.convergence_history = []
        
        # Adaptive parameters
        self.adaptive_factor = 1.0
        self.stagnation_count = 0
        self.last_best_fitness = np.inf

        # SIAO parameters
        self.N = self.pop_size
        self.D = self.dim
        self.T = self.max_iters

    def _get_flat_weights(self):
        """Extract LSTM + FC parameters into a single 1D numpy array."""
        params = []
        for name, p in self.model.named_parameters():
            if "lstm" in name or "classifier" in name or "fc" in name:
                params.append(p.detach().cpu().numpy().ravel())
        flat = np.concatenate(params)
        return flat

    def _set_flat_weights(self, flat_vector):
        """Write flat_vector back into the model's parameters."""
        index = 0
        for name, p in self.model.named_parameters():
            if "lstm" in name or "classifier" in name or "fc" in name:
                shape = p.data.shape
                numel = p.data.numel()
                new_vals = flat_vector[index : index + numel]
                new_tensor = torch.from_numpy(new_vals.reshape(shape)).to(self.device)
                p.data.copy_(new_tensor)
                index += numel
        assert index == flat_vector.shape[0]

    def _compute_fitness(self, flat_vector):
        """Compute fitness with multiple objectives."""
        self._set_flat_weights(flat_vector)
        self.model.eval()
        
        running_sq_error = 0.0
        total_count = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i, (X_win, X_stat, y_true) in enumerate(self.val_loader):
                X_win = X_win.to(self.device)
                X_stat = X_stat.to(self.device)
                y_true = y_true.to(self.device)
                
                logits = self.model(X_win, X_stat)
                probs = F.softmax(logits, dim=1)
                
                # Multi-objective fitness: RMSE + classification accuracy
                num_classes = logits.shape[1]  # Get actual number of classes from logits
                y_onehot = F.one_hot(y_true, num_classes=num_classes).float()
                sq_err = (probs - y_onehot).pow(2).sum().item()
                running_sq_error += sq_err
                total_count += probs.shape[0] * probs.shape[1]
                
                # Accuracy component
                pred = torch.argmax(logits, dim=1)
                correct_predictions += (pred == y_true).sum().item()
                total_predictions += y_true.size(0)
                
                if i >= 5:  # Limit evaluation for speed
                    break
        
        rmse = np.sqrt(running_sq_error / total_count)
        accuracy = correct_predictions / total_predictions
        
        # Combined fitness (minimize RMSE, maximize accuracy)
        fitness = rmse - 0.5 * accuracy  # Weight accuracy contribution
        return fitness

    def _enhanced_levy_flight(self, size, beta=1.5, scale_factor=None):
        """Enhanced Lévy flight with adaptive scaling."""
        if scale_factor is None:
            scale_factor = self.adaptive_factor
            
        # Use scipy's Lévy stable distribution for more accurate Lévy flights
        try:
            levy_samples = levy_stable.rvs(alpha=beta, beta=0, size=size, scale=scale_factor)
            # Clip extreme values
            levy_samples = np.clip(levy_samples, -10, 10)
        except:
            # Fallback to original implementation
            import math
            sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                      (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.randn(size) * sigma_u * scale_factor
            v = np.random.randn(size)
            levy_samples = u / (np.abs(v) ** (1 / beta))
            levy_samples = np.clip(levy_samples, -10, 10)
        
        return levy_samples

    def _adaptive_chaotic_qf(self, t):
        """Adaptive chaotic QF with parameter adjustment."""
        base_qf = self._chaotic_qf(t)
        
        # Adapt based on convergence state
        if self.stagnation_count > 5:
            # Increase exploration when stagnating
            return base_qf * (1 + 0.1 * self.stagnation_count)
        else:
            return base_qf

    def _chaotic_qf(self, t):
        """Original chaotic QF implementation."""
        choice = random.choice([17, 18, 19])
        if choice == 17:
            return 1.95 - 2 ** (t ** (1/4)) / (t ** (1/3) + 1e-8)
        elif choice == 18:
            return 1.95 - 2 ** (t ** (1/3)) / (t ** (1/4) + 1e-8)
        else:
            return 0.5 + 2 * np.exp(- (4 * t / self.T) ** 2)

    def _calculate_diversity(self):
        """Calculate population diversity."""
        if len(self.X) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.X)):
            for j in range(i + 1, len(self.X)):
                distance = np.linalg.norm(self.X[i] - self.X[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0

    def _maintain_diversity(self):
        """Maintain population diversity by replacing similar solutions."""
        if not self.diversity_control:
            return
            
        diversity_threshold = 0.01 * np.linalg.norm(self.ub - self.lb)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                distance = np.linalg.norm(self.X[i] - self.X[j])
                if distance < diversity_threshold:
                    # Replace the worse solution with a random one
                    if self.fitness[i] > self.fitness[j]:
                        self.X[i] = np.random.uniform(self.lb, self.ub)
                        self.fitness[i] = np.inf
                    else:
                        self.X[j] = np.random.uniform(self.lb, self.ub)
                        self.fitness[j] = np.inf

    def _update_elite(self):
        """Update elite solutions."""
        if not self.elite_preservation:
            return
            
        # Combine current population with elite
        all_solutions = list(zip(self.X, self.fitness))
        if self.elite_solutions:
            all_solutions.extend(zip(self.elite_solutions, self.elite_fitness))
        
        # Sort by fitness and keep best
        all_solutions.sort(key=lambda x: x[1])
        
        # Update elite
        self.elite_solutions = [sol[0].copy() for sol in all_solutions[:self.elite_size]]
        self.elite_fitness = [sol[1] for sol in all_solutions[:self.elite_size]]

    def _adaptive_parameter_update(self, t):
        """Update adaptive parameters based on search progress."""
        if not self.adaptive_params:
            return
            
        # Check for stagnation
        if abs(self.best_fitness - self.last_best_fitness) < 1e-6:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            
        self.last_best_fitness = self.best_fitness
        
        # Adaptive factor based on progress and diversity
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        # Adjust adaptive factor
        if self.stagnation_count > 3:
            self.adaptive_factor = min(2.0, self.adaptive_factor * 1.1)  # Increase exploration
        else:
            self.adaptive_factor = max(0.5, self.adaptive_factor * 0.95)  # Decrease exploration

    def optimize(self):
        """Enhanced SIAO optimization loop."""
        # Initialize fitness
        for i in range(self.N):
            self.fitness[i] = self._compute_fitness(self.X[i, :])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.X[i, :].copy()
                self.best_index = i

        # Store initial elite
        self._update_elite()
        
        print(f"Initial best fitness: {self.best_fitness:.6f}")

        # Main optimization loop
        for t in range(1, self.T + 1):
            # Update adaptive parameters
            self._adaptive_parameter_update(t)
            
            # Calculate mean position
            X_M = self.X.mean(axis=0)

            for i in range(self.N):
                Xi = self.X[i].copy()
                Xb = self.best_solution
                r = np.random.rand()
                rand = np.random.rand()

                # Enhanced phase-based updates
                if t <= 0.2 * self.T:
                    # Enhanced exploration with elite guidance
                    if self.elite_solutions and np.random.rand() < 0.3:
                        elite_idx = np.random.randint(len(self.elite_solutions))
                        Xb = self.elite_solutions[elite_idx]
                    
                    new_Xi = (Xb * (1 - t/self.T) + (X_M - Xb) * rand) * self.adaptive_factor
                    
                elif t <= 0.4 * self.T:
                    # Enhanced narrowed exploration
                    levy = self._enhanced_levy_flight(self.D, beta=1.5)
                    X_R = self.X[np.random.randint(0, self.N)]
                    new_Xi = Xb * levy + X_R + (r - rand) * rand * self.adaptive_factor
                    
                elif t <= 0.7 * self.T:
                    # Enhanced exploitation with local search
                    alpha = 0.1 * self.adaptive_factor
                    Ub = self.ub
                    Lb = self.lb
                    RAND = np.random.rand(self.D)
                    delta = 0.1 * (1 - t/self.T)  # Decreasing delta
                    new_Xi = (Xb - X_M) * alpha - rand + ((Ub - Lb) * RAND + Lb) * delta
                    
                else:
                    # Enhanced final exploitation with chaotic fine-tuning
                    H1 = 2 * np.random.rand() - 1
                    H2 = 2 * (1 - t/self.T)
                    QF_t = self._adaptive_chaotic_qf(t)
                    X_P = self.X[np.random.randint(0, self.N)]
                    levy = self._enhanced_levy_flight(self.D, beta=1.5, scale_factor=0.1)
                    new_Xi = (QF_t * Xb - (H1 * X_P * rand) - H2 * levy + rand * H1) * self.adaptive_factor

                # Boundary handling
                new_Xi = np.clip(new_Xi, self.lb, self.ub)

                # Evaluate new solution
                new_fit = self._compute_fitness(new_Xi)

                # Enhanced selection with probability acceptance for worse solutions
                accept_probability = 1.0
                if new_fit > self.fitness[i]:
                    # Accept worse solutions with decreasing probability
                    temp = 0.1 * (1 - t/self.T)
                    accept_probability = np.exp(-(new_fit - self.fitness[i]) / temp)

                if new_fit < self.fitness[i] or np.random.rand() < accept_probability:
                    self.X[i, :] = new_Xi
                    self.fitness[i] = new_fit
                    
                    if new_fit < self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_solution = new_Xi.copy()
                        self.best_index = i

            # Maintain population diversity
            self._maintain_diversity()
            
            # Update elite solutions
            self._update_elite()
            
            # Track convergence
            self.convergence_history.append(self.best_fitness)
            
            if t % 10 == 0 or t == self.T:
                diversity = self._calculate_diversity()
                print(f"[Iter {t}/{self.T}] Best RMSE: {self.best_fitness:.6f}, "
                      f"Diversity: {diversity:.6f}, Adaptive Factor: {self.adaptive_factor:.3f}")

        return self.best_solution, self.best_fitness
