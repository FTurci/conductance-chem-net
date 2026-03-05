import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
from sympy import Matrix


# ============================================================
# NUMBA-ACCELERATED SSA CORE
# ============================================================

@njit
def ssa_core(
    SM,
    current_pops,
    rates_list,
    current_pops_index,
    final_time,
    num_internal_species,
    stoich_cols,
    max_steps,
    store_trajectories,
    burn_in
):
    # ---- Internal copy: prevents in-place mutation of caller's array ----
    pops = current_pops.copy()

    t = 0.0
    T = final_time
    steady_time = 0.0

    n_species = len(pops)
    n_reactions = SM.shape[1]

    if store_trajectories:
        time_history = np.zeros(max_steps)
        pop_history = np.zeros((max_steps, n_species))
    else:
        time_history = np.zeros(1)
        pop_history = np.zeros((1, n_species))

    reaction_chosen_tracker = np.zeros(n_reactions)
    force_sums = np.zeros(stoich_cols)

    step_counter = 0
    if store_trajectories:
        time_history[0] = t
        pop_history[0, :] = pops

    while t < T:
        propensity_vector = np.zeros(n_reactions)
        for a in range(n_reactions):
            product_of_counts = 1.0
            for idx in current_pops_index[a]:
                product_of_counts *= pops[idx]
            propensity_vector[a] = product_of_counts * rates_list[a]

        a0 = np.sum(propensity_vector)
        if a0 == 0.0:
            break

        r1 = np.random.rand()
        r2 = np.random.rand()
        tau = -math.log(r1) / a0
        target_value = r2 * a0

        cumulative = 0.0
        reaction_chosen = 0
        for n in range(n_reactions):
            cumulative += propensity_vector[n]
            if target_value <= cumulative:
                reaction_chosen = n
                break

        t_next = t + tau
        if t_next > burn_in:
            if t < burn_in:
                tau_effective = t_next - burn_in
            else:
                tau_effective = tau
            steady_time += tau_effective
            reaction_chosen_tracker[reaction_chosen] += 1

            l = 0
            pair_index = 0
            while l < 2 * stoich_cols:
                if propensity_vector[l] > 0.0 and propensity_vector[l+1] > 0.0:
                    force_sums[pair_index] += tau_effective * math.log(
                        propensity_vector[l] / propensity_vector[l+1]
                    )
                l += 2
                pair_index += 1

        t += tau
        for m in range(num_internal_species):
            pops[m] += SM[m, reaction_chosen]

        if store_trajectories:
            step_counter += 1
            if step_counter >= max_steps:
                break
            time_history[step_counter] = t
            pop_history[step_counter, :] = pops

    if store_trajectories:
        time_history = time_history[:step_counter + 1]
        pop_history = pop_history[:step_counter + 1, :]

    return (
        pops,
        reaction_chosen_tracker,
        force_sums,
        steady_time,
        time_history,
        pop_history
    )


# ============================================================
# CLASS
# ============================================================

class RunSSA:

    def __init__(self, module_for_simulating, initial_counts, rates,
                 simulation_length, burn_in):
        """
        Parameters
        ----------
        module_for_simulating : object
            Must have:
              - .stoich_matrix (sympy Matrix)
              - .species_names (list of str)
              - .num_internal_species (int)
              - .external_stoich_matrix (sympy Matrix)
              - .calculate_reaction_cycle_matrix() -> sympy Matrix
              - .calculate_selection_matrix() -> sympy Matrix
        initial_counts : list[float]
            Starting molecule counts for all species.
        rates : list[float] or np.ndarray
            Forward and backward rates for each reaction.
        simulation_length : float
            Total simulation time.
        burn_in : float
            Time before which data is discarded (transient period).
        """
        self.module = module_for_simulating
        self.stoich_matrix = module_for_simulating.stoich_matrix
        self.species_names = module_for_simulating.species_names
        self.current_pops = list(initial_counts)
        self.initial_counts = list(initial_counts)
        self.rates_list = rates
        self.final_time = simulation_length
        self.num_internal_species = module_for_simulating.num_internal_species
        self.burn_in = burn_in
        self.n_reactions = self.stoich_matrix.cols

        self.SM_with_reverse_stoichiometry = self.create_SM_with_reverse_stoichiometry()
        self.current_pops_index = self.determine_consumed_species_in_each_reaction()
        

    # ----------------------------------------------------------
    def create_SM_with_reverse_stoichiometry(self):
        SM = []
        for p in range(self.stoich_matrix.cols):
            SM.append(self.stoich_matrix[:, p])
            SM.append(-self.stoich_matrix[:, p])
        self.SM_with_reverse_stoichiometry = Matrix.hstack(*SM)
        return self.SM_with_reverse_stoichiometry

    # ----------------------------------------------------------
    def determine_consumed_species_in_each_reaction(self):
        self.current_pops_index = []
        for l in range(self.SM_with_reverse_stoichiometry.cols):
            reaction = self.SM_with_reverse_stoichiometry[:, l]
            current_reaction_indexes = []
            for p in range(len(reaction)):
                if reaction[p] < 0:
                    current_reaction_indexes.append(p)
            self.current_pops_index.append(current_reaction_indexes)
        return self.current_pops_index

    # ----------------------------------------------------------
    def run_SSA_and_plot_counts(self, store_trajectories=True, starting_pops=None):
        """
        Run a single SSA simulation.

        Parameters
        ----------
        store_trajectories : bool, optional (default=True)
            If True, stores full time/population histories and plots them.
        starting_pops : list or None, optional (default=None)
            If None, uses self.current_pops.
            Pass explicitly for guaranteed independent runs.
        """
        SM_np = np.array(self.SM_with_reverse_stoichiometry).astype(np.float64)
        rates_np = np.array(self.rates_list, dtype=np.float64)
        current_pops_index_np = [
            np.array(lst, dtype=np.int64) for lst in self.current_pops_index
        ]

        if starting_pops is None:
            pops_for_sim = np.array(self.current_pops, dtype=np.float64)
        else:
            pops_for_sim = np.array(starting_pops, dtype=np.float64)

        max_steps = 10_000_000

        loop_time_start = time.time()

        (
            final_pops,
            reaction_chosen_tracker,
            force_sums,
            steady_time,
            time_history,
            pop_history
        ) = ssa_core(
            SM_np,
            pops_for_sim,
            rates_np,
            current_pops_index_np,
            self.final_time,
            self.num_internal_species,
            self.stoich_matrix.cols,
            max_steps,
            store_trajectories,
            self.burn_in
        )

        loop_time_end = time.time()

        self.final_pops = final_pops.tolist()
        self.steady_time = steady_time

        if store_trajectories:
            self.time_history = time_history
            self.pop_history = pop_history

        self.average_reaction_currents = []
        g = 0
        while g < len(reaction_chosen_tracker):
            current = (
                reaction_chosen_tracker[g] - reaction_chosen_tracker[g + 1]
            ) / steady_time
            self.average_reaction_currents.append(current)
            g += 2

        self.averaged_forces = (force_sums / steady_time).tolist()

        
        

        self.average_resistances = []
        for i in range(self.stoich_matrix.cols):
            if (self.averaged_forces[i] != 0 and
                    self.average_reaction_currents[i] != 0):
                self.average_resistances.append(
                    self.averaged_forces[i] / self.average_reaction_currents[i]
                )
            else:
                self.average_resistances.append(np.nan)

        if store_trajectories:
            plt.figure(figsize=(8, 5))
            for m in range(self.num_internal_species):
                plt.step(
                    self.time_history,
                    self.pop_history[:, m],
                    where="post",
                    label=self.species_names[m]
                )
            plt.xlabel("Time")
            plt.ylabel("Molecule count")
            plt.title("Gillespie SSA Simulation")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # ==========================================================
    # PLOT CURRENT AND MEANS COMPARED TO GAUSSIANS
    # ==========================================================

    def plot_gaussian_comparison(
            self,
            bins=50,
            num_iterations=10,
            Gaussian_points=1000
            ):
        
        # forces and currents have shape: [[a1,a2,a3], [b1,b2,b3], ... [num_iterations times]], where each inner list is in the order [reaction 1, reaction 2, reaction 3] 
        # (for a 3-reaction system). So we can plot the distribution of currents and forces for each reaction across all iterations.

        # To obtain these lists, use the IF sweep function for a single count value which is the exact same as the users initial counts. This runs num_iterations many independent
        # SSA simulations and returns the currents and forces for each in form as explained above.
        
        currents, forces = self.run_IF_sweep(
            0,
            [self.initial_counts[0]],
            num_iterations,
            covariance_reaction_indices=None,
            verbose=False
    )
        
        
        for j in range(self.stoich_matrix.cols):
            
            plt.hist([cur[j] for cur in currents],bins,edgecolor='black', density=True)

            mean_I = np.mean([cur[j] for cur in currents])
            std_dev_I = np.std([cur[j] for cur in currents])
            x_F = np.linspace(min([cur[j] for cur in currents]), max([cur[j] for cur in currents]), Gaussian_points)
            y_F = (1 / (np.sqrt(2 * np.pi) * std_dev_I)) * np.exp(-((x_F - mean_I)**2) / (2 * std_dev_I**2))
            plt.plot(x_F, y_F, color='red')
            plt.title(f'Distribution of Average Current for Reaction {j + 1} across {num_iterations} SSA runs')
            plt.xlabel('Microscopic Current')
            plt.ylabel('Density')
            plt.grid(True)

            plt.show()

            plt.hist([force[j] for force in forces],bins,edgecolor='black', density=True)

            mean_F = np.mean([force[j] for force in forces])
            std_dev_F = np.std([force[j] for force in forces])
            x_F = np.linspace(min([force[j] for force in forces]), max([force[j] for force in forces]), Gaussian_points)
            y_F = (1 / (np.sqrt(2 * np.pi) * std_dev_F)) * np.exp(-((x_F - mean_F)**2) / (2 * std_dev_F**2))
            plt.plot(x_F, y_F, color='red')
            plt.title(f'Distribution of Average Force for Reaction {j+1} across {num_iterations} SSA runs')
            plt.xlabel('Microscopic Force')
            plt.ylabel('Density')
            plt.grid(True)
            plt.show()

    # ==========================================================
    # I-F SWEEP
    # ==========================================================

    def run_IF_sweep(
    self,
    species_index,
    count_values,
    total_iterations,
    covariance_reaction_indices=None,
    verbose=True
):
        """
        Sweep one or more species' initial counts, running total_iterations
        independent SSA simulations at each value.

        Parameters
        ----------
        species_index : int or list of int
            Index or indices into initial_counts of the species to vary.
        count_values : array-like or list of array-like
            If species_index is int:  1D array of counts to sweep over.
            If species_index is list: list of 1D arrays, one per species.
            All arrays must have the same length.
        total_iterations : int
            Independent SSA runs per count value.
        covariance_reaction_indices : list of int, optional
            Reaction indices for the rescaled covariance matrix.
            Default: all reactions.
        verbose : bool, optional (default=True)
            Print progress.
        """

        

        # ── Normalise inputs to lists ─────────────────────────────────────────────


        if isinstance(species_index, int):
            species_index = [species_index]
            count_values  = [count_values]

        self.species_indexes = species_index
        self.count_values = count_values

        if len(species_index) != len(count_values):
            raise ValueError(
                f"species_index has {len(species_index)} entries but "
                f"count_values has {len(count_values)} — must be equal."
            )

        count_values = [np.asarray(cv, dtype=np.float64) for cv in count_values]

        # Check all count_values arrays have the same length
        n_sweeps = len(count_values[0])
        if not all(len(cv) == n_sweeps for cv in count_values):
            raise ValueError(
                "All count_values arrays must have the same length."
            )

        n_rxn = self.n_reactions

        if covariance_reaction_indices is None:
            covariance_reaction_indices = list(range(n_rxn))
        cov_idx = np.array(covariance_reaction_indices, dtype=np.int64)
        n_cov   = len(cov_idx)

        I_means      = np.zeros((n_sweeps, n_rxn))
        F_means      = np.zeros((n_sweeps, n_rxn))
        I_vars       = np.zeros((n_sweeps, n_rxn))
        F_vars       = np.zeros((n_sweeps, n_rxn))
        cov_matrices = np.zeros((n_sweeps, n_cov, n_cov))

        t_start = time.time()

        for s in range(n_sweeps):

            currents_block = np.zeros((total_iterations, n_rxn))
            forces_block   = np.zeros((total_iterations, n_rxn))

            for it in range(total_iterations):

                # Start from initial counts and overwrite each varied species
                fresh_pops = list(self.initial_counts)
                for idx, cv in zip(species_index, count_values):
                    fresh_pops[idx] = float(cv[s])

                self.run_SSA_and_plot_counts(
                    store_trajectories=False,
                    starting_pops=fresh_pops
                )

                currents_block[it, :] = self.average_reaction_currents
                forces_block[it, :]   = self.averaged_forces

            I_means[s, :]  = np.mean(currents_block, axis=0)
            F_means[s, :]  = np.mean(forces_block,   axis=0)
            I_vars[s, :]   = np.var(currents_block,  axis=0, ddof=1)
            F_vars[s, :]   = np.var(forces_block,    axis=0, ddof=1)

            T_eff      = self.final_time - self.burn_in
            cov_subset = currents_block[:, cov_idx]
            Sigma_J    = np.cov(cov_subset, rowvar=False)

            if n_cov == 1:
                cov_matrices[s, 0, 0] = T_eff * float(Sigma_J)
            else:
                cov_matrices[s, :, :] = T_eff * Sigma_J

            if verbose:
                varied_str = ', '.join(
                    f"[{self.species_names[idx]}] = {cv[s]:.0f}"
                    for idx, cv in zip(species_index, count_values)
                )
                print(f"Sweep {s + 1}/{n_sweeps}  ({varied_str})")

        t_end = time.time()
        if verbose:
            print(f"Total sweep time: {t_end - t_start:.2f} s")

        self.sweep_count_values              = count_values[0]   # primary sweep axis for plotting
        self.sweep_species_index             = species_index[0]  # primary species for plotting
        self.sweep_I_means                   = I_means
        self.sweep_F_means                   = F_means
        self.sweep_I_variances               = I_vars
        self.sweep_F_variances               = F_vars
        self.sweep_covariance_matrices       = cov_matrices
        self.sweep_covariance_reaction_indices = covariance_reaction_indices

        return currents_block, forces_block
        

    # ==========================================================
    # PLOT I-F CURVES
    # ==========================================================

    def plot_IF_curves(
        self,
        reaction_indices=None,
        analytical_currents=None,
        analytical_forces=None,
        show_errorbars=True,
        colour_by_count=True,
        marker_size=60,
        cmap='viridis'
    ):
        """
        Plot Current vs Force for each reaction from sweep results.
        """
        if not hasattr(self, 'sweep_I_means'):
            raise RuntimeError("No sweep data found. Call run_IF_sweep() first.")

        if reaction_indices is None:
            reaction_indices = list(range(self.n_reactions))
        if analytical_currents is None:
            analytical_currents = {}
        if analytical_forces is None:
            analytical_forces = {}

        # for l in range(len(self.species_indexes)): # for every varies species


        for r in reaction_indices:
            fig, ax = plt.subplots(figsize=(8, 8))

            F_vals = self.sweep_F_means[:, r]
            I_vals = self.sweep_I_means[:, r]
            c_vals = self.sweep_count_values

            if show_errorbars:
                F_err = np.sqrt(self.sweep_F_variances[:, r])
                I_err = np.sqrt(self.sweep_I_variances[:, r])
                ax.errorbar(
                    F_vals, I_vals,
                    xerr=F_err, yerr=I_err,
                    fmt='none', ecolor='grey', alpha=0.3,
                    elinewidth=0.8, zorder=1
                )

            if colour_by_count:
                sc = ax.scatter(
                    F_vals, I_vals,
                    c=c_vals, cmap=cmap, s=marker_size,
                    edgecolors='black', linewidths=0.5,
                    vmin=np.min(c_vals), vmax=np.max(c_vals),
                    zorder=3
                )
                cbar = plt.colorbar(sc, ax=ax)
                species_label = self.species_names[self.species_indexes[0]]
                cbar.set_label(f'Initial {species_label} count')
            else:
                ax.scatter(F_vals, I_vals, s=marker_size, label='SSA', zorder=3)

            if r in analytical_currents and r in analytical_forces:
                ax.scatter(
                    analytical_forces[r], analytical_currents[r],
                    label='Analytical', marker='x', c='red',
                    s=marker_size, zorder=4
                )
                ax.legend()

            ax.set_xlabel("Average Force")
            ax.set_ylabel("Average Current")
            ax.set_title(f"I–F Curve: Reaction {r + 1}")
            ax.grid(True)
            fig.tight_layout()
            plt.show()

    # ==========================================================
    # CONDUCTANCE COMPUTATION
    # ==========================================================

    def compute_conductances(self, analytical_currents=None, analytical_forces=None):
        """
        Compute the fundamental conductance at each sweep point.
        """
        if not hasattr(self, 'sweep_I_means'):
            raise RuntimeError("No sweep data found. Call run_IF_sweep() first.")

        # from sympy import Float as SympyFloat

        n_sweeps = len(self.sweep_count_values)
        n_rxn = self.n_reactions

        C = self.module.calculate_reaction_cycle_matrix()
        S_ext = self.module.external_stoich_matrix
        L = self.module.calculate_selection_matrix()

        # Create objects: SM_externals * cycle_matrix, pseudoinverse of selection matrix
        S_ext_C = S_ext * C
        L_pinv = L.pinv()

        n_indep = L_pinv.rows

        G_fundamental_list = []
        G_eigenvalue_list = []
        G_scalar_list = []

        for s in range(n_sweeps):

            resistances = []
            skip = False

            # Create the resistances along each reaction

            for r in range(n_rxn):
                F_r = self.sweep_F_means[s, r]
                I_r = self.sweep_I_means[s, r]
                if I_r != 0.0 and F_r != 0.0 and not np.isnan(F_r) and not np.isnan(I_r):
                    resistances.append(float(F_r / I_r))
                else:
                    skip = True
                    break

            # If we have any zero or NaN resistances, we cannot compute the conductance for this sweep point, so we skip it and store NaNs.

            if skip:
                G_fundamental_list.append(None)
                G_scalar_list.append(float('nan'))
                G_eigenvalue_list.append([float('nan')] * n_indep)
                continue
            
            # Create the diagonal resistance matrix for this sweep

            R_diag = Matrix.zeros(n_rxn, n_rxn)
            for r in range(n_rxn):
                R_diag[r, r] = resistances[r]

            # Create cycle conductance matrix for this sweep, protect against non-invertibility

            try:
                G_cycle = (C.T * R_diag * C).inv()
            except Exception as e:
                print(f"Warning: Could not invert at sweep {s}: {e}")
                G_fundamental_list.append(None)
                G_scalar_list.append(float('nan'))
                G_eigenvalue_list.append([float('nan')] * n_indep)
                continue
            
            # Create physical and fundamental conductance matrices for this sweep

            G_phys = S_ext_C * G_cycle * S_ext_C.T
            G_fund = L_pinv * G_phys * L_pinv.T

            G_fundamental_list.append(G_fund) # store the full fundamental CM for this sweep

            # Check shape of fundamental CM

            if G_fund.shape == (1, 1):

                # If scalar, store the single value

                G_scalar_list.append(float(G_fund[0, 0]))
                G_eigenvalue_list.append([float(G_fund[0, 0])])
            else:

                # If not scalar, store eigenvalues

                G_fund_np = np.array(G_fund.tolist(), dtype=np.float64)
                eigvals = np.sort(np.linalg.eigvalsh(G_fund_np))
                G_eigenvalue_list.append(eigvals.tolist())
                G_scalar_list.append(float('nan'))

            # Create the fundamental forces and currents, then entropy production lists to plot against

            self.fundamental_forces = []
            self.fundamental_currents = []
            self.fundamental_EPRs = []

            F_map = -self.module.selection_matrix.T * \
                                    self.module.coupling_matrix.T.pinv() * \
                                    self.module.cycle_matrix.T
            
            I_map = -self.module.selection_matrix.pinv() * self.module.external_stoich_matrix
            
            for microscopic_force_vector, microscopic_current_vector in zip(self.sweep_F_means, self.sweep_I_means): 

                reshape_f = Matrix(microscopic_force_vector.tolist()).reshape(len(microscopic_force_vector), 1)
                reshape_i = Matrix(microscopic_current_vector.tolist()).reshape(len(microscopic_current_vector), 1)

                fund_force = F_map * reshape_f
                fund_current = I_map * reshape_i

                self.fundamental_forces.append(fund_force)
                self.fundamental_currents.append(fund_current)

                # should always be a scaler so store as float

                self.fundamental_EPRs.append(float((fund_force.T * fund_current)[0, 0])) 

        self.fundamental_EPRs = np.array(self.fundamental_EPRs, dtype = float)
        self.fundamental_forces = np.array(self.fundamental_forces, dtype=float)

        

                
                                

        valid_G = [G for G in G_fundamental_list if G is not None]
        if len(valid_G) > 0:
            is_scalar = all(G.shape == (1, 1) for G in valid_G)
        else:
            is_scalar = True

        self.conductance_type = 'scalar' if is_scalar else 'matrix'
        self.sweep_G_fundamental = G_fundamental_list
        self.sweep_G_scalar = np.array(G_scalar_list)
        self.sweep_G_eigenvalues = np.array(G_eigenvalue_list)

        # Compute analytical data if it is passed into the function.

        if analytical_currents is not None and analytical_forces is not None:
            analytical_G = []
            for s in range(n_sweeps):
                resistances_analytical = []
                skip_a = False

                for r in range(n_rxn):
                    if r in analytical_currents and r in analytical_forces:
                        I_a = analytical_currents[r][s]
                        F_a = analytical_forces[r][s]
                        if I_a != 0 and F_a != 0:
                            resistances_analytical.append(float(F_a / I_a))
                        else:
                            skip_a = True
                            break
                    else:
                        F_r = self.sweep_F_means[s, r]
                        I_r = self.sweep_I_means[s, r]
                        if I_r != 0.0 and F_r != 0.0:
                            resistances_analytical.append(float(F_r / I_r))
                        else:
                            skip_a = True
                            break

                if skip_a:
                    analytical_G.append(float('nan'))
                    continue

                R_diag_a = Matrix.zeros(n_rxn, n_rxn)
                for r in range(n_rxn):
                    R_diag_a[r, r] = resistances_analytical[r]

                try:
                    G_cycle_a = (C.T * R_diag_a * C).inv()
                    G_phys_a = S_ext_C * G_cycle_a * S_ext_C.T
                    G_fund_a = L_pinv * G_phys_a * L_pinv.T
                    if G_fund_a.shape == (1, 1):
                        analytical_G.append(float(G_fund_a[0, 0]))
                    else:
                        G_a_np = np.array(G_fund_a.tolist(), dtype=np.float64)
                        analytical_G.append(
                            np.sort(np.linalg.eigvalsh(G_a_np)).tolist()
                        )
                except Exception:
                    analytical_G.append(float('nan'))

            self.analytical_G = analytical_G

        # return analytical_G iff it is not empty.
        if analytical_currents is not None and analytical_forces is not None and len(analytical_G) > 0:
            return G_fundamental_list, analytical_G
        else:
            return G_fundamental_list, None
    
    # ==========================================================
    # PLOT CONDUCTANCE
    # ==========================================================

    def plot_conductance(
        self,
        analytical_G=None,
        marker_size=60,
        cmap='viridis',
        fit_order=2,
        show_covariance=True,
        show_difference=True
    ):
        """
        Plot fundamental conductance vs swept species count.

        Scalar case:
            - G vs count
            - Cov(j)/2 vs count (with polynomial fit)
            - |G - Cov(j)/2| vs count (with mean line)

        Matrix case:
            - Eigenvalues of G vs count
            - ||Cov(J)/2|| (spectral norm) vs count
            - min eigenvalue of (Cov(J)/2 - G) vs count
        """
        if not hasattr(self, 'sweep_G_scalar'):
            raise RuntimeError(
                "No conductance data found. Call compute_conductances() first."
            )
        if not hasattr(self, 'sweep_covariance_matrices'):
            raise RuntimeError(
                "No covariance data found. Call run_IF_sweep() first."
            )

        if analytical_G is None and hasattr(self, 'analytical_G'):
            analytical_G = self.analytical_G

        counts = self.sweep_count_values
        n_sweeps = len(counts)

        # ?
        # species_label = (
        #     self.species_names[self.sweep_species_index]
        #     if self.sweep_species_index < len(self.species_names)
        #     else f'Species [{self.sweep_species_index}]'
        # )
        # #

        # =======================
        # SCALAR CONDUCTANCE
        # =======================

        if self.conductance_type == 'scalar':

            fig, ax = plt.subplots(figsize=(10, 7))

            # G coloured by EPR 
            sc = ax.scatter(
                counts, self.sweep_G_scalar,
                c=self.fundamental_EPRs, cmap=cmap,
                s=marker_size, edgecolors='black', linewidths=0.5,
                label='$G$ (SSA)', zorder=3
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(r'$\dot{\sigma} = A^T I$')

            if analytical_G is not None:
                ax.scatter(
                    counts, analytical_G,
                    marker='x', c='red', s=marker_size,
                    label='$G$ (Analytical)', zorder=4
                )

            if show_covariance:
                cov_half = np.array([
                    self.sweep_covariance_matrices[s][0, 0] / 2.0
                    for s in range(len(counts))
                ])

                ax.scatter(
                    counts, cov_half,
                    marker='x', color='blue', s=marker_size,
                    label=r'$\mathrm{Cov}(I) / 2$  (SSA)',
                    zorder=3
                )

                # fitted plot for covariance
                valid = ~np.isnan(cov_half) & ~np.isnan(counts)
                if np.sum(valid) > fit_order + 1:
                    coeffs = np.polyfit(counts[valid], cov_half[valid], fit_order)
                    x_fit = np.linspace(
                        np.min(counts[valid]), np.max(counts[valid]), 200
                    )
                    y_fit = np.polyval(coeffs, x_fit)
                    ax.plot(
                        x_fit, y_fit,
                        linestyle='--', color='blue', alpha=0.6
                    )
                

            if show_difference and show_covariance:

                print("No difference plotted: Option Deprecated")
                # difference = np.abs(self.sweep_G_scalar - cov_half)
                # mean_diff = np.nanmean(difference)

                # ax.scatter(
                #     self.fundamental_forces, difference,
                #     marker='s', color='purple', s=marker_size * 0.6,
                #     label=(
                #         r'$|G - \mathrm{Cov}(I)/2|$'
                #         f',  $\\mu = {mean_diff:.4g}$'
                #     ),
                #     zorder=2
                # )
                # ax.axhline(
                #     mean_diff, color='purple', linewidth=1, alpha=0.6
                # )

            ax.set_xlabel(f'Initial count of {self.species_names[self.species_indexes[0]]}')
            ax.set_ylabel('$G$')
            ax.set_title(r'$G$ vs varied species, colour graded against fundamental EPR')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            plt.show()

   

        else:
            eigvals = self.sweep_G_eigenvalues
            n_indep = eigvals.shape[1]

            fig, ax = plt.subplots(figsize=(10, 7))

            # Create a ScalarMappable for the colorbar
            norm = mpl.colors.Normalize(
                vmin=np.min(self.fundamental_EPRs),
                vmax=np.max(self.fundamental_EPRs)
            )
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Required for colorbar

            # Scatter SSA eigenvalues with color mapped to fundamental_EPRs
            for e in range(n_indep):
                scatter = ax.scatter(
                    counts, eigvals[:, e],
                    c=self.fundamental_EPRs,
                    cmap=cmap,
                    norm=norm,  # Use the same norm as the colorbar
                    s=marker_size,
                    edgecolors='black', linewidths=0.5,
                    label=f'$\\lambda_{{{e+1}}}(G)$  (SSA)',
                    zorder=3
                )

            # Scatter analytical G if provided
            if analytical_G is not None:
                analytical_G_np = np.array(analytical_G)
                if analytical_G_np.ndim == 1:
                    ax.scatter(
                        counts, analytical_G_np,
                        marker='x', c='red', s=marker_size,
                        label='$G$ (Analytical)', zorder=4
                    )
                else:
                    for e in range(analytical_G_np.shape[1]):
                        ax.scatter(
                            counts, analytical_G_np[:, e],
                            marker='x', color='red', s=marker_size,
                            label=f'$\\lambda_{{{e+1}}}(G)$  (Analytical)',
                            zorder=4
                        )

            # Scatter covariance norms if needed
            if show_covariance:
                cov_spectral_norms = np.zeros(n_sweeps)
                for s in range(n_sweeps):
                    cov_half = 0.5 * self.sweep_covariance_matrices[s]
                    cov_spectral_norms[s] = np.linalg.norm(cov_half, 2)

                ax.scatter(
                    counts, cov_spectral_norms,
                    marker='x', color='blue', s=marker_size * 0.8,
                    label=r'$\| \mathrm{Cov}(\mathbf{I})/2 \|$',
                    zorder=3
                )

            # Create the colorbar
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(r'$\dot{\sigma} = A^T I$')

            if show_difference:

                print("No differenec plotted: Option Deprecated.")

                # min_eig_diff = np.zeros(n_sweeps)
                # for s in range(n_sweeps):
                #     cov_half = 0.5 * self.sweep_covariance_matrices[s]

                #     G_fund = self.sweep_G_fundamental[s]
                #     if G_fund is None:
                #         min_eig_diff[s] = float('nan')
                #         continue

                #     G_fund_np = np.array(
                #         G_fund.tolist(), dtype=np.float64
                #     )
                #     diff_matrix = cov_half - G_fund_np
                #     min_eig_diff[s] = np.min(
                #         np.linalg.eigvalsh(diff_matrix)
                #     )

                # ax.scatter(
                #     self.fundamental_forces, min_eig_diff,
                #     marker='s', color='purple', s=marker_size * 0.6,
                #     label=r'$\lambda_{\min}(\mathrm{Cov}(\mathbf{I})/2 - G)$',
                #     zorder=2
                # )
                # ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

            ax.set_xlabel(f'Initial count of {self.species_names[self.species_indexes[0]]}')
            ax.set_ylabel('Conductance Eigenvalue')
            ax.set_title('Conductance Matrix Eigenvalues vs Initial Count')

            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            plt.show()
        # =======================

        # This currently plots against the force components seperately
        # else:

        #     eigvals  = self.sweep_G_eigenvalues   # (n_sweeps, n_eigs)
        #     forces   = self.fundamental_forces    # (n_sweeps, n_forces)
        #     n_indep  = eigvals.shape[1]
        #     n_forces = forces.shape[1]

        #     markers  = ['o', 's', '^', 'D', 'v', 'p']

        #     # ── One subplot per force component, all eigenvalues on same axes ────
        #     fig, axes = plt.subplots(
        #         1, n_forces,
        #         figsize=(5 * n_forces + 2, 5),
        #         sharey=True,
        #         constrained_layout=True
        #     )

        #     # Ensure axes always iterable
        #     if n_forces == 1:
        #         axes = [axes]

        #     sc = None  # for shared colourbar

        #     # Compute covariance norms once outside loops
        #     if show_covariance:
        #         cov_spectral_norms = np.zeros(n_sweeps)
        #         for s in range(n_sweeps):
        #             cov_half = 0.5 * self.sweep_covariance_matrices[s]
        #             cov_spectral_norms[s] = np.linalg.norm(cov_half, 2)

        #     for f in range(n_forces):

        #         ax = axes[f]

        #         # ── Plot all eigenvalues on same axes ─────────────────────────
        #         for e in range(n_indep):

        #             sc = ax.scatter(
        #                 forces[:, f],
        #                 eigvals[:, e],
        #                 c=self.fundamental_EPRs,
        #                 cmap=cmap,
        #                 vmin=np.min(self.fundamental_EPRs),   # ← consistent colour
        #                 vmax=np.max(self.fundamental_EPRs),   # ← scale across subplots
        #                 s=marker_size,
        #                 marker=markers[e % len(markers)],
        #                 edgecolors='black',
        #                 linewidths=0.5,
        #                 zorder=3,
        #                 label=f'$\\lambda_{{{e+1}}}(G)$'
        #             )

        #         if show_covariance:
        #             ax.scatter(
        #                 forces[:, f],
        #                 cov_spectral_norms,
        #                 color = 'blue',
        #                 marker='x',
        #                 s=marker_size * 1.2,
        #                 edgecolors='blue',
        #                 linewidths=0.8,
        #                 zorder=2,
        #                 label=r'$\|\mathrm{Cov}(\mathbf{I})/2\|_2$'
        #             )

        #         ax.set_xlabel(f'$A_{f+1}$',           fontsize=13)
        #         ax.set_ylabel('Eigenvalue',            fontsize=13)
        #         ax.set_title(f'All $\\lambda_k$ vs $A_{f+1}$', fontsize=12)
        #         ax.legend(fontsize=9, loc='upper left')
        #         ax.grid(True, alpha=0.3)

        #     # ── Single shared colourbar ───────────────────────────────────────
        #     if sc is not None:
        #         cbar = fig.colorbar(
        #             sc,
        #             ax=axes,
        #             label=r'$\dot{\sigma} = A^T I$',
        #             shrink=0.8,
        #             pad=0.02,
        #             fraction=0.03
        #         )
        #         cbar.ax.tick_params(labelsize=10)

        #     fig.suptitle(
        #         'Conductance Eigenvalues vs Fundamental Force Components',
        #         fontsize=14
        #     )
        #     plt.show()


def plot_combined_conductance(
    combined_CMs,
    count_values,
    species_label='Species',
    marker_size=60,
    cmap='viridis'
):
    """
    Plot combined fundamental conductance vs swept species count.

    Scalar case (1x1 matrices):
        - G vs count plotted directly.

    Matrix case:
        - Eigenvalues of G vs count.

    Parameters
    ----------
    combined_CMs : list of sympy Matrix
        List of combined fundamental conductance matrices, one per sweep point.
        As returned by CombiningModules.numerical_combined_fundamental_CMs.
    count_values : array-like
        The swept species count values used to generate the data.
    species_label : str, optional
        Label for the swept species on the x-axis.
    marker_size : int, optional
        Scatter plot marker size.
    cmap : str, optional
        Matplotlib colourmap name.
    """

    count_values = np.asarray(count_values, dtype=np.float64)

    # Determine scalar or matrix case from first valid entry (all will be the same shape)

    first_valid = next((CM for CM in combined_CMs if CM is not None), None)

    if first_valid is None:
        raise RuntimeError("No valid conductance matrices to plot.")

    is_scalar = (first_valid.shape == (1, 1))

    # =========================================================================
    # SCALAR CASE
    # =========================================================================

    if is_scalar:

        valid_counts = []
        valid_G      = []

        for i, CM in enumerate(combined_CMs):

            if CM is None:
                continue

            try:
                valid_counts.append(count_values[i])
                valid_G.append(float(CM[0, 0]))
            except Exception as e:
                print(f"Sweep point {i}: could not extract scalar value — {e}")

        valid_counts = np.array(valid_counts)
        valid_G      = np.array(valid_G)

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(
            valid_counts,
            valid_G,
            s=marker_size,
            # c=self.fundamental_EPRs, cmap=cmap,
            color = 'green',
            edgecolors='black',
            linewidths=0.5,
            label='$G^{(3)}$',
            zorder=3
        )

        # cbar = fig.colorbar(sc, ax=ax)
        # cbar.set_label(r'$\dot{\sigma} = A^T I$')

        ax.set_xlabel(f'Initial {species_label} count', fontsize=13)
        ax.set_ylabel('Conductance',                    fontsize=13)
        ax.set_title('Combined Module: Fundamental Conductance vs Count',
                     fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    # =========================================================================
    # MATRIX CASE
    # =========================================================================

    else:

        eigenvalue_list = []

        for i, CM in enumerate(combined_CMs):

            if CM is None:
                eigenvalue_list.append(None)
                continue

            try:
                CM_np   = np.array(CM.tolist(), dtype=np.float64)
                eigvals = np.sort(np.linalg.eigvalsh(CM_np))
                eigenvalue_list.append(eigvals)
            except Exception as e:
                print(f"Sweep point {i}: could not compute eigenvalues — {e}")
                eigenvalue_list.append(None)

        n_eigs = next(
            (len(e) for e in eigenvalue_list if e is not None), 0
        )

        if n_eigs == 0:
            raise RuntimeError("No valid eigenvalues could be computed.")

        valid_counts  = []
        valid_eigvals = [[] for _ in range(n_eigs)]

        for i, eigvals in enumerate(eigenvalue_list):
            if eigvals is not None:
                valid_counts.append(count_values[i])
                for e in range(n_eigs):
                    valid_eigvals[e].append(eigvals[e])

        valid_counts = np.array(valid_counts)

        colours_map = plt.cm.get_cmap(cmap, max(n_eigs, 3))

        fig, ax = plt.subplots(figsize=(10, 7))

        for e in range(n_eigs):
            ax.scatter(
                valid_counts,
                valid_eigvals[e],
                s=marker_size,
                color=colours_map(e),
                edgecolors='black',
                linewidths=0.5,
                label=f'$\\lambda_{{{e+1}}}(G^{{(3)}})$',
                zorder=3
            )

        ax.set_xlabel(f'Initial {species_label} count', fontsize=13)
        ax.set_ylabel('Conductance Eigenvalue',          fontsize=13)
        ax.set_title('Combined Module: Fundamental Conductance Eigenvalues vs Count',
                     fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        fig.tight_layout()
        plt.show()

# all imports

import hypernetx as hnx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import warnings 
#init_printing(use_unicode=True)
warnings.simplefilter('ignore')
import re
from sympy import *
import sympy

init_printing()

class ModuleProperties:

    def __init__(self, stoich_matrix, num_internal_species, species_names):
        self.stoich_matrix = sympy.Matrix(stoich_matrix)
        self.num_internal_species = num_internal_species
        self.num_external_species = self.stoich_matrix.shape[0] - num_internal_species
        self.num_species = self.stoich_matrix.shape[0]
        self.num_reactions = self.stoich_matrix.shape[1]
        self.species_labels = {i: item for i, item in enumerate(species_names)}
        self.species_names = species_names
        self.internal_stoich_matrix = self.stoich_matrix[0:self.num_internal_species, :]
        self.external_stoich_matrix = self.stoich_matrix[self.num_internal_species: len(self.stoich_matrix), :]
        self.selection_matrix = self.calculate_selection_matrix()
        self.cycle_matrix = self.calculate_reaction_cycle_matrix()
        self.coupling_matrix = self.calculate_coupling_matrix()
        


        # LABELLING FOR SPECIES, FORCES, EDGE CURRENTS, CHEMICAL POTENTIALS

        #self.species_labels = []

        self.chemical_potentials = []
     

        for n in range(self.num_species):
            
            #species_symbol = species_names[n]
            species_symbol = symbols(species_names[n])
            #self.species_labels.append(species_symbol)

            chem_pot = symbols(f"\mu_{species_symbol}")
            self.chemical_potentials.append(chem_pot)
            

        self.chemical_potentials_vector = Matrix(self.chemical_potentials).T  # make a vector out of the labelled chemical potentials
        
        # LABELS FOR ALL RESISTANCES AND REACTIONS

        
        resistances = [] # define list to hold reaction labels
        edge_currents_j = [] # to hold the js
        forces = [] # to hold reaction level forces

        for n in range(self.num_reactions): # loop over each reaction
            nth_resistance = symbols(f"r{n+1}") # assign name of nth resistance
            resistances.append(nth_resistance) # add to list of resistance

            nth_edge_currents_j = symbols(f"j{n+1}") # assign name of nth edge current
            edge_currents_j.append(nth_edge_currents_j) # add to list of currents

            reaction_vector = -1* self.stoich_matrix[:,n] # take the column of SM that corresponds to nth reaction
            
            forces.append(self.chemical_potentials_vector*reaction_vector) # use reaction vector *-1  in SM to create forces in terms of chem potentials
        
        
        self.force_vector = Matrix(forces) # create a vector of reaction level forces
        self.edge_currents_vector = Matrix(edge_currents_j) # make a vector out of the js


        # reaction resistance in terms of r = f/j

        reaction_level_res = [] # to hold reaction level resistances

        for n in range(self.num_reactions): # loop over each reaction
            symbolic_resistance = self.force_vector[n] / self.edge_currents_vector[n]

            reaction_level_res.append(symbolic_resistance)

        self.reaction_level_resistances = reaction_level_res # output reaction level resistances in terms of r = f/j

        self.kinetic_form_resistance_matrix = Matrix.diag(reaction_level_res) # output reaction level res. matrix in terms of r = f/j

        self.fundamental_current_vector = self.selection_matrix.pinv() * self.calculate_physical_currents()
    #==========================================================================================================================================
    # REACTION LEVEL CYCLES

    def calculate_reaction_cycle_matrix(self):

        """ This method calculates the reaction level cycles matrix for the internal species of the module using the 
        kernel of the internal stoichiometric matrix.
        
        Returns:
            cycle_matrix (Sympy Matrix): Reaction level cycles matrix for internal species
        """
        
        reaction_cycles = (self.internal_stoich_matrix).nullspace() # finds the kernel for the SM internal

        # Check if there are any cycles:

        if not reaction_cycles:

            print("No internal cycles. Kernel is empty.")

        # build cycle matrix from kernel vectors if kernel is NOT empty

        else:

            cycle_matrix = reaction_cycles[0] # add first vector to cycle matrix so we can add rest later

            for cycle in reaction_cycles[1:]: # starting at second vector in kernel

                cycle_matrix = cycle_matrix.row_join(cycle) # connect vectors from kernel column-wise, row_join puts elemetns of adjacent vectors together


            self.cycle_matrix = cycle_matrix # assign cycle matrix to self for use in other methods
            
            return cycle_matrix # return the cycle matrix
        
    #==========================================================================================================================================
    # COUPLING MATRICES 
 
    def calculate_coupling_matrix(self):

        """ This method calculates the coupling matrix between internal and external species using reaction cycle matrix 
        and SM of external species.

        Returns:
            phi (Sympy Matrix): Coupling matrix between internal and external species
        """

        phi = self.external_stoich_matrix * self.calculate_reaction_cycle_matrix()

        self.phi = phi
        return phi
    
    #==========================================================================================================================================
    # CONSERVATION LAW MATRICES

    def calculate_conservation_laws(self):

        """ This method calculates the conservation law matrices for the full stoichiometric matrix and the chemostat species only.
        
        Returns:
            cons_laws.T (Sympy Matrix): Conservation law matrix for the full stoichiometric matrix  
            chemostat_laws.T (Sympy Matrix): Conservation law matrix for the chemostat species only
        """

        cokernel_SM = (self.stoich_matrix.T).nullspace() # finds the cokernel of the full SM

        if not cokernel_SM:

            print("No conservation laws. Cokernel of Stoichiometric Matrix empty.")

        else:

            cons_laws = cokernel_SM[0] # adds first element of cokernel

            for vec in cokernel_SM[1:]: # add vectors from next row onwards

                cons_laws = cons_laws.row_join(vec)


        #
        # Broken external laws for chemostat , deriving from the coupling matrix
        #

        coupling_matrix = self.calculate_coupling_matrix() # define the coupling matrix using the function defined previously

        cokernel_coupling_matrix = coupling_matrix.T.nullspace() # find the cokernel of the coupling matrix

        if not cokernel_coupling_matrix:

            print("No chemostat conservation laws. Cokernel of Coupling Matrix is empty.")

        # if cokernel is NOT empty

        else:

            chemostat_laws = cokernel_coupling_matrix[0] # add first vector to chemostat conservation law matrix so we can add rest later

            for law in cokernel_coupling_matrix[1:]: # starting at second vector in kernel

                chemostat_laws = chemostat_laws.row_join(law) # connect vectors from kernel column-wise, row_join puts elemetns of adjacent vectors together



        self.cons_laws = cons_laws.T # assign to self for use in other methods
        self.chemostat_cons_laws = chemostat_laws.T # assign to self for use in other methods

        return cons_laws.T, chemostat_laws.T # return transpose to match equations in paper { L^(1) and l^(1) respectively}
    
    #==========================================================================================================================================
    # SELECTION MATRIX

    def calculate_selection_matrix(self):

        """ This method calculates the selection matrix for the chemostat conservation laws.
        
        Returns:
            selection_matrix (Sympy Matrix): Selection matrix for the chemostat conservation laws
        """

        chemostat_laws = self.calculate_conservation_laws()[1] # get chemostat conservation laws from previous method

        null_basis_chemostat_laws = (chemostat_laws).nullspace() # find nullspace of chemostat conservation laws

        if null_basis_chemostat_laws:

            selection_matrix = sympy.Matrix.hstack(*null_basis_chemostat_laws) # build selection matrix from nullspace vectors

        else:

            selection_matrix = sympy.Matrix([]) # empty matrix if no nullspace

        self.selection_matrix = selection_matrix # assign to self for use in other methods

        return selection_matrix

    #==========================================================================================================================================
    # PHYSICAL CURRENTS

    def calculate_physical_currents(self):

        currents_constraints = sympy.solve (self.internal_stoich_matrix * self.edge_currents_vector, self.edge_currents_vector)

        physical_currents = (-1* self.external_stoich_matrix * self.edge_currents_vector).subs(currents_constraints)

        self.physical_currents = physical_currents # assign to self for use in other methods

        return physical_currents

    #==========================================================================================================================================
    # REACTION LEVEL RESISTANCE MATRIX
    
    def calculate_reaction_resistance_matrix(self):

        """ Calculates the reaction level resistance matrix for the module, including an auto-labelling of reactions in the SM 
        according to the number of columns in the SM for use in Sympy operations.

        Returns:
            reaction_resistance_matrix (Sympy Matrix): Reaction level resistance matrix for the module"""
                    
        resistances = [] # define list to hold reaction labels
    
        for n in range(self.num_reactions): # loop over each reaction

            nth_reaction = symbols(f"r{n+1}") # assign name of nth reaction

            resistances.append(nth_reaction) # add to list of reactions


    
        reaction_resistance_matrix = Matrix.diag(resistances) # create diagonal reaction level resistance matrix from list of reactions

        self.reaction_resistance_matrix = reaction_resistance_matrix # assign to self for use in other methods

        return reaction_resistance_matrix

    #==========================================================================================================================================
    # CYCLE RESISTANCE MATRIX

    def calculate_cycle_resistance_matrix(self):

        """ Uses the reaction level resistance matrix and reaction cycles matrix to calculate the cycle resistance matrix for the module.
        
        Returns:
            cycle_resistance_matrix (Sympy Matrix): Cycle resistance matrix for the module"""

        cycle_resistance_matrix = self.calculate_reaction_cycle_matrix().T * self.calculate_reaction_resistance_matrix() \
        * self.calculate_reaction_cycle_matrix()

        self.cycle_resistance_matrix = cycle_resistance_matrix # assign to self for use in other methods

        return cycle_resistance_matrix    
    
    #==========================================================================================================================================
    # PHYSICAL CONDUCATANCE MATRICES

    def calculate_physical_conductance_matrix(self):
        """ This method calculates the physical conductance matrix for the module using the coupling matrix and cycle resistance matrix.

        Returns:
            physical_conductance_matrix (Sympy Matrix): Physical conductance matrix for the module
        """

        physical_conductance_matrix = self.calculate_coupling_matrix() * self.calculate_cycle_resistance_matrix().inv() \
              * self.calculate_coupling_matrix().T        

        self.physical_conductance_matrix = physical_conductance_matrix # assign to self for use in other methods

        return physical_conductance_matrix
    
    #==========================================================================================================================================
    # FUNDAMENTAL CONDUCTANCE MATRIX

    def calculate_fundamental_conductance_matrix(self):
        """ This method calculates the fundamental conductance matrix for the module using the selection matrix and physical conductance matrix.

        Returns:
            fundamental_conductance_matrix (Sympy Matrix): Fundamental conductance matrix for the module
        """

        fundamental_conductance_matrix = self.calculate_selection_matrix().pinv() * self.calculate_physical_conductance_matrix() * self.calculate_selection_matrix().T.pinv()

        self.fundamental_conductance_matrix = fundamental_conductance_matrix # assign to self for use in other methods

        return fundamental_conductance_matrix
    
    #==========================================================================================================================================
    # FUNDAMENTAL RESISTANCE MATRIX

    def calculate_fundamental_resistance_matrix(self):
        """ This method calculates the fundamental resistance matrix for the module using the selection matrix and physical conductance matrix.

        Returns:
            fundamental_resistance_matrix (Sympy Matrix): Fundamental resistance matrix for the module
        """

        fundamental_resistance_matrix = self.calculate_fundamental_conductance_matrix().inv()

        self.fundamental_resistance_matrix = fundamental_resistance_matrix # assign to self for use in other methods

        return fundamental_resistance_matrix



class CombiningModules:

    def __init__(self, left_mod, right_mod, left_mod_numerical_CM=None, right_mod_numerical_CM=None):
                
        #=====================================================================================================================================
        # IDENTIFY MATCHING EXTERNAL SPECIES

        left_ext_indices  = list(range(left_mod.num_internal_species, left_mod.num_species)) # external species indices for left module
        right_ext_indices = list(range(right_mod.num_internal_species, right_mod.num_species)) # external species indices for right module

        left_ext_names  = {i: left_mod.species_labels[i] for i in left_ext_indices} # external species names for left module
        right_ext_names = {j: right_mod.species_labels[j] for j in right_ext_indices} # external species names for right module

        matching_pairs = []
        for i_ext, name_i in left_ext_names.items(): # loop over left external species
            for j_ext, name_j in right_ext_names.items(): # loop over right external species
                if name_i == name_j: # check for matching names
                    matching_pairs.append((i_ext, j_ext)) # store matching index pairs

        if matching_pairs: # if there are matching pairs
            keys_1, keys_2 = zip(*matching_pairs) # unzip into separate lists
            keys_1, keys_2 = list(keys_1), list(keys_2) # convert to lists
        else:
            keys_1, keys_2 = [], [] # no matches found

        # Convert global species indices → external row indices
        left_match_rows  = [i - left_mod.num_internal_species  for i in keys_1] # left module matching external row indices
        right_match_rows = [j - right_mod.num_internal_species for j in keys_2] # right module matching external row indices

        #===================================================================================================================================
        # COMPUTE LEFT AND RIGHT PHYSICAL CURRENTS AND SOLVE MATCHING CONSTRAINTS

        left_curr  = left_mod.calculate_physical_currents() # physical currents for left module
        right_curr = right_mod.calculate_physical_currents() # physical currents for right module

        left_right_current  = sympy.Matrix([left_curr[r]  for r in left_match_rows]) # left module matching external currents
        right_left_current  = sympy.Matrix([right_curr[r] for r in right_match_rows]) # right module matching external currents

        left_left_current = sympy.Matrix([left_curr[r] for r in range(left_mod.num_external_species)
            if r not in left_match_rows]) # creates a matrix of left unmatched external currents

        right_right_current = sympy.Matrix([right_curr[r] for r in range(right_mod.num_external_species)
            if r not in right_match_rows]) # creates a matrix of right unmatched external currents

        constraint_eqs = [left_right_current[k] + right_left_current[k]
            for k in range(len(left_right_current)) ] # build list of constraint equations for matching currents

        symbols_to_solve = left_right_current.free_symbols # get symbols to solve for from left matching currents
        solutions = sympy.solve(constraint_eqs, symbols_to_solve) # solve constraints for matching currents

        left_curr  = left_curr.subs(solutions) # substitute solutions into left physical currents
        right_curr = right_curr.subs(solutions) # substitute solutions into right physical currents

        left_left_current = sympy.Matrix([left_curr[r] for r in range(left_mod.num_external_species)
            if r not in left_match_rows]) # creates a matrix of left unmatched external currents after substitution

        right_right_current = sympy.Matrix([right_curr[r] for r in range(right_mod.num_external_species)
            if r not in right_match_rows]) # creates a matrix of right unmatched external currents after substitution

        combined_currents = sympy.Matrix.vstack(left_left_current, right_right_current) # creates a vector of physical currents for combined module
        self.physical_currents = combined_currents # assign to self for use in other methods

        #======================================================================================================================================
        # BUILDING THE COMBINED STOICHIOMETRIC MATRIX

        matching_stoich_left = sympy.Matrix([left_mod.external_stoich_matrix.row(r) for r in left_match_rows]) # matching external stoich rows from left module

        matching_stoich_right = sympy.Matrix([right_mod.external_stoich_matrix.row(r) for r in right_match_rows]) # matching external stoich rows from right module

        external_left_unmatched = sympy.Matrix([left_mod.external_stoich_matrix.row(r)
            for r in range(left_mod.external_stoich_matrix.rows)
            if r not in left_match_rows]) # unmatched external stoich rows from left module

        external_right_unmatched = sympy.Matrix([right_mod.external_stoich_matrix.row(r)
            for r in range(right_mod.external_stoich_matrix.rows)
            if r not in right_match_rows]) # unmatched external stoich rows from right module

        stoich_matrix = BlockMatrix([
            [left_mod.internal_stoich_matrix,
            zeros(left_mod.internal_stoich_matrix.rows, right_mod.internal_stoich_matrix.cols)],

            [matching_stoich_left,
            matching_stoich_right],

            [zeros(right_mod.internal_stoich_matrix.rows, left_mod.internal_stoich_matrix.cols),
            right_mod.internal_stoich_matrix],

            [external_left_unmatched,
            zeros(external_left_unmatched.rows, right_mod.internal_stoich_matrix.cols)],

            [zeros(external_right_unmatched.rows, left_mod.internal_stoich_matrix.cols),
            external_right_unmatched]]) # build combined stoichiometric matrix using sympy BlockMatrix

        self.stoich_matrix = sympy.Matrix(stoich_matrix) # convert BlockMatrix to regular Matrix and assign to self

        # ===================================================================================================================================
        # UPDATING ATTRIBUTES OF THE COMBINED MODULE TO WORK WITH OTHER METHODS

        self.num_internal_species = (left_mod.num_internal_species + right_mod.num_internal_species + len(keys_1)) # total internal species count for combined module

        self.num_external_species = self.stoich_matrix.rows - self.num_internal_species # total external species count for combined module
        self.num_species = self.stoich_matrix.rows # total species count for combined module
        self.num_reactions = left_mod.num_reactions + right_mod.num_reactions # total reactions count for combined module

        self.internal_stoich_matrix = self.stoich_matrix[:self.num_internal_species, :]
        self.external_stoich_matrix = self.stoich_matrix[self.num_internal_species:, :]

        # ===================================================================================================================================
        # CREATING A COMBINED DICTIONARY OF SPECIES LABELS

        left_internal_idx  = list(range(left_mod.num_internal_species)) # internal species indices for left module
        right_internal_idx = list(range(right_mod.num_internal_species)) # internal species indices for right module

        left_external_idx  = left_ext_indices # external species indices for left module
        right_external_idx = right_ext_indices # external species indices for right module

        left_ext_name_to_idx  = {left_mod.species_labels[i]: i for i in left_external_idx} # mapping of left external species names to indices
        right_ext_name_to_idx = {right_mod.species_labels[i]: i for i in right_external_idx} # mapping of right external species names to indices

        matching_names = set(left_ext_name_to_idx) & set(right_ext_name_to_idx) # find matching external species names

        left_matching_idx  = [left_ext_name_to_idx[name]  for name in matching_names] # indices of matching external species in left module
        right_matching_idx = [right_ext_name_to_idx[name] for name in matching_names] # indices of matching external species in right module

        left_unmatched_idx  = [i for i in left_external_idx  if i not in left_matching_idx] # indices of unmatched external species in left module
        right_unmatched_idx = [i for i in right_external_idx if i not in right_matching_idx] # indices of unmatched external species in right module

        combined_labels = {} # initialize empty dictionary for combined species labels
        counter = 0 # counter for combined species indices

        for i in left_internal_idx: # add left internal species labels
            combined_labels[counter] = left_mod.species_labels[i]
            counter += 1

        for i in left_matching_idx: # add matching species labels from left module
            combined_labels[counter] = left_mod.species_labels[i]
            counter += 1

        for i in right_internal_idx: # add right internal species labels
            combined_labels[counter] = right_mod.species_labels[i]
            counter += 1

        for i in left_unmatched_idx: # add unmatched species labels from left module
            combined_labels[counter] = left_mod.species_labels[i]
            counter += 1

        for i in right_unmatched_idx: # add unmatched species labels from right module
            combined_labels[counter] = right_mod.species_labels[i]
            counter += 1

        self.species_labels = combined_labels # assign to self for use in other methods
        self.species_names = list(combined_labels.values()) # list of combined species names

        # For the overlapping counts attribute

        self.matched_species_names = [left_mod.species_labels[i] for i in keys_1]
        self.left_mod              = left_mod
        self.right_mod             = right_mod

        #=========================================================================================================================================
        # Left right splitting the conservation law matrices

        left_mod_chemostat_cons_laws = left_mod.calculate_conservation_laws()[1]
        right_mod_chemostat_cons_laws = right_mod.calculate_conservation_laws()[1]

        left_mod_left_cons_laws = left_mod_chemostat_cons_laws[:, :len(left_left_current)]
        left_mod_right_cons_laws = left_mod_chemostat_cons_laws[:, len(left_left_current):]
        right_mod_left_cons_laws = right_mod_chemostat_cons_laws[:, :len(right_left_current)]
        right_mod_right_cons_laws = right_mod_chemostat_cons_laws[:, len(right_left_current):]

        #=========================================================================================================================================
        # Constructing matrices L_i, L_e and v using these to determine the chemostat conservation laws of the combined module and the selection matrix

        L_i = sympy.Matrix.vstack(-left_mod_right_cons_laws, right_mod_left_cons_laws)

        L_e = sympy.BlockMatrix([
        [left_mod_left_cons_laws, sympy.ZeroMatrix(left_mod_left_cons_laws.shape[0], right_mod_right_cons_laws.shape[1])],
        [sympy.ZeroMatrix(right_mod_right_cons_laws.shape[0], left_mod_left_cons_laws.shape[1]), right_mod_right_cons_laws]])
        
        L_e = sympy.Matrix(L_e)  

        null_basis_L_i = (L_i.T).nullspace()
        if null_basis_L_i:
            v = sympy.Matrix.hstack(*null_basis_L_i).T
        else:
            v = sympy.Matrix([])

        combined_conservation_laws_chemostat = v * L_e
        self.conservation_laws_chemostat = combined_conservation_laws_chemostat

        null_basis_cons_laws = (combined_conservation_laws_chemostat).nullspace()
        if null_basis_cons_laws:
            combined_selection_matrix = sympy.Matrix.hstack(*null_basis_cons_laws)
        else:
            combined_selection_matrix = sympy.Matrix([])

        self.selection_matrix = combined_selection_matrix

        #=========================================================================================================================================
        # Calculating pi and PI matricies 

        pi = sympy.Matrix(L_i.pinv() * L_e)
        pi_rows, pi_cols = pi.shape

        identity_part_1 = sympy.eye(len(left_left_current), pi_cols)
        pi_1_3 = identity_part_1.col_join(pi)

        identity_part_2 = sympy.eye(len(right_right_current), pi_cols-pi_rows)
        zeros_part_2 = sympy.zeros(len(right_right_current), pi_rows)
        bottom_part_2 = zeros_part_2.row_join(identity_part_2)
        pi_2_3 = (-pi).col_join(bottom_part_2)

        PI_1_3 = sympy.Matrix(left_mod.selection_matrix.pinv() * pi_1_3 * combined_selection_matrix)
        PI_2_3 = sympy.Matrix(right_mod.selection_matrix.pinv() * pi_2_3 * combined_selection_matrix)

        #=========================================================================================================================================
        # Relabelling conductance matrix of right module so that the indicies do not overlap with those of the left module

        SUB_TO_DIGIT = {'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9'}
        DIGIT_TO_SUB = {v:k for k,v in SUB_TO_DIGIT.items()}

        def parse_symbol_name(name):
            if '_' in name:
                prefix, idx = name.split('_', 1)
                if idx.isdigit():
                    return prefix, idx, 'underscore'
            i = len(name)-1
            while i >= 0 and name[i].isdigit():
                i -= 1
            if i < len(name)-1:
                return name[:i+1], name[i+1:], 'ascii'
            i = len(name)-1
            while i >= 0 and name[i] in SUB_TO_DIGIT:
                i -= 1
            if i < len(name)-1:
                prefix = name[:i+1]
                idx = ''.join(SUB_TO_DIGIT[ch] for ch in name[i+1:])
                return prefix, idx, 'unicode'
            return None, None, None

        def build_name(prefix, new_index, style):
            if style == 'underscore':
                return f"{prefix}_{new_index}"
            if style == 'unicode':
                return f"{prefix}{''.join(DIGIT_TO_SUB[d] for d in str(new_index))}"
            return f"{prefix}{new_index}"

        def shift_expr_variables(expr, shift):
            symbols = expr.atoms(sympy.Symbol)
            if not symbols:
                return expr
            subs = {}
            for s in symbols:
                prefix, idx_str, style = parse_symbol_name(s.name)
                if prefix is None:
                    continue
                new_idx = int(idx_str) + int(shift)
                new_name = build_name(prefix, new_idx, style)
                subs[s] = sympy.Symbol(new_name, **s.assumptions0)
            return expr.xreplace(subs)

        def shift_matrix_variables(matrix, shift):
            return matrix.applyfunc(lambda e: shift_expr_variables(e, shift))
        
        #  If we have numerical data for the CMs

        if left_mod_numerical_CM is not None and right_mod_numerical_CM is not None:

            if len(left_mod_numerical_CM) != len(right_mod_numerical_CM):

                raise ValueError("The number of numerical conductance matrices provided for the left and right modules must be the same.")

            self.numerical_combined_fundamental_CMs = []

            for i in range(len(left_mod_numerical_CM)):

                left_CM = left_mod_numerical_CM[i]
                right_CM = right_mod_numerical_CM[i]


                #=========================================================================================================================================
                # Calculating the conductance matrix of the combined module

                combined_fundamental_resistance_matrix = PI_1_3.T * left_CM.inv() * PI_1_3 + PI_2_3.T * right_CM.inv() * PI_2_3

                self.numerical_combined_fundamental_CMs.append(combined_fundamental_resistance_matrix.inv())

                #=========================================================================================================================================

        else:

            left_mod_fundamental_resistance_matrix = left_mod.fundamental_resistance_matrix
            right_mod_fundamental_resistance_matrix = shift_matrix_variables(right_mod.fundamental_resistance_matrix, left_mod.num_reactions)

            #=========================================================================================================================================
            # Calculating the conductance matrix of the combined module

            combined_fundamental_resistance_matrix = PI_1_3.T * left_mod_fundamental_resistance_matrix * PI_1_3 + PI_2_3.T * right_mod_fundamental_resistance_matrix * PI_2_3

            #=========================================================================================================================================

        # Storing attributes to self that are need for an iterative process of combining modules

        self.fundamental_resistance_matrix = combined_fundamental_resistance_matrix
        self.fundamental_conductance_matrix = combined_fundamental_resistance_matrix.inv()
        self.selection_matrix = combined_selection_matrix
        self.conservation_laws_chemostat = combined_conservation_laws_chemostat

    def calculate_physical_currents(self):
        
        return self.physical_currents
    
    def calculate_conservation_laws(self):
        
        return 0, self.conservation_laws_chemostat
    
    #=========================================================================================================================================
    # Create new counts list automatically from the counts lists of the two modules (From Claude Sonnet 4.6)

    def build_combined_initial_counts_and_rates(self, left_initial_counts, right_initial_counts, left_rates, right_rates):
        """
        Merges two initial count dictionaries into the combined module species
        ordering, prompting the user for values of overlapping species. Combines rates from two modules 
        into the correct ordering for the combined module. Since reactions are ordered as [left | right] in 
        the combined stoichiometric matrix, this is a direct concatenation.

        Parameters
        ----------
        left_initial_counts : dict
            {species_name: count} for left module.
        right_initial_counts : dict
            {species_name: count} for right module.
        left_rates : list
            Rates for left module reactions, ordered as forward/backward 
            pairs [k1+, k1-, k2+, k2-, ...].
        right_rates : list
            Rates for right module reactions, ordered as forward/backward
            pairs [k4+, k4-, k5+, k5-, ...].

        Returns
        -------
        combined_initial_counts : list
            Initial counts in the correct species order for the combined module.
        combined_rates : list
            Combined rates in correct order for the combined module. 
        """

        left_initial_counts  = dict(zip(self.left_mod.species_names, left_initial_counts))
        right_initial_counts = dict(zip(self.right_mod.species_names, right_initial_counts))

        # ── Prompt user for overlapping species ──────────────────────────────────

        overlap_values = {}

        print("\n=== Overlapping species detected ===")
        print(f"  {self.matched_species_names}")
        print("These species appear in both modules and are now internal.")
        print("Please enter a single initial count for each:\n")

        for name in self.matched_species_names:

            left_val  = left_initial_counts.get(name,  None)
            right_val = right_initial_counts.get(name, None)

            print(f"  Species '{name}':")
            if left_val  is not None: print(f"    Left  module value : {left_val}")
            if right_val is not None: print(f"    Right module value : {right_val}")

            while True:
                try:
                    user_val = float(input(f"  Enter initial count for '{name}': "))
                    overlap_values[name] = user_val
                    break
                except ValueError:
                    print("  Invalid input. Please enter a number.")

        #  Merge into combined species order

        combined_initial_counts = []

        for idx, name in self.species_labels.items():

            if name in overlap_values:
                # for overlapping species use user value
                combined_initial_counts.append(overlap_values[name])

            elif name in left_initial_counts:
                # Left module species
                combined_initial_counts.append(left_initial_counts[name])

            elif name in right_initial_counts:
                # Right module species
                combined_initial_counts.append(right_initial_counts[name])

            else:
                # If not found in either prompt user
                print(f"\nWarning: '{name}' not found in either initial count dict.")
                while True:
                    try:
                        user_val = float(input(f"  Enter initial count for '{name}': "))
                        combined_initial_counts.append(user_val)
                        break
                    except ValueError:
                        print("  Invalid input. Please enter a number.")

        # Print the new combined initial counts

        print("\n=== Combined initial counts ===")
        for idx, (name, val) in enumerate(
            zip(self.species_names, combined_initial_counts)
        ):
            overlap_flag = ' ← user entered' if name in self.matched_species_names else ''
            print(f"  [{idx}] {name:12s} : {val}{overlap_flag}")

        

       # Now do the rates, simply need to concatenate the lists

        # Validate lengths against known number of reactions
        expected_left  = self.left_mod.num_reactions  * 2   # forward + backward per reaction
        expected_right = self.right_mod.num_reactions * 2

        if len(left_rates) != expected_left:
            raise ValueError(
                f"Expected {expected_left} rates for left module "
                f"({self.left_mod.num_reactions} reactions × 2), "
                f"got {len(left_rates)}."
            )

        if len(right_rates) != expected_right:
            raise ValueError(
                f"Expected {expected_right} rates for right module "
                f"({self.right_mod.num_reactions} reactions × 2), "
                f"got {len(right_rates)}."
            )

        combined_rates = list(left_rates) + list(right_rates)

        # Print the new combined rates

        print("\n=== Combined rates ===")
        rxn_idx = 1
        for i in range(0, len(combined_rates), 2):
            module_label = (
                'left'  if i <  expected_left else
                'right'
            )
            print(f"  Reaction {rxn_idx:2d} ({module_label:5s}) : "
                f"k+ = {combined_rates[i]:.4g},  "
                f"k- = {combined_rates[i+1]:.4g}")
            rxn_idx += 1

        return combined_initial_counts, combined_rates

        

        