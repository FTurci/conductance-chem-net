
# ModuleProperties for converting stoichiometry into combinable module

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

        
# CombiningModules class and helper functions for serial associaiton of module objects.

def primitive_nullspace(matrix):
    null_basis = matrix.nullspace()
    if not null_basis:
        return Matrix([])
    cols = []
    for vec in null_basis:
        denoms = [fraction(x)[1] for x in vec]
        lcm_denom = Integer(1)
        for d in denoms:
            lcm_denom = lcm(lcm_denom, d)
        vec = vec * lcm_denom
        numers = [abs(x) for x in vec if x != 0]
        if numers:
            gcd_numer = numers[0]
            for n in numers[1:]:
                gcd_numer = gcd(gcd_numer, n)
            vec = vec / gcd_numer
        first_nonzero = next((vec[i] for i in range(len(vec)) if vec[i] != 0), 0)
        if first_nonzero < 0:
            vec = -vec
        cols.append(vec)
    return Matrix.hstack(*cols)

def normalise_selection_matrix(S):
    """
    Scale each column so that S.T * S = I
    while preserving the number of rows.
    """
    cols = []
    for j in range(S.cols):
        col = S.col(j)
        norm_sq = (col.T * col)[0, 0]
        if norm_sq == 0:
            cols.append(col)
        else:
            cols.append(col / sqrt(norm_sq))
    return Matrix.hstack(*cols)


def build_selection_matrix(full_Lambda):
    rows_to_keep = [row for row in range(full_Lambda.rows)
                    if any(full_Lambda[row, col] != 0
                           for col in range(full_Lambda.cols))]
    if rows_to_keep:
        Lambda_reduced = full_Lambda.extract(
            rows_to_keep, list(range(full_Lambda.cols)))
    else:
        Lambda_reduced = full_Lambda
    return primitive_nullspace(Lambda_reduced)


# ── symbol-shifting helpers (module-level so Step 2 can use them) ──────────

SUB_TO_DIGIT = {'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4',
                '₅':'5','₆':'6','₇':'7','₈':'8','₉':'9'}
DIGIT_TO_SUB = {v: k for k, v in SUB_TO_DIGIT.items()}

def _parse_symbol_name(name):
    if '_' in name:
        prefix, idx = name.split('_', 1)
        if idx.isdigit():
            return prefix, idx, 'underscore'
    i = len(name) - 1
    while i >= 0 and name[i].isdigit(): i -= 1
    if i < len(name) - 1:
        return name[:i+1], name[i+1:], 'ascii'
    i = len(name) - 1
    while i >= 0 and name[i] in SUB_TO_DIGIT: i -= 1
    if i < len(name) - 1:
        return (name[:i+1],
                ''.join(SUB_TO_DIGIT[c] for c in name[i+1:]),
                'unicode')
    return None, None, None

def _build_name(prefix, new_index, style):
    if style == 'underscore': return f"{prefix}_{new_index}"
    if style == 'unicode':
        return prefix + ''.join(DIGIT_TO_SUB[d] for d in str(new_index))
    return f"{prefix}{new_index}"

def shift_expr_variables(expr, shift):
    syms = expr.atoms(Symbol)
    if not syms: return expr
    subs = {}
    for s in syms:
        prefix, idx_str, style = _parse_symbol_name(s.name)
        if prefix is None: continue
        subs[s] = Symbol(
            _build_name(prefix, int(idx_str) + int(shift), style),
            **s.assumptions0)
    return expr.xreplace(subs)

def shift_matrix_variables(matrix, shift):
    return matrix.applyfunc(lambda e: shift_expr_variables(e, shift))


class CombiningModules:

    def __init__(self, left_mod, right_mod,
                 left_mod_numerical_CM=None, right_mod_numerical_CM=None):

        #=====================================================================
        # 0. SYMBOL SHIFTING HELPERS (also available inside __init__)
        # (re-bound as locals for clarity; module-level definitions are used
        #  by external callers)
        _shift_expr  = shift_expr_variables
        _shift_mat   = shift_matrix_variables

        n_rxn_l = left_mod.num_reactions
        n_rxn_r = right_mod.num_reactions

        print("Direct derivation force ordering:")
        print("  row/col 0: F-W        → resistance r4+r5+r6")
        print("  row/col 1: ???        → resistance r10+r9")
        print("  row/col 2: S-Pb       → resistance r1+r11+r12+r2+r5+r7+r8")  
        print("  row/col 3: Ne-Pe      → resistance r13+r14")

        print(f"\nr9  = resistance of reaction 9  in M3 (shifted from r1 in M3 unshifted)")
        print(f"r10 = resistance of reaction 10 in M3 (shifted from r2 in M3 unshifted)")
        print(f"r11 = resistance of reaction 11 in M3")
        print(f"r12 = resistance of reaction 12 in M3")
        print(f"r13 = resistance of reaction 13 in M3")
        print(f"r14 = resistance of reaction 14 in M3")

        # What reactions in M3 correspond to r9, r10?
        print(f"\nM3 has reactions c1..c6, shifted by n_rxn_l={n_rxn_l}")
        print(f"So M3 reaction c1 → r{n_rxn_l+1}, c2 → r{n_rxn_l+2}, etc.")
        print(f"r9=c1, r10=c2, r11=c3, r12=c4, r13=c5, r14=c6")
        print(f"\nc1,c2 form cycle ε_c  (Nex→Pex): resistance r9+r10")
        print(f"c3,c4 form cycle ε_c' (Nb→Pb):   resistance r11+r12") 
        print(f"c5,c6 form cycle ε_c''(Ne→Pe):   resistance r13+r14")

        #=====================================================================
        # 1. IDENTIFY MATCHING EXTERNAL SPECIES BY NAME
        print("\n" + "="*60)
        print("STEP 1: IDENTIFY INTERFACE SPECIES")
        print("="*60)

        left_ext_indices  = list(range(left_mod.num_internal_species,
                                       left_mod.num_species))
        right_ext_indices = list(range(right_mod.num_internal_species,
                                       right_mod.num_species))

        left_ext_names_list  = [left_mod.species_labels[i] for i in left_ext_indices]
        right_ext_names_list = [right_mod.species_labels[j] for j in right_ext_indices]

        interface_names = [name for name in left_ext_names_list
                           if name in right_ext_names_list]

        left_interface_rows  = [left_ext_names_list.index(name)
                                 for name in interface_names]
        right_interface_rows = [right_ext_names_list.index(name)
                                 for name in interface_names]

        left_free_rows  = [i for i in range(len(left_ext_names_list))
                           if i not in left_interface_rows]
        right_free_rows = [i for i in range(len(right_ext_names_list))
                           if i not in right_interface_rows]

        print(f"left_ext_names_list:  {left_ext_names_list}")
        print(f"right_ext_names_list: {right_ext_names_list}")
        print(f"interface_names:      {interface_names}")
        print(f"left_free_rows:       {left_free_rows}")
        print(f"right_free_rows:      {right_free_rows}")

        #=====================================================================
        # 2. PHYSICAL CURRENTS
        print("\n" + "="*60)
        print("STEP 2: PHYSICAL CURRENTS")
        print("="*60)

        left_curr      = left_mod.calculate_physical_currents()
        right_curr_raw = right_mod.calculate_physical_currents()

        # Shift right module's current symbols by n_rxn_l to avoid
        # collisions with left module's symbols (j1, j2, ... must be disjoint)
        right_curr = right_curr_raw.applyfunc(
            lambda e: _shift_expr(e, n_rxn_l))

        print(f"left_curr:       {left_curr.T}")
        print(f"right_curr_raw:  {right_curr_raw.T}")
        print(f"right_curr (shifted): {right_curr.T}")

        i1_r = Matrix([left_curr[r]  for r in left_interface_rows])
        i2_l = Matrix([right_curr[r] for r in right_interface_rows])
        i1_l = Matrix([left_curr[r]  for r in left_free_rows])
        i2_r = Matrix([right_curr[r] for r in right_free_rows])

        constraint_eqs   = [i1_r[k] + i2_l[k] for k in range(len(i1_r))]
        symbols_to_solve = i1_r.free_symbols
        solutions        = solve(constraint_eqs, symbols_to_solve)

        i1_l = i1_l.subs(solutions)
        i2_r = i2_r.subs(solutions)
        i1_r = i1_r.subs(solutions)

        self.physical_currents = Matrix.vstack(i1_l, i2_r)

        print(f"i1_l: {i1_l.T}")
        print(f"i1_r: {i1_r.T}")
        print(f"i2_l: {i2_l.T}")
        print(f"i2_r: {i2_r.T}")
        print(f"physical_currents: {self.physical_currents.T}")

        #=====================================================================
        # 3. BUILD COMBINED STOICHIOMETRIC MATRIX
        print("\n" + "="*60)
        print("STEP 3: COMBINED STOICHIOMETRIC MATRIX")
        print("="*60)

        match_stoich_left  = Matrix([left_mod.external_stoich_matrix.row(r)
                                     for r in left_interface_rows])
        match_stoich_right = Matrix([right_mod.external_stoich_matrix.row(r)
                                     for r in right_interface_rows])
        ext_left_free  = Matrix([left_mod.external_stoich_matrix.row(r)
                                 for r in left_free_rows])
        ext_right_free = Matrix([right_mod.external_stoich_matrix.row(r)
                                 for r in right_free_rows])

        stoich_matrix = BlockMatrix([
            [left_mod.internal_stoich_matrix,
             zeros(left_mod.internal_stoich_matrix.rows, n_rxn_r)],
            [match_stoich_left,  match_stoich_right],
            [zeros(right_mod.internal_stoich_matrix.rows, n_rxn_l),
             right_mod.internal_stoich_matrix],
            [ext_left_free,
             zeros(ext_left_free.rows, n_rxn_r)],
            [zeros(ext_right_free.rows, n_rxn_l),
             ext_right_free]])

        self.stoich_matrix = Matrix(stoich_matrix)
        print(f"stoich_matrix shape: {self.stoich_matrix.shape}")

        #=====================================================================
        # 4. MODULE ATTRIBUTES

        self.num_internal_species = (left_mod.num_internal_species +
                                     right_mod.num_internal_species +
                                     len(interface_names))
        self.num_external_species = self.stoich_matrix.rows - self.num_internal_species
        self.num_species          = self.stoich_matrix.rows
        self.num_reactions        = n_rxn_l + n_rxn_r

        self.internal_stoich_matrix = self.stoich_matrix[:self.num_internal_species, :]
        self.external_stoich_matrix = self.stoich_matrix[self.num_internal_species:, :]

        #=====================================================================
        # 5. SPECIES LABELS

        combined_labels = {}
        counter = 0
        for i in range(left_mod.num_internal_species):
            combined_labels[counter] = left_mod.species_labels[i]; counter += 1
        for name in interface_names:
            combined_labels[counter] = name; counter += 1
        for i in range(right_mod.num_internal_species):
            combined_labels[counter] = right_mod.species_labels[i]; counter += 1
        for i in left_free_rows:
            combined_labels[counter] = left_ext_names_list[i]; counter += 1
        for i in right_free_rows:
            combined_labels[counter] = right_ext_names_list[i]; counter += 1

        self.species_labels        = combined_labels
        self.species_names         = list(combined_labels.values())
        self.matched_species_names = interface_names
        self.left_mod              = left_mod
        self.right_mod             = right_mod

        self.chemostat_species_names = (
            [left_ext_names_list[i]  for i in left_free_rows] +
            [right_ext_names_list[i] for i in right_free_rows])

        #=====================================================================
        # 6. CONSERVATION LAW SPLITTING
        print("\n" + "="*60)
        print("STEP 6: CONSERVATION LAW SPLITTING")
        print("="*60)

        left_Lambda  = left_mod.calculate_conservation_laws()[1]
        right_Lambda = right_mod.calculate_conservation_laws()[1]

        if hasattr(left_mod, 'chemostat_species_names'):
            left_cons_col_names = left_mod.chemostat_species_names
        else:
            left_cons_col_names = left_ext_names_list

        if hasattr(right_mod, 'chemostat_species_names'):
            right_cons_col_names = right_mod.chemostat_species_names
        else:
            right_cons_col_names = right_ext_names_list

        left_interface_cons_cols  = [left_cons_col_names.index(name)
                                     for name in interface_names]
        left_free_cons_cols       = [i for i in range(len(left_cons_col_names))
                                     if i not in left_interface_cons_cols]

        right_interface_cons_cols = [right_cons_col_names.index(name)
                                     for name in interface_names]
        right_free_cons_cols      = [i for i in range(len(right_cons_col_names))
                                     if i not in right_interface_cons_cols]

        Lambda1_l = left_Lambda[:,  left_free_cons_cols]
        Lambda1_r = left_Lambda[:,  left_interface_cons_cols]
        Lambda2_l = right_Lambda[:, right_interface_cons_cols]
        Lambda2_r = right_Lambda[:, right_free_cons_cols]

        print(f"left_Lambda:\n{left_Lambda}")
        print(f"right_Lambda:\n{right_Lambda}")
        print(f"left_cons_col_names:  {left_cons_col_names}")
        print(f"right_cons_col_names: {right_cons_col_names}")
        print(f"left_interface_cons_cols:  {left_interface_cons_cols}")
        print(f"left_free_cons_cols:       {left_free_cons_cols}")
        print(f"right_interface_cons_cols: {right_interface_cons_cols}")
        print(f"right_free_cons_cols:      {right_free_cons_cols}")
        print(f"Lambda1_l:\n{Lambda1_l}")
        print(f"Lambda1_r:\n{Lambda1_r}")
        print(f"Lambda2_l:\n{Lambda2_l}")
        print(f"Lambda2_r:\n{Lambda2_r}")

        left_mixed_rows = []
        for row in range(left_Lambda.rows):
            has_interface = any(left_Lambda[row, c] != 0
                                for c in left_interface_cons_cols)
            has_free      = any(left_Lambda[row, c] != 0
                                for c in left_free_cons_cols)
            if has_interface and has_free:
                left_mixed_rows.append(row)

        right_mixed_rows = []
        for row in range(right_Lambda.rows):
            has_interface = any(right_Lambda[row, c] != 0
                                for c in right_interface_cons_cols)
            has_free      = any(right_Lambda[row, c] != 0
                                for c in right_free_cons_cols)
            if has_interface and has_free:
                right_mixed_rows.append(row)

        left_pure_rows  = [r for r in range(left_Lambda.rows)
                           if r not in left_mixed_rows]
        right_pure_rows = [r for r in range(right_Lambda.rows)
                           if r not in right_mixed_rows]

        print(f"\nleft_mixed_rows:  {left_mixed_rows}")
        print(f"left_pure_rows:   {left_pure_rows}")
        print(f"right_mixed_rows: {right_mixed_rows}")
        print(f"right_pure_rows:  {right_pure_rows}")

        Lambda1_l_pure = (left_Lambda.extract(left_pure_rows, left_free_cons_cols)
                          if left_pure_rows
                          else Matrix(zeros(0, len(left_free_cons_cols))))
        Lambda2_r_pure = (right_Lambda.extract(right_pure_rows, right_free_cons_cols)
                          if right_pure_rows
                          else Matrix(zeros(0, len(right_free_cons_cols))))

        print(f"\nLambda1_l_pure:\n{Lambda1_l_pure}")
        print(f"Lambda2_r_pure:\n{Lambda2_r_pure}")

        #=====================================================================
        # 7. BUILD L_i AND L_e
        print("\n" + "="*60)
        print("STEP 7: L_i AND L_e")
        print("="*60)

        L_i = Matrix.vstack(-Lambda1_r, Lambda2_l)

        L_e = Matrix(BlockMatrix([
            [Lambda1_l,
             ZeroMatrix(Lambda1_l.rows, Lambda2_r.cols)],
            [ZeroMatrix(Lambda2_r.rows, Lambda1_l.cols),
             Lambda2_r]]))

        print(f"L_i shape: {L_i.shape}\n{L_i}")
        print(f"L_e shape: {L_e.shape}\n{L_e}")

        #=====================================================================
        # 8. COKERNEL OF L_i
        print("\n" + "="*60)
        print("STEP 8: COKERNEL v")
        print("="*60)

        null_basis_L_i = (L_i.T).nullspace()
        v = Matrix.hstack(*null_basis_L_i).T if null_basis_L_i else Matrix([])

        print(f"v shape: {v.shape}\n{v}")

        #=====================================================================
        # 9. COMBINED CONSERVATION LAWS
        print("\n" + "="*60)
        print("STEP 9: LAMBDA_COMBINED")
        print("="*60)

        Lambda_combined_raw = v * L_e

        print(f"Lambda_combined_raw shape: {Lambda_combined_raw.shape}")
        print(f"Lambda_combined_raw rank:  {Lambda_combined_raw.rank()}")
        print(f"Lambda_combined_raw:\n{Lambda_combined_raw}")

        _, pivot_rows = Lambda_combined_raw.T.rref()
        Lambda_combined = Lambda_combined_raw.extract(
            list(pivot_rows), list(range(Lambda_combined_raw.cols)))

        print(f"Lambda_combined (after rref row reduction):\n{Lambda_combined}")
        print(f"Lambda_combined shape: {Lambda_combined.shape}")
        print(f"Lambda_combined rank:  {Lambda_combined.rank()}")

        #=====================================================================
        # 10. SELECTION MATRIX S3
        print("\n" + "="*60)
        print("STEP 10: S3")
        print("="*60)

        S3 = primitive_nullspace(Lambda_combined)
        print(f"S3:\n{S3}")
        print(f"S3 shape: {S3.shape}")
        print(f"S3.T * S3:\n{S3.T * S3}")

        print("S3 columns vs chemostat species:")
        print(f"chemostat_species_names: {self.chemostat_species_names}")
        print(f"S3 columns represent force directions:")
        for j in range(S3.cols):
            col = S3.col(j)
            direction = [(self.chemostat_species_names[i], int(col[i])) 
                        for i in range(col.rows) if col[i] != 0]
            print(f"  col {j}: {direction}")

        #=====================================================================
        # 11. COMPUTE pi
        print("\n" + "="*60)
        print("STEP 11: pi")
        print("="*60)

        pi = Matrix(L_i.pinv() * L_e)
        pi_rows, pi_cols = pi.shape

        print(f"pi:\n{pi}")
        print(f"L_i.T * L_i:\n{L_i.T * L_i}")

        #=====================================================================
        # 12. BUILD π^(1,3) AND π^(2,3)
        print("\n" + "="*60)
        print("STEP 12: pi_1_3 AND pi_2_3")
        print("="*60)

        if hasattr(left_mod, 'chemostat_species_names'):
            left_phys_curr_order = left_mod.chemostat_species_names
        else:
            left_phys_curr_order = left_ext_names_list

        if hasattr(right_mod, 'chemostat_species_names'):
            right_phys_curr_order = right_mod.chemostat_species_names
        else:
            right_phys_curr_order = right_ext_names_list

        n1_l = len(i1_l)
        n2_r = len(i2_r)

        left_free_names  = [n for n in left_phys_curr_order
                            if n not in interface_names]
        right_free_names = [n for n in right_phys_curr_order
                            if n not in interface_names]

        print(f"left_phys_curr_order:  {left_phys_curr_order}")
        print(f"right_phys_curr_order: {right_phys_curr_order}")
        print(f"left_free_names:  {left_free_names}")
        print(f"right_free_names: {right_free_names}")

        pi_1_3_rows = []
        pi_interface_row_idx = 0
        for name in left_phys_curr_order:
            if name in interface_names:
                pi_1_3_rows.append(pi[pi_interface_row_idx, :])
                pi_interface_row_idx += 1
            else:
                # Index into combined chemostat ordering
                combined_idx = self.chemostat_species_names.index(name)
                identity_row = zeros(1, pi_cols)
                identity_row[0, combined_idx] = 1  # ← FIX
                pi_1_3_rows.append(identity_row)

        pi_1_3 = Matrix.vstack(*pi_1_3_rows)

        pi_2_3_rows = []
        pi_interface_row_idx = 0
        for name in right_phys_curr_order:
            if name in interface_names:
                pi_2_3_rows.append(-pi[pi_interface_row_idx, :])
                pi_interface_row_idx += 1
            else:
                # Index into combined chemostat ordering, not right_free_names
                combined_idx = self.chemostat_species_names.index(name)
                identity_row = zeros(1, pi_cols)
                identity_row[0, combined_idx] = 1  # ← FIX
                pi_2_3_rows.append(identity_row)

        pi_2_3 = Matrix.vstack(*pi_2_3_rows)

        print(f"pi_1_3:\n{pi_1_3}")
        print(f"pi_2_3:\n{pi_2_3}")

        #=====================================================================
        # 13. COMPUTE S1, S2 AND Π^(1,3), Π^(2,3)
        print("\n" + "="*60)
        print("STEP 13: S1, S2, PI_1_3, PI_2_3")
        print("="*60)

        S1 = -left_mod.selection_matrix
        S2 = -right_mod.selection_matrix

        print(f"\nLambda_combined:\n{Lambda_combined}")
        print(f"chemostat_species_names: {self.chemostat_species_names}")
        print(f"S1:\n{S1}")
        print(f"S2:\n{S2}")
        print(f"S2 shape: {S2.shape}, rank: {S2.rank()}")
        print(f"S2.T * S2:\n{S2.T * S2}")

        print(f"\nS1.pinv():\n{S1.pinv()}")
        print(f"S2.pinv():\n{S2.pinv()}")

        PI_1_3 = Matrix(S1.pinv() * pi_1_3 * S3)
        PI_2_3 = Matrix(S2.pinv() * pi_2_3 * S3)

        print(f"\nKEY DIAGNOSTIC:")
        print(f"S1.pinv() * pi_1_3 * S3 =\n{S1.pinv() * pi_1_3 * S3}")
        print(f"S2.pinv() * pi_2_3 * S3 =\n{S2.pinv() * pi_2_3 * S3}")

        # Also show S3 columns split by left/right species
        print(f"\nS3 column decomposition (left={left_free_names}, right={right_free_names}):")
        n_left = len(left_free_names)
        for j in range(S3.cols):
            col = S3.col(j)
            left_part  = col[:n_left, :]
            right_part = col[n_left:, :]
            print(f"  col {j}: left_part={left_part.T}  right_part={right_part.T}")
            mixes = any(x != 0 for x in left_part) and any(x != 0 for x in right_part)
            print(f"          mixes left+right? {mixes}  ← {'⚠ causes fractions' if mixes else '✓ clean'}")

        print(f"\nPI_1_3:\n{PI_1_3}")
        print(f"PI_1_3 shape: {PI_1_3.shape}, rank: {PI_1_3.rank()}")
        print(f"\nPI_2_3:\n{PI_2_3}")
        print(f"PI_2_3 shape: {PI_2_3.shape}, rank: {PI_2_3.rank()}")

        print(f"\nS3 column analysis:")
        for j in range(S3.cols):
            col = S3.col(j)
            direction = [(self.chemostat_species_names[i], col[i])
                         for i in range(col.rows) if col[i] != 0]
            p1 = pi_1_3 * col
            p2 = pi_2_3 * col
            print(f"  col {j}: {direction}")
            print(f"    pi_1_3*col={p1.T} → S1.pinv()*={( S1.pinv()*p1).T}")
            print(f"    pi_2_3*col={p2.T} → S2.pinv()*={(S2.pinv()*p2).T}")

        self.big_PI_13 = PI_1_3
        self.big_PI_23 = PI_2_3

        print(f"S2+ * (pi_2_3 * S3):\n{S2.pinv() * pi_2_3 * S3}")
        print(f"S1+ * (pi_1_3 * S3):\n{S1.pinv() * pi_1_3 * S3}")

        #=====================================================================
        # 15. COMBINED RESISTANCE MATRIX
        print("\n" + "="*60)
        print("STEP 15: COMBINED RESISTANCE MATRIX")
        print("="*60)

        if left_mod_numerical_CM is not None and right_mod_numerical_CM is not None:

            if len(left_mod_numerical_CM) != len(right_mod_numerical_CM):
                raise ValueError(
                    "Left and right numerical CM lists must have same length.")

            self.numerical_combined_fundamental_CMs = []
            for i in range(len(left_mod_numerical_CM)):
                G1_num = left_mod_numerical_CM[i]
                G2_num = right_mod_numerical_CM[i]
                R_comb = (PI_1_3.T * G1_num.inv() * PI_1_3 +
                          PI_2_3.T * G2_num.inv() * PI_2_3)
                self.numerical_combined_fundamental_CMs.append(R_comb.inv())

            combined_fundamental_resistance_matrix = (
                PI_1_3.T * left_mod_numerical_CM[0].inv()  * PI_1_3 +
                PI_2_3.T * right_mod_numerical_CM[0].inv() * PI_2_3)

        else:
            is_left_base  = not hasattr(left_mod,  'chemostat_species_names')
            is_right_base = not hasattr(right_mod, 'chemostat_species_names')

            R1_sym = left_mod.fundamental_resistance_matrix

            if is_right_base:
                R2_sym = shift_matrix_variables(
                    right_mod.fundamental_resistance_matrix, n_rxn_l)
            else:
                R2_sym = shift_matrix_variables(
                    right_mod.fundamental_resistance_matrix,
                    left_mod.num_reactions)

            print(f"R1_sym:\n{R1_sym}")
            print(f"R2_sym:\n{R2_sym}")
            print(f"PI_1_3:\n{PI_1_3}")
            print(f"PI_2_3:\n{PI_2_3}")
            print(f"PI_1_3.T * R1_sym * PI_1_3:\n{PI_1_3.T * R1_sym * PI_1_3}")
            print(f"PI_2_3.T * R2_sym * PI_2_3:\n{PI_2_3.T * R2_sym * PI_2_3}")

            combined_fundamental_resistance_matrix = (
                PI_1_3.T * R1_sym * PI_1_3 +
                PI_2_3.T * R2_sym * PI_2_3)

        print(f"\nR_combined:\n{combined_fundamental_resistance_matrix}")
        print(f"R_combined rank:  {combined_fundamental_resistance_matrix.rank()}")
        print(f"R_combined shape: {combined_fundamental_resistance_matrix.shape}")

        zero_rows = [i for i in range(combined_fundamental_resistance_matrix.rows)
                     if all(combined_fundamental_resistance_matrix[i, j] == 0
                            for j in range(combined_fundamental_resistance_matrix.cols))]
        zero_cols = [j for j in range(combined_fundamental_resistance_matrix.cols)
                     if all(combined_fundamental_resistance_matrix[i, j] == 0
                            for i in range(combined_fundamental_resistance_matrix.rows))]
        print(f"Zero rows: {zero_rows}")
        print(f"Zero cols: {zero_cols}")

        #=====================================================================
        # 16. STORE FINAL ATTRIBUTES

        self.fundamental_resistance_matrix  = combined_fundamental_resistance_matrix
        # self.fundamental_conductance_matrix = combined_fundamental_resistance_matrix.inv()
        self.selection_matrix               = S3
        self.conservation_laws_chemostat    = Lambda_combined

        # print(f"\nDiagnostics:")
        # print(f"S1.pinv() * S1:\n{S1.pinv() * S1}")
        # print(f"S2.pinv() * S2:\n{S2.pinv() * S2}")
        # print(f"S3.pinv() * S3:\n{S3.pinv() * S3}")
        # print(f"PI_1_3.T*PI_1_3 + PI_2_3.T*PI_2_3 rank: "
        #       f"{(PI_1_3.T*PI_1_3 + PI_2_3.T*PI_2_3).rank()}")

        # print("=== TRACING PI_2_3 ===")

        # # Get external species names correctly
        # left_mod_ext_names  = left_mod.species_names[left_mod.num_internal_species:]
        # right_mod_ext_names = right_mod.species_names[right_mod.num_internal_species:]

        # print(f"\nleft_mod external species:  {left_mod_ext_names}")
        # print(f"right_mod external species: {right_mod_ext_names}")

        # print(f"\nright_phys_curr_order: {right_phys_curr_order}")

        # print(f"\nS2 = -right_mod.selection_matrix columns:")
        # S2_orig = right_mod.selection_matrix
        # for j in range(S2_orig.cols):
        #     col = S2_orig.col(j)
        #     direction = [(right_mod_ext_names[i], int(col[i])) 
        #                 for i in range(col.rows) if col[i] != 0]
        #     print(f"  S2_orig col {j}: {direction}")

        # print(f"\nS2 (negated) columns:")
        # for j in range(S2.cols):
        #     col = S2.col(j)
        #     direction = [(right_mod_ext_names[i], int(col[i])) 
        #                 for i in range(col.rows) if col[i] != 0]
        #     print(f"  S2 col {j}: {direction}")

        # print(f"\nFor each S3 column, what does pi_2_3 map it to?")
        # for j in range(S3.cols):
        #     s3col   = S3.col(j)
        #     mapped  = pi_2_3 * s3col
        #     s3_dir  = [(self.chemostat_species_names[i], int(s3col[i])) 
        #                 for i in range(s3col.rows) if s3col[i] != 0]
        #     mapped_dir = [(right_phys_curr_order[i], mapped[i]) 
        #                 for i in range(mapped.rows) if mapped[i] != 0]
        #     coords  = S2.pinv() * mapped
        #     print(f"\n  S3 col {j}: {s3_dir}")
        #     print(f"    pi_2_3 maps to (in right_phys_curr_order): {mapped_dir}")
        #     print(f"    S2.pinv() * mapped = {coords.T}")
        #     print(f"    integer entries?   {all(x == int(x) for x in coords)}")
        #     # Show which M3 cycle this corresponds to
        #     print(f"    → this is PI_2_3 col {j} = {(S2.pinv()*mapped).T}")

    #=========================================================================
    def calculate_physical_currents(self):
        return self.physical_currents

    def calculate_conservation_laws(self):
        return 0, self.conservation_laws_chemostat

    #=========================================================================
    def build_combined_initial_counts_and_rates(
            self, left_initial_counts, right_initial_counts,
            left_rates, right_rates):

        left_initial_counts  = dict(zip(self.left_mod.species_names,
                                        left_initial_counts))
        right_initial_counts = dict(zip(self.right_mod.species_names,
                                        right_initial_counts))

        overlap_values = {}
        print("\n=== Overlapping species detected ===")
        print(f"  {self.matched_species_names}")
        print("These species appear in both modules and are now internal.\n")

        for name in self.matched_species_names:
            left_val  = left_initial_counts.get(name)
            right_val = right_initial_counts.get(name)
            print(f"  Species '{name}':")
            if left_val  is not None: print(f"    Left  module value: {left_val}")
            if right_val is not None: print(f"    Right module value: {right_val}")
            while True:
                try:
                    overlap_values[name] = float(
                        input(f"  Enter initial count for '{name}': "))
                    break
                except ValueError:
                    print("  Invalid. Enter a number.")

        combined_initial_counts = []
        for idx, name in self.species_labels.items():
            if name in overlap_values:
                combined_initial_counts.append(overlap_values[name])
            elif name in left_initial_counts:
                combined_initial_counts.append(left_initial_counts[name])
            elif name in right_initial_counts:
                combined_initial_counts.append(right_initial_counts[name])
            else:
                print(f"\nWarning: '{name}' not found.")
                while True:
                    try:
                        combined_initial_counts.append(
                            float(input(f"  Enter initial count for '{name}': ")))
                        break
                    except ValueError:
                        print("  Invalid. Enter a number.")

        print("\n=== Combined initial counts ===")
        for idx, (name, val) in enumerate(
                zip(self.species_names, combined_initial_counts)):
            flag = (' ← user entered'
                    if name in self.matched_species_names else '')
            print(f"  [{idx}] {name:12s} : {val}{flag}")

        expected_left  = self.left_mod.num_reactions * 2
        expected_right = self.right_mod.num_reactions * 2

        if len(left_rates) != expected_left:
            raise ValueError(
                f"Expected {expected_left} left rates, got {len(left_rates)}.")
        if len(right_rates) != expected_right:
            raise ValueError(
                f"Expected {expected_right} right rates, got {len(right_rates)}.")

        combined_rates = list(left_rates) + list(right_rates)

        print("\n=== Combined rates ===")
        rxn_idx = 1
        for i in range(0, len(combined_rates), 2):
            label = 'left' if i < expected_left else 'right'
            print(f"  Reaction {rxn_idx:2d} ({label:5s}): "
                  f"k+ = {combined_rates[i]:.4g},  "
                  f"k- = {combined_rates[i+1]:.4g}")
            rxn_idx += 1

        return combined_initial_counts, combined_rates
    

# RunSSA class for running numerical simulations


import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
from sympy import Matrix
from scipy.stats import norm


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
    first_step = True  # <-- debug flag

    if store_trajectories:
        time_history[0] = t
        pop_history[0, :] = pops

    while t < T:

        # 1) Build propensity vector
        propensity_vector = np.zeros(n_reactions)
        for a in range(n_reactions):
            product_of_counts = 1.0
            for idx in current_pops_index[a]:
                product_of_counts *= pops[idx]
            propensity_vector[a] = product_of_counts * rates_list[a]

        # DEBUG: print propensities and indices on first step only
        # if first_step:
        #     print("=== DEBUG: First step propensities ===")
        #     print("Initial pops:", pops)
        #     for a in range(n_reactions):
        #         print("  reaction", a,
        #               "| indices:", current_pops_index[a],
        #               "| propensity:", propensity_vector[a],
        #               "| rate:", rates_list[a])
        #     print("--- Force pairs (l=forward, l+1=backward) ---")
        # l = 0
        # pair_index = 0
        # while l < 2 * stoich_cols:
        #     if propensity_vector[l] > 0.0 and propensity_vector[l + 1] > 0.0:
        #         print("  pair_index", pair_index,
        #                 "| a+ =", propensity_vector[l],
        #                 "| a- =", propensity_vector[l + 1],
        #                 "| log(a+/a-) =", math.log(propensity_vector[l] / propensity_vector[l + 1]))
        #     else:
        #         print("  pair_index", pair_index,
        #                 "| SKIPPED (zero propensity)",
        #                 "| a+ =", propensity_vector[l],
        #                 "| a- =", propensity_vector[l + 1])
        #     l += 2
        #     pair_index += 1
        # print("======================================")
        # first_step = False

        # 2) Sum propensities
        a0 = np.sum(propensity_vector)
        if a0 == 0.0:
            break

        # 3) Generate random numbers and determine which reaction fires
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

        # 4) Determine time spent in this state
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
                if propensity_vector[l] > 0.0 and propensity_vector[l + 1] > 0.0:
                    force_sums[pair_index] += tau_effective * math.log(
                        propensity_vector[l] / propensity_vector[l + 1]
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
    def run_SSA_and_plot_counts(self, store_trajectories=True, plot=True, starting_pops=None):
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

        self.raw_force = force_sums
        self.averaged_forces = (force_sums / steady_time).tolist()

        # print("steady_time:", self.steady_time)
        # print("final_time - burn_in:", self.final_time - self.burn_in)
        # print("ratio:", self.steady_time / (self.final_time - self.burn_in))
        # print("raw force_sums:", self.raw_force)
        # print("averaged_forces:", self.averaged_forces)

        self.average_resistances = []
        for i in range(self.stoich_matrix.cols):
            if (self.averaged_forces[i] != 0 and
                    self.average_reaction_currents[i] != 0):
                self.average_resistances.append(
                    self.averaged_forces[i] / self.average_reaction_currents[i]
                )
            else:
                self.average_resistances.append(np.nan)

        if plot:
            plt.figure(figsize=(8, 5))
            for m in range(self.num_internal_species):
                plt.step(
                    self.time_history,
                    self.pop_history[:, m],
                    where="post",
                    label=self.species_names[m]
                )
            plt.xlabel("$t$")
            plt.ylabel("$N$")
            plt.grid(True)
            plt.subplots_adjust(right=0.75)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.savefig('counts_vs_time_fourmod.png', dpi=300, bbox_inches='tight')

    # ==========================================================
    # PLOT CURRENT AND MEANS COMPARED TO GAUSSIANS
    # ==========================================================

    def plot_gaussian_comparison(
            self,
            bins=50,
            num_iterations=500,
            Gaussian_points=1000
            ):

        plt.rcParams.update({'font.size': 12})

        currents, forces = self.run_IF_sweep(
            [0],
            np.array([[self.initial_counts[0]]]),
            num_iterations,
            covariance_reaction_indices=None,
            verbose=False
        )

        for j in range(self.stoich_matrix.cols):

            # ── Current plot ──────────────────────────────────────────────
            fig, ax = plt.subplots()

            current_vals = [cur[j] for cur in currents]
            mean_I    = np.mean(current_vals)
            std_dev_I = np.std(current_vals, ddof=1)

            ax.hist(current_vals, bins, edgecolor='black', density=True,
                    color='steelblue', alpha=0.7, label=r'SSA samples')

            x_I = np.linspace(min(current_vals), max(current_vals), Gaussian_points)
            y_I = norm.pdf(x_I, mean_I, std_dev_I)
            ax.plot(x_I, y_I, color='red', linewidth=1.5, label='Gaussian fit')

            hist_counts, bin_edges = np.histogram(current_vals, bins=bins, density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            y_fit = norm.pdf(bin_centres, mean_I, std_dev_I)
            ss_res = np.sum((hist_counts - y_fit)**2)
            ss_tot = np.sum((hist_counts - np.mean(hist_counts))**2)
            r2 = 1 - ss_res / ss_tot

            ax.text(0.05, 0.95,
                    f'$\\mu = {mean_I:.3f}$\n$\\sigma = {std_dev_I:.3f}$\n$R^2 = {r2:.4f}$',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel(r'$\langle j_{' + str(j+1) + r'} \rangle$')
            ax.set_ylabel('Density')
            ax.set_title(f'$K = {num_iterations}$ SSA runs, reaction {j+1}')
            ax.legend(loc='upper right')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'gaussian_current_rxn{j+1}.png', dpi=300, bbox_inches='tight')
            plt.show()

            # ── Force plot ────────────────────────────────────────────────
            fig, ax = plt.subplots()

            force_vals = [force[j] for force in forces]
            mean_F    = np.mean(force_vals)
            std_dev_F = np.std(force_vals, ddof=1)

            ax.hist(force_vals, bins, edgecolor='black', density=True,
                    color='forestgreen', alpha=0.7, label='SSA samples')

            x_F = np.linspace(min(force_vals), max(force_vals), Gaussian_points)
            y_F = norm.pdf(x_F, mean_F, std_dev_F)
            ax.plot(x_F, y_F, color='red', linewidth=1.5, label='Gaussian fit')

            hist_counts, bin_edges = np.histogram(force_vals, bins=bins, density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            y_fit = norm.pdf(bin_centres, mean_F, std_dev_F)
            ss_res = np.sum((hist_counts - y_fit)**2)
            ss_tot = np.sum((hist_counts - np.mean(hist_counts))**2)
            r2 = 1 - ss_res / ss_tot

            ax.text(0.05, 0.95,
                    f'$\\mu = {mean_F:.3f}$\n$\\sigma = {std_dev_F:.3f}$\n$R^2 = {r2:.4f}$',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel(r'$f_{' + str(j+1) + r'} = \log{{a_{f}/a_{b}}}$')
            ax.set_ylabel('Density')
            ax.set_title(f'$K = {num_iterations}$ SSA runs, reaction {j+1}')
            ax.legend(loc='upper right')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'gaussian_force_rxn{j+1}.png', dpi=300, bbox_inches='tight')
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

        if isinstance(species_index, int):
            self.species_index = [species_index]
            count_values = [count_values]
        self.species_index = species_index
        self.species_indexes = self.species_index
        self.count_values = count_values

        count_values = [np.asarray(cv, dtype=np.float64) for cv in count_values]

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

                fresh_pops = list(self.initial_counts)
                for idx, cv in zip(self.species_index, count_values):
                    fresh_pops[idx] = float(cv[s])

                self.run_SSA_and_plot_counts(
                    store_trajectories=False,plot=False,
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
                    for idx, cv in zip(self.species_index, count_values)
                )
                print(f"Sweep {s + 1}/{n_sweeps}  ({varied_str})")

        t_end = time.time()
        if verbose:
            print(f"Total sweep time: {t_end - t_start:.2f} s")

        self.sweep_I_means                     = I_means
        self.sweep_F_means                     = F_means
        self.sweep_I_variances                 = I_vars
        self.sweep_F_variances                 = F_vars
        self.sweep_covariance_matrices         = cov_matrices
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

        if colour_by_count:

                while True:

                    try:
                        user_selected_xaxis = int(input("Enter index of varied species for colour-grading (str) : "))
                        if 0 <= user_selected_xaxis < len(self.count_values):
                            break
                        else:
                            print(f"Please enter an index between 0 and {len(self.count_values)-1}.")
                    except ValueError:
                        print("Invalid input. Please enter an integer.")

                c_vals = self.count_values[user_selected_xaxis]
                
                potential_col_bar_axis_label = []

                for index in self.species_index:

                    potential_col_bar_axis_label.append(self.species_names[index])

                col_bar_axis_label = potential_col_bar_axis_label[user_selected_xaxis]


        for r in reaction_indices:
            fig, ax = plt.subplots(figsize=(8, 8))

            F_vals = self.sweep_F_means[:, r]
            I_vals = self.sweep_I_means[:, r]

            

            if show_errorbars:
                F_err = np.sqrt(self.sweep_F_variances[:, r])
                I_err = np.sqrt(self.sweep_I_variances[:, r])
                ax.errorbar(
                    F_vals, I_vals,
                    xerr=F_err, yerr=I_err,
                    fmt='none', ecolor='grey', alpha=0.75,
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
                # species_label = self.species_names[self.species_indexes[0]]
                cbar.set_label(f'Initial {col_bar_axis_label} count')
            else:
                ax.scatter(F_vals, I_vals, s=marker_size, label='SSA', zorder=3)

            if analytical_currents and  analytical_forces:
                ax.scatter(
                    analytical_forces[r], analytical_currents[r],
                    label='Analytical', marker='x', c='orange',
                    s=marker_size, zorder=4
                )
                ax.legend()

            ax.set_xlabel("Average Force")
            ax.set_ylabel("Average Current")
            ax.set_title(f"I-F Curve: Reaction {r + 1}")
            ax.grid(True)
            fig.tight_layout()
            plt.legend()
            
            plt.savefig('I-F_comparison_plot.png', dpi=300, bbox_inches='tight')
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

        n_sweeps = len(self.count_values[0]) # all must be the same length anyway
        n_rxn = self.n_reactions

        

        C = self.module.calculate_reaction_cycle_matrix()
        S_ext = self.module.external_stoich_matrix
        L = self.module.calculate_selection_matrix()

        # Create objects: SM_externals * cycle_matrix, p    seudoinverse of selection matrix
        S_ext_C = S_ext * C
        L_pinv = L.pinv()

        n_indep = L_pinv.rows

        G_fundamental_list = []
        G_physical_list = []
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
            G_physical_list.append(G_phys)
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

        self.sweep_G_physical = G_physical_list
        
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

# COMPUTE CONDUCTANCES FOR CASE OF USING A COMBININGMODULES 
    def compute_conductances2(self, analytical_currents=None, analytical_forces=None):
        if not hasattr(self, 'sweep_I_means'):
            raise RuntimeError("No sweep data found. Call run_IF_sweep() first.")

        n_sweeps = len(self.count_values[0])
        n_rxn = self.n_reactions

        # ── Resolve S_ext ────────────────────────────────────────────────────────
        if hasattr(self.module, "external_stoich_matrix") and self.module.external_stoich_matrix is not None:
            S_ext = self.module.external_stoich_matrix
        else:
            S_ext = self.stoich_matrix[self.num_internal_species:, :]

        # ── Resolve internal SM ──────────────────────────────────────────────────
        internal_SM = self.stoich_matrix[:self.num_internal_species, :]

        # ── Resolve cycle matrix C ───────────────────────────────────────────────
        if hasattr(self.module, "calculate_reaction_cycle_matrix"):
            C = self.module.calculate_reaction_cycle_matrix()
        else:
            reaction_cycles = internal_SM.nullspace()
            if not reaction_cycles:
                raise RuntimeError("No internal cycles. Kernel is empty.")
            C = reaction_cycles[0]
            for cycle in reaction_cycles[1:]:
                C = C.row_join(cycle)

        # ── Resolve selection matrix L ───────────────────────────────────────────
        if hasattr(self.module, "selection_matrix") and self.module.selection_matrix is not None:
            L = self.module.selection_matrix
        else:
            phi = S_ext * C
            cokernel_coupling = phi.T.nullspace()
            if not cokernel_coupling:
                raise RuntimeError("No chemostat conservation laws.")
            chemostat_laws = cokernel_coupling[0]
            for law in cokernel_coupling[1:]:
                chemostat_laws = chemostat_laws.row_join(law)
            chemostat_laws = chemostat_laws.T
            null_basis = chemostat_laws.nullspace()
            L = sympy.Matrix.hstack(*null_basis) if null_basis else sympy.Matrix([])

        # ── Resolve coupling matrix for F_map/I_map ──────────────────────────────
        if hasattr(self.module, "coupling_matrix") and self.module.coupling_matrix is not None:
            coupling_matrix = self.module.coupling_matrix
        else:
            coupling_matrix = S_ext * C

        # ── Precompute shared objects ─────────────────────────────────────────────
        S_ext_C = S_ext * C
        L_pinv  = L.pinv()
        n_indep = L_pinv.rows

        F_map = -L.T * coupling_matrix.T.pinv() * C.T
        I_map = -L_pinv * S_ext

        G_fundamental_list = []
        G_eigenvalue_list  = []
        G_scalar_list      = []

        self.fundamental_forces   = []
        self.fundamental_currents = []
        self.fundamental_EPRs     = []

        for s in range(n_sweeps):

            # ── Resistances ──────────────────────────────────────────────────────
            resistances = []
            skip = False
            for r in range(n_rxn):
                F_r = self.sweep_F_means[s, r]
                I_r = self.sweep_I_means[s, r]
                if I_r != 0.0 and F_r != 0.0 and not np.isnan(F_r) and not np.isnan(I_r):
                    resistances.append(float(F_r / I_r))
                else:
                    skip = True
                    break

            if skip:
                G_fundamental_list.append(None)
                G_scalar_list.append(float('nan'))
                G_eigenvalue_list.append([float('nan')] * n_indep)
                continue

            # ── Conductance matrix ───────────────────────────────────────────────
            R_diag = Matrix.zeros(n_rxn, n_rxn)
            for r in range(n_rxn):
                R_diag[r, r] = resistances[r]

            try:
                G_cycle = (C.T * R_diag * C).inv()
            except Exception as e:
                print(f"Warning: Could not invert at sweep {s}: {e}")
                G_fundamental_list.append(None)
                G_scalar_list.append(float('nan'))
                G_eigenvalue_list.append([float('nan')] * n_indep)
                continue

            G_phys = S_ext_C * G_cycle * S_ext_C.T
            G_fund = L_pinv * G_phys * L_pinv.T
            G_fundamental_list.append(G_fund)

            if G_fund.shape == (1, 1):
                G_scalar_list.append(float(G_fund[0, 0]))
                G_eigenvalue_list.append([float(G_fund[0, 0])])
            else:
                G_fund_np = np.array(G_fund.tolist(), dtype=np.float64)
                eigvals = np.sort(np.linalg.eigvalsh(G_fund_np))
                G_eigenvalue_list.append(eigvals.tolist())
                G_scalar_list.append(float('nan'))

            # ── Fundamental forces, currents, EPR ────────────────────────────────
            reshape_f = Matrix(self.sweep_F_means[s].tolist()).reshape(n_rxn, 1)
            reshape_i = Matrix(self.sweep_I_means[s].tolist()).reshape(n_rxn, 1)

            fund_force   = F_map * reshape_f
            fund_current = I_map * reshape_i

            self.fundamental_forces.append(fund_force)
            self.fundamental_currents.append(fund_current)
            self.fundamental_EPRs.append(float((fund_force.T * fund_current)[0, 0]))

        self.fundamental_EPRs   = np.array(self.fundamental_EPRs, dtype=float)
        self.fundamental_forces = np.array(self.fundamental_forces, dtype=float)

        valid_G    = [G for G in G_fundamental_list if G is not None]
        is_scalar  = all(G.shape == (1, 1) for G in valid_G) if valid_G else True

        self.conductance_type     = 'scalar' if is_scalar else 'matrix'
        self.sweep_G_fundamental  = G_fundamental_list
        self.sweep_G_scalar       = np.array(G_scalar_list)
        self.sweep_G_eigenvalues  = np.array(G_eigenvalue_list)

        # ── Analytical conductances (optional) ───────────────────────────────────
        if analytical_currents is not None and analytical_forces is not None:
            analytical_G = []
            for s in range(n_sweeps):
                resistances_analytical = []
                skip_a = False
                for r in range(n_rxn):
                    if r in analytical_currents and r in analytical_forces:
                        I_a, F_a = analytical_currents[r][s], analytical_forces[r][s]
                        if I_a != 0 and F_a != 0:
                            resistances_analytical.append(float(F_a / I_a))
                        else:
                            skip_a = True; break
                    else:
                        F_r, I_r = self.sweep_F_means[s, r], self.sweep_I_means[s, r]
                        if I_r != 0.0 and F_r != 0.0:
                            resistances_analytical.append(float(F_r / I_r))
                        else:
                            skip_a = True; break

                if skip_a:
                    analytical_G.append(float('nan'))
                    continue

                R_diag_a = Matrix.zeros(n_rxn, n_rxn)
                for r in range(n_rxn):
                    R_diag_a[r, r] = resistances_analytical[r]
                try:
                    G_cycle_a = (C.T * R_diag_a * C).inv()
                    G_fund_a  = L_pinv * S_ext_C * G_cycle_a * S_ext_C.T * L_pinv.T
                    if G_fund_a.shape == (1, 1):
                        analytical_G.append(float(G_fund_a[0, 0]))
                    else:
                        analytical_G.append(np.sort(np.linalg.eigvalsh(
                            np.array(G_fund_a.tolist(), dtype=np.float64)
                        )).tolist())
                except Exception:
                    analytical_G.append(float('nan'))

            self.analytical_G = analytical_G
            return G_fundamental_list, analytical_G

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

        # counts = self.sweep_count_values
        n_sweeps = len(self.count_values[0])

        while True:

            try:
                user_selected_xaxis = int(input("Enter index of species for colour-grading (int): "))
                if 0 <= user_selected_xaxis < len(self.count_values):
                    break
                else:
                    print(f"Please enter an index between 0 and {len(self.count_values)-1}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        x_axis_for_plot = self.count_values[user_selected_xaxis]

        potential_x_axis_label = []

        for index in self.species_index:

            potential_x_axis_label.append(self.species_names[index])

        x_axis_label = potential_x_axis_label[user_selected_xaxis]

        # =======================
        # SCALAR CONDUCTANCE
        # =======================

        if self.conductance_type == 'scalar':

            fig, ax = plt.subplots(figsize=(10, 7))

            # G coloured by EPR 
            sc = ax.scatter(
                x_axis_for_plot, self.sweep_G_scalar,
                c=self.fundamental_EPRs, cmap=cmap,
                s=marker_size, edgecolors='black', linewidths=0.5,
                label='$G$ (SSA)', zorder=3
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(r'$\dot{\sigma} = A^T I$')

            if analytical_G is not None:
                ax.scatter(
                    x_axis_for_plot, analytical_G,
                    marker='x', c='red', s=marker_size,
                    label='$G$ (User Data)', zorder=4
                )

            if show_covariance:
                cov_half = np.array([
                    self.sweep_covariance_matrices[s][0, 0] / 2.0
                    for s in range(n_sweeps)
                ])

                ax.scatter(
                    x_axis_for_plot, cov_half,
                    marker='x', color='blue', s=marker_size,
                    label=r'$\mathrm{Cov}(I) / 2$  (SSA)',
                    zorder=3
                )

                # fitted plot for covariance
                valid = ~np.isnan(cov_half) & ~np.isnan(x_axis_for_plot)
                if np.sum(valid) > fit_order + 1:
                    coeffs = np.polyfit(x_axis_for_plot[valid], cov_half[valid], fit_order)
                    x_fit = np.linspace(
                        np.min(x_axis_for_plot[valid]), np.max(x_axis_for_plot[valid]), 200
                    )
                    y_fit = np.polyval(coeffs, x_fit)
                    ax.plot(
                        x_fit, y_fit,
                        linestyle='--', color='blue', alpha=0.6
                    )
                

            if show_difference and show_covariance:

                print("No difference plotted: Option Deprecated")
                

            ax.set_xlabel(f'Initial count of {x_axis_label}')
            ax.set_ylabel('$G$')
            # ax.set_title(r'$G$ vs varied species, colour graded against fundamental EPR')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            plt.show()

   

        else:
            eigvals = np.abs(self.sweep_G_eigenvalues)
            n_indep = eigvals.shape[1]

            fig, ax = plt.subplots(figsize=(10, 7))

            # # Create a ScalarMappable for the colorbar
            # norm = mpl.colors.Normalize(
            #     vmin=np.min(self.fundamental_EPRs),
            #     vmax=np.max(self.fundamental_EPRs)
            # )
            # sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            # sm.set_array([])  # Required for colorbar

            # Scatter SSA eigenvalues with color mapped to fundamental_EPRs
            for e in range(n_indep):
                scatter = ax.scatter(
                    x_axis_for_plot, eigvals[:, e],
                    # c=self.fundamental_EPRs,
                    # cmap=cmap,
                    # norm=norm,  # Use the same norm as the colorbar
                    s=marker_size,
                    edgecolors='black', linewidths=0.5,
                    label=r'$\lambda_{D}$',
                    zorder=3, color = 'blue'
                )
            

            # Scatter analytical G if provided
            if analytical_G is not None:
                analytical_G_np = np.abs(np.array(analytical_G))
                if analytical_G_np.ndim == 1:
                    ax.scatter(
                        x_axis_for_plot, analytical_G_np,
                        marker='*', c='yellow', s=marker_size*2,
                        label=r'$\lambda_{C}$', zorder=4, edgecolors='black', linewidths=0.5
                    )
                else:
                    for e in range(analytical_G_np.shape[1]):
                        ax.scatter(
                            x_axis_for_plot, analytical_G_np[:, e],
                            marker='*', color='yellow', s=marker_size*2,
                            label=r'$\lambda_{C}$', zorder=4, edgecolors='black', linewidths=0.5
                        )

            # Scatter covariance norms if needed
            if show_covariance:
                cov_spectral_norms = np.zeros(n_sweeps)
                for s in range(n_sweeps):
                    cov_half = 0.5 * self.sweep_covariance_matrices[s]
                    cov_spectral_norms[s] = np.linalg.norm(cov_half, 2)

                ax.scatter(
                    x_axis_for_plot, cov_spectral_norms,
                    marker='x', color='blue', s=marker_size * 0.8,
                    label=r'$\| \mathrm{Cov}(I^{(3)})/2 \|$',
                    zorder=3
                )

            # Create the colorbar
            # cbar = fig.colorbar(sm, ax=ax)
            # cbar.set_label(r'$\dot{\sigma} = A^T I$')

            if show_difference:

                print("No difference plotted: Option Deprecated.")

                
                
                

            ax.set_xlabel(r'$[Nd_0]$')#{x_axis_label}')
            ax.set_ylabel(r'$\lambda({\mathbf{G}})$')
            # ax.set_title('Conductance Matrix Eigenvalues vs Initial Count')
            
            ax.legend(loc='center left')
            ax.grid(True)
            fig.tight_layout()
            plt.savefig('conductances_comparison_mod4.png', dpi=300, bbox_inches='tight')
            plt.show()