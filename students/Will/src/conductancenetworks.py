
import hypernetx as hnx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sympy
import warnings 

class StoichMatrix: #FT: Capital letter to define class

    #==========================================================================================================================================
    # INIT
    #

    def __init__(self, SM, num_internal_species):
        """
        HOW DO YOU EXPECT THIS TO WORK?
        INPUTS? OUTPUTS? (Operations that are possible...)

        Attributes: [CHECK THESE]

            - self.matrix : full stoichiometric matrix 
            - self.internal_SM : internal species stoichiometric matrix
            - self.external_SM : external species stoichiometric matrix
            - self.module_1_labels : labels for the species in module 1

        Methods: [CHECK THESE]

            - hypergraph_internals() : creates hypergraph for internal species
            - hypergraph_full() : creates hypergraph for full stoichiometric matrix
            - reaction_cycles_matrix() : finds the reaction level cycles from the internal stoichiometric matrix
            - coupling_matrix() : finds the coupling matrix from the external stoichiometric matrix and the
            - reaction cycles matrix
            - conservation_laws() : finds the conservation laws from the full stoichiometric matrix and


        """

        self.matrix = SM # Returns the passed SM, ready for print
        self.num_internal_species = num_internal_species

        self.internal_SM = SM[0:num_internal_species, :] # finds the internal species SM by selecting the number of rows needed

        self.external_SM = SM[num_internal_species: len(SM), :] # finds SM for external species using remaning rows after internal species

        self.module_1_labels = {0: 'Ea', 1: 'EaS', 2: 'EaS2', 3: 'S', 4: 'Na'} # for the hypergraphs

        # FT: MAYBE WE WANT TO DO
        self.calculate_coupling_matrix()
        self.calculate_conservation_laws()
        
    
    #==========================================================================================================================================
    # HYPERGRAPHS
    #
    # Define a new function to find the hypergraphs for the internal species
    #

    def hypergraph_internals(self):


        internals_HG = hnx.Hypergraph.from_incidence_matrix(self.internal_SM) # create hypergraph, using the internal SM defined in self

        hnx.draw(internals_HG, node_labels=self.module_1_labels, with_edge_labels=True) # print this using the labels defined in self

    #
    # Define a new function to find the hypergraphs for the full stoichiometric matrix
    #

    def hypergraph_full(self):

        full_HG = hnx.Hypergraph.from_incidence_matrix(self.matrix) # create hypergraph, using the full SM defined in self

        hnx.draw(full_HG, node_labels=self.module_1_labels, with_edge_labels=True) # print this using the labels defined in self

    #==========================================================================================================================================
    # REACTION LEVEL CYCLES
    #

    def reaction_cycles_matrix(self):
        
        reaction_cycles = (self.internal_SM).nullspace() # finds the kernel for the SM internal

        # Check if there are any cycles:

        if not reaction_cycles:

            print("No internal cycles. Kernel is empty.")

        # build cycle matrix from kernel vectors if kernel is NOT empty

        else:

            cycle_matrix = reaction_cycles[0] # add first vector to cycle matrix so we can add rest later

            for cycle in reaction_cycles[1:]: # starting at second vector in kernel

                cycle_matrix = cycle_matrix.row_join(cycle) # connect vectors from kernel column-wise, row_join puts elemetns of adjacent vectors together


            return cycle_matrix
        
    #==========================================================================================================================================
    # COUPLING MATRICES
    #  
    
    # you can use the @property decorator to make this a property....
    def calculate_coupling_matrix(self):

        cycle_matrix = self.reaction_cycles_matrix()

        phi = self.external_SM * cycle_matrix
        self.coupling_matrix = phi
        
        return phi


    #==========================================================================================================================================
    # CONSERVATION LAW MATRICES
    #
    def calculate_conservation_laws(self):

        cokernel_SM = (self.matrix.T).nullspace() # finds the cokernel of the full SM

        if not cokernel_SM:

            print("No conservation laws. Cokernel of Stoichiometric Matrix empty.")

        else:

            cons_laws = cokernel_SM[0] # adds first element of cokernel

            for vec in cokernel_SM[1:]: # add vectors from next row onwards

                cons_laws = cons_laws.row_join(vec)


        #
        # Broken external laws for chemostat , deriving from the coupling matrix
        #

        # coupling_matrix = self.calculate_coupling_matrix() # define the coupling matrix using the function defined previously

        cokernel_coupling_matrix = self.coupling_matrix.T.nullspace() # find the cokernel of the coupling matrix

        if not cokernel_coupling_matrix:

            print("No chemostat conservation laws. Cokernel of Coupling Matrix is empty.")

        # if cokernel is NOT empty

        else:

            chemostat_laws = cokernel_coupling_matrix[0] # add first vector to chemostat conservation law matrix so we can add rest later

            for law in cokernel_coupling_matrix[1:]: # starting at second vector in kernel

                chemostat_laws = chemostat_laws.row_join(law) # connect vectors from kernel column-wise, row_join puts elemetns of adjacent vectors together

        self.conservation_laws = cons_laws.T
        self.chemostat_laws = chemostat_laws.T

        # return cons_laws.T, chemostat_laws.T # return transpose to match equations in paper
    
    #==========================================================================================================================================



class Module:

    def __init__(self, SM, num_internal_species):
        self.stoichiometric_matrix = StoichMatrix(SM, num_internal_species)

    def add(self, other, bridge_species):
        print("Serial combination")
        pass
        # return Module(...)
        
        # # matrix = sympy.matrices.dense.matrix_multiply_elementwise(self.stoichiometric_matrix.matrix, other.stoichiometric_matrix.matrix)
        # matrix = self.stoichiometric_matrix.matrix
        # internal = self.stoichiometric_matrix.num_internal_species #+ other.stoichiometric_matrix.num_internal_species
        # return Module(matrix,internal ) #DUMMY OPERATION