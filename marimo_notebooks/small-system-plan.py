import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Finding microscopic conductances from macro covariance matrices

    - Take an easy graph to study (two cycles?)
    - Calculate covariance $C$ at $N$ operating points , it's a function of $A$ (the forces). Random sample A around 0 (0 is the equilibrium). This can be sampled directly (via Gillespie simulations) or exactly expressed using cumulant generating function techniques (ask Paul & Gatien). Do **Gillespie**, this will require choosing reservoir concentration and hence setting the units. $C$ is numerical. So we have a set $\{ C_p\}$ of N matrices, and a set of vectors $\{\vec{A}_p\}$ (we'll try and stay in $\mathbb{R}^d$ with $d=2$ with our choice of the graph). $d$ is predictable from the stoichiometry matrix. We also measure the time-averaged current, and we have a set $\{ \vec{I}_p \}$ where $\vec{I}$ is the independent current vectors of dimension d.
    - we get from this the entropy production $\{\vec{A}^T \vec{I}\}$, it is just a scalar $\sigma_p$ (as expected) for every $p$ (again, we will have N of them) (it is $\sim$ the power of the circuit)

    - Express the conductance matrix $G$ in terms of the microscopic currents $r$ by solely using the topology of the graph

        - step 0: the cycle matrix is $\mathcal{C}$ from the fact that $\mathcal{C}$ has columns that form the $\operatorname{ker}\left(\boldsymbol{\nabla}_{\mathrm{x}}\right)$,
        - step1 : $\mathbf{g}=\boldsymbol{\nabla}_{\mathrm{Y}} \mathcal{C C}^{+} \boldsymbol{r}^{-1}\left(\mathcal{C C}^{+}\right)^T \boldsymbol{\nabla}_{\mathrm{Y}}^T$
        - step2 : $\mathbf{\ell} \mathbf{\phi}=0$ (both matrices) , where $\mathbf{\phi} = \mathbf{\nabla}_Y C$, this define $\mathbf{\ell}$ and the baiss of the kernel of $\ell$ is $S$
        - and from $S$ one gets $G$ as $\boldsymbol{G} \equiv \boldsymbol{S}^{+} \boldsymbol{g} \boldsymbol{S}^{T+}$
        - at the end (somehow, we need to find exactly how) we get $G(r)$, in general nonlinear in r (if linear, maybe this becomes a linear programming exercise)

    - for every $p$ we search $r_p$ to estimate $G(r_p)$:
        - The solution $r_p$ sits on a specific manifold where $G\vec{A}=\vec{I}$, which is a system of $d$ simultaneous equations (possibly nonlinear)
        - (I could multiply by $A^T$ means that $A_p^TG(r_p)A_p=\sigma_p$ . This is just one equation in $r_1, r_2, \dots$)
        - a suitable matrix norm $||G(r)-C/2||$ is as small as possible
    - The final G should be compared with

        - exact results (when possible)
        - direct simulation (Gillespie)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The minimisation problem should technically be trivial to implement numerically via a generalisation of Lagrange multipliers.

    Given:

    $$
    \min_x f(x) \quad \text{subject to} \quad h(x) = 0
    $$

    Form the **Lagrangian**:

    $$
    \mathcal{L}(x, \lambda) = f(x) + \lambda h(x)
    $$

    Then solve:

    $$
    \nabla_x \mathcal{L} = 0, \quad h(x) = 0
    $$


    Using Karush-Kuhn-Tucker (KKT) conditions, i.e.

    $$
    \min_x f(x) \quad \text{subject to} \quad h(x) = 0, \quad g(x) \leq 0
    $$

    Form the **Lagrangian**:

    $$
    \mathcal{L}(x, \lambda, \mu) = f(x) + \lambda h(x) + \mu g(x)
    $$

    Satisfy the following conditions

    - Stationarity:   $\nabla_x \mathcal{L} = 0$
    - Primal feasibility: $h(x) = 0,\ g(x) \leq 0$
    - Dual feasibility:  $\mu \geq 0$
    - Complementary slackness: $\mu g(x) = 0$
    """
    )
    return


@app.cell
def _():
    # example 

    import numpy as np
    from scipy.optimize import minimize

    # quantity to minimsie, e.g. our norm
    def f(x,c=np.array([0.2, 0.2])):
        # and arbitrary nonlinear function
        return np.linalg.norm(c-x)

    # Equality constraint
    def h(x, offset = 1.0):
        return x[0] - x[1] + offset

    # Inequality constraint
    def g(x):
        # only one component is negative
        return - x[0]

    # Initial guess
    x0 = [0, 0]

    # Constraints dictionary
    constraints = [
        {'type': 'eq', 'fun': h},
        {'type': 'ineq', 'fun': g}
    ]

    # Solve
    result = minimize(f, x0, constraints=constraints)

    # Output
    print("Optimal x:", result.x)
    print("Objective value:", result.fun)

    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
