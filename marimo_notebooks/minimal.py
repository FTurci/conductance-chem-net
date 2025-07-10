import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Minimal chemical reaction network to check nonlinear solver vs analytical solution""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We construct a simple reaction network that has serial modules and test that their numerical solution matches the analytical one.

    ### TODOs:

    - Wrap code in self-contained Classes
    - generalise
    - Solve by module and use Gatien Verley's modular approach
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import hypernetx as hnx
    import matplotlib.pyplot as plt

    # Combine reactants and products into one set per reaction (undirected)
    edges = {
        'R1': {'A', 'X'},
        'R2': {'X', 'Y'},
        'R3': {'Y', 'B'},
    }

    H = hnx.Hypergraph(edges)

    plt.figure(figsize=(8,4))
    hnx.drawing.draw(H, with_edge_labels=True)
    plt.title("Undirected Hypergraph of A ⇌ X ⇌ Y ⇌ B")
    plt.axis('off')
    plt.show()

    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    from scipy.optimize import fsolve
    import sympy as sp

    # ---- Parameters ----
    k1, km1 = 1.0, 0.5    # A ⇌ X
    k2, km2 = 0.8, 0.4    # X ⇌ Y
    k3, km3 = 0.6, 0.3    # Y ⇌ B

    A_ext = 1.0
    B_ext = 0.2

    # ---- Analytical Solver (Symbolic) ----
    x, y, J = sp.symbols('x y J')

    # Mass-action rates
    v1 = k1 * A_ext - km1 * x
    v2 = k2 * x - km2 * y
    v3 = k3 * y - km3 * B_ext

    # Steady-state: v1 = v2 = v3 = J
    eq1 = sp.Eq(v1, J)
    eq2 = sp.Eq(v2, J)
    eq3 = sp.Eq(v3, J)

    # Solve system
    sol = sp.solve((eq1, eq2, eq3), (x, y, J), dict=True)[0]
    x_analytical = float(sol[x])
    y_analytical = float(sol[y])
    J_analytical = float(sol[J])

    # ---- Modular Solver ----
    # Module 1: A ⇌ X ⇌ Y
    # Module 2: Y ⇌ B

    def module1_flux(y):
        # Solve for x given y
        def flux_residual(x):
            v1 = k1 * A_ext - km1 * x
            v2 = k2 * x - km2 * y
            return v1 - v2
        x_sol = fsolve(flux_residual, x0=0.1)[0]
        v1_val = k1 * A_ext - km1 * x_sol
        return v1_val, x_sol

    def module2_flux(y):
        return k3 * y - km3 * B_ext

    # Interface condition: fluxes match at Y
    def interface_residual(y):
        f1, _ = module1_flux(y)
        f2 = module2_flux(y)
        return f1 - f2

    # Solve for interface y
    y_modular = fsolve(interface_residual, x0=0.1)[0]
    J_modular, x_modular = module1_flux(y_modular)

    # ---- Results ----
    results = {
        "Analytical J": J_analytical,
        "Modular J": J_modular,
        "Analytical x": x_analytical,
        "Modular x": x_modular,
        "Analytical y": y_analytical,
        "Modular y": y_modular,
    }
    results

    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
