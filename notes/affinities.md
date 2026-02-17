## Gillespie affinities and forces

You have a vector of populations $\bf{n}$ . This is instantaneous and varies over time stochastically.


The instantaneous propensities depend on the rates and stoichiometry. For a general reaction like
$$
\sum_{i=1}^N \nu_i S_i \underset{k^{-}}{\stackrel{k^{+}}{\rightleftharpoons}} \sum_{i=1}^N \nu_i^{\prime} S_i
$$
Where $S_i$ is the names of species $i$ and $\nu_i$ is the stoichioetry.

You then have that the **forward propensities** are
$$
a^{+}(\mathbf{n})=k^{+} \prod_{i=1}^N\binom{n_i}{\nu_i}
$$
and similarly the **backward propensites** are 
$$
a^{-}(\mathbf{n})=k^{-} \prod_{i=1}^N\binom{n_i}{\nu_i^{\prime}}
$$
You can then get an **instantaneous force** as 
$$
f_i = \ln(a^+/a^-)
$$


And can average over time as usual.

