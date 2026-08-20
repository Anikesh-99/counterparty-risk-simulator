"""Correlated Brownian shock generator.

All randomness in the engine enters here. The driver draws independent standard
normals for every (path, step, factor) and applies the Cholesky factor of the
correlation matrix, so correlation is handled in exactly one place and the
individual risk-factor models stay pure "given these shocks, evolve" functions.
"""

from __future__ import annotations

import numpy as np


class CorrelatedDriver:
    """Generates correlated N(0,1) shocks of shape ``(n_paths, n_steps, n_factors)``.

    Parameters
    ----------
    corr:
        Symmetric positive-semidefinite correlation matrix, shape (n_factors, n_factors).
    seed:
        Seed for the underlying ``numpy`` random generator (reproducibility).
    """

    def __init__(self, corr: np.ndarray, seed: int | None = None) -> None:
        corr = np.asarray(corr, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if not np.allclose(corr, corr.T, atol=1e-12):
            raise ValueError("Correlation matrix must be symmetric.")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("Correlation matrix must have unit diagonal.")
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < -1e-8:
            raise ValueError("Correlation matrix is not positive semidefinite.")
        self.corr = corr
        self.n_factors = corr.shape[0]
        # Nudge tiny negative eigenvalues before Cholesky for numerical safety.
        self._chol = np.linalg.cholesky(corr + np.eye(self.n_factors) * 1e-12)
        self._rng = np.random.default_rng(seed)

    def draw(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Return correlated shocks, shape ``(n_paths, n_steps, n_factors)``."""
        z = self._rng.standard_normal((n_paths, n_steps, self.n_factors))
        # Apply Cholesky along the factor axis: corr_shocks = z @ L^T.
        return z @ self._chol.T
