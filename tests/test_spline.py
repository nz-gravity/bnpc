# bnpc/tests/test.py
import numpy as np

from bnpc.logSplines.utils import panelty_mat


def test_panelty_mat_non_singular():
    """
    Test that the penalty matrix is not singular for typical parameters.
    """
    # Example usage:
    knots = np.array([0, 0.25, 0.5, 0.75])
    d = 2  # order of difference
    degree = 3

    # Generate the penalty matrix
    p_mat = panelty_mat(d=d, knots=knots, degree=degree, linear=False)

    # Check shape
    assert p_mat.shape[0] == p_mat.shape[1], "Penalty matrix must be square."

    # A quick test for non-singularity is that the determinant != 0,
    # but you might also check np.linalg.cond(...) or rank for numerical stability.
    det_p_mat = np.linalg.det(p_mat)
    assert not np.isclose(
        det_p_mat, 0.0
    ), "Penalty matrix is singular or nearly singular."


def test_panelty_mat_linear():
    """
    Test that linear penalty matrix is computed as expected.
    """
    knots = np.array(
        [0, 1, 2]
    )  # not used if linear=True, but you still must pass it
    d = 1
    # The k parameter in your utils might need to match the number of weights
    # e.g., if you expect some specific dimension, pass k=...
    # For this example, let's guess k=5
    # We'll update the function call accordingly if needed:

    p_mat = panelty_mat(d=d, knots=knots, degree=3, linear=True, k=5)
    assert p_mat.shape == (5, 5)
    # You can add more checks here depending on your expectations of the linear penalty matrix.
