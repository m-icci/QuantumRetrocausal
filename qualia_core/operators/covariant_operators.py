import numpy as np

def discretized_covariant_laplacian_1d_radial(field_1d, r_grid, A_r, boundary_condition):
    """Laplaciano covariante discreto em 1D com Jump Conditions aprimoradas."""
    laplacian_cov_field = np.zeros_like(field_1d)
    dr = r_grid[1] - r_grid[0]

    for i in range(1, len(field_1d) - 1):
        if A_r[i] == 0:
            laplacian_cov_field[i] = 0  # Ou defina um valor padrão
        else:
            laplacian_cov_field_i = (A_r[i+1] * (field_1d[i+1] - field_1d[i]) -
                                      A_r[i] * (field_1d[i] - field_1d[i-1])) / (A_r[i] * dr ** 2)
            laplacian_cov_field[i] = laplacian_cov_field_i if not np.isnan(laplacian_cov_field_i) else 0

    if boundary_condition == 'Jump':
        jump_index = np.argmax(np.abs(np.diff(field_1d)))  # Localiza a maior variação
        q = np.abs(field_1d[jump_index + 1] - field_1d[jump_index - 1])  # Fluxo no ponto de salto

        if A_r[jump_index] != 0:
            laplacian_cov_field[jump_index] = q / (A_r[jump_index] * dr ** 2)
        else:
            laplacian_cov_field[jump_index] = 0  # Ou defina um valor padrão

        laplacian_cov_field[jump_index - 1] = (field_1d[jump_index] - field_1d[jump_index - 1]) / (dr ** 2)
        laplacian_cov_field[jump_index + 1] = (field_1d[jump_index + 1] - field_1d[jump_index]) / (dr ** 2)

    return laplacian_cov_field

def test_boundary_conditions(field, r_grid, boundary_condition):
    """Testa a implementação de condições de contorno aplicando diferenças finitas."""
    laplacian = np.zeros_like(field)
    if boundary_condition == 'Dirichlet':
        laplacian[0] = 0
        laplacian[-1] = 0

    elif boundary_condition == 'Neumann':
        PHI = (1 + np.sqrt(5)) / 2
        dr = r_grid[1] - r_grid[0]
        laplacian[0] = PHI * (field[1] - field[0]) / (dr**2)
        laplacian[-1] = -PHI * (field[-1] - field[-2]) / (dr**2)

    else:
        raise ValueError("Invalid boundary condition. Choose 'Dirichlet' or 'Neumann'.")

    return laplacian

def apply_folding_covariant_1d_radial(field_1d, r_grid, A_r, alpha=0.1):
    """
    Aplica o operador de Folding covariante 1D radial.

    Args:
        field_1d (np.ndarray): Campo 1D radial.
        r_grid (np.ndarray): Grade radial.
        A_r (np.ndarray): Valor de A(r) = 1 - b(r)/r.
        alpha (float): Força do operador de folding covariante.

    Returns:
        np.ndarray: Campo 1D radial modificado pelo operador de folding covariante.
    """
    laplacian_cov = discretized_covariant_laplacian_1d_radial(field_1d, r_grid, A_r, boundary_condition='Jump')
    return field_1d - alpha * laplacian_cov
