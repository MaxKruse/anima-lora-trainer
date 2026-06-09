"""Generate Cartesian product of parameter ranges for matrix training."""

from itertools import product


def generate_permutations(param_ranges: dict) -> list[dict]:
    """Compute Cartesian product of parameter ranges and resolve % values.

    Args:
        param_ranges: Dict mapping param names to lists of values.
                      Values can be int, float, str, or strings ending with '%'.

    Returns:
        List of flat dicts, each representing one permutation with resolved values.

    Raises:
        ValueError: If a % value references a non-numeric parameter.
    """
    if not param_ranges:
        return [{}]

    # Generate Cartesian product
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    permutations = []

    for combination in product(*param_values):
        perm = dict(zip(param_names, combination))
        _resolve_percent_values(perm, param_ranges)
        permutations.append(perm)

    return permutations


def _resolve_percent_values(perm: dict, param_ranges: dict) -> None:
    """Resolve % suffix values in a permutation.

    A value like '25%' means 25% of the corresponding base parameter's value.
    The base parameter is determined by convention: network_alpha % refers to
    network_dim, etc.

    Args:
        perm: The permutation dict to resolve (modified in place).
        param_ranges: Original parameter ranges for reference.
    """
    # Find all keys with % values
    percent_keys = {k for k, v in perm.items() if isinstance(v, str) and v.endswith("%")}

    if not percent_keys:
        return

    # Resolve each % value
    for key in percent_keys:
        percent_str = perm[key]
        percent_value = float(percent_str.rstrip("%"))

        # Determine the base parameter
        base_key = _get_base_param_for(key)
        if base_key not in perm:
            raise ValueError(
                f"Parameter '{key}' has a % value ('{percent_str}') "
                f"but the reference parameter '{base_key}' is not in the permutation"
            )

        base_value = perm[base_key]
        if not isinstance(base_value, (int, float)):
            raise ValueError(
                f"Parameter '{key}' references '{base_key}' for % resolution, "
                f"but '{base_key}' has a non-numeric value: {base_value}"
            )

        perm[key] = base_value * (percent_value / 100.0)


def _get_base_param_for(param_name: str) -> str:
    """Get the base parameter that a % value should reference.

    Convention:
        - network_alpha % → network_dim
        - Any other % → try to find a numeric param with the same prefix

    Args:
        param_name: The parameter name with a % value.

    Returns:
        The name of the base parameter to reference.
    """
    # Known conventions
    conventions = {
        "network_alpha": "network_dim",
        "learning_rate": "network_dim",  # LR as % of dim (unusual but possible)
    }

    if param_name in conventions:
        return conventions[param_name]

    # Default: try to find a numeric param
    return param_name.replace("_alpha", "_dim").replace("_scale", "_dim")
