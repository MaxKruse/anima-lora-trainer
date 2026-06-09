"""Parse comma-separated parameter ranges for matrix training."""


def parse_param_range(value_str: str) -> list:
    """Parse a comma-separated string of parameter values.

    Args:
        value_str: Comma-separated values like "1,2,3" or "1e-4,5e-4" or "AdamW8Bit,Prodigy"

    Returns:
        List of parsed values. Integers are returned as int, floats as float,
        strings with non-numeric characters as str. Values with % suffix are
        preserved as strings for later resolution.

    Raises:
        ValueError: If the input string is empty or whitespace-only.
    """
    if not value_str or not value_str.strip():
        raise ValueError("Parameter range cannot be empty")

    parts = [p.strip() for p in value_str.split(",")]
    result = []

    for part in parts:
        if not part:
            continue

        # Preserve % markers as strings
        if part.endswith("%"):
            result.append(part)
            continue

        # Try integer first
        try:
            int_val = int(part)
            # Only use int if the string representation matches (avoid "1.0" → 1)
            if str(int_val) == part:
                result.append(int_val)
                continue
        except ValueError:
            pass

        # Try float
        try:
            float_val = float(part)
            result.append(float_val)
            continue
        except ValueError:
            pass

        # Keep as string
        result.append(part)

    return result
