# Table to convert from string to code
# Will be called in setparam

# Example mapper (Python-level only):
def _to_bkg_code(bkg):
    # Map user-friendly strings -> tiny ints; Numba never sees strings.
    table = {"None": -1, "iso": 0, "nfw": 1}
    if isinstance(bkg, str):
        return np.int32(table[bkg])
    # If caller passes an int already:
    return np.int32(bkg)

# For hydrostatic:
# --- tiny Python dispatcher (not jitted) ---
def compute_he_resid_norm(r, rho, p, m, bkg=None):
    """
    bkg can be:
      - None
      - a string like "iso", "nfw", etc. (converted to code here)
      - or already an int code / parameter array
    """
    if bkg is None:
        return _compute_he_resid_norm_no_bkg(r, rho, p, m)
    # convert strings to a small int code (or a small parameter array)
    bkg_code = _to_bkg_code(bkg)  # pure Python; returns int32 or small tuple/array
    return _compute_he_resid_norm_with_bkg(r, rho, p, m, bkg_code)

# Hernquist profile