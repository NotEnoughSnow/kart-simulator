

def bin2dec(x, n_int, n_frac):
    int_part, frac_part = x.split(".")
    int_part = int(int_part, 2)
    frac_part = 0
    for i in range(n_frac):
        frac_part += int(x[n_int + 1 + i]) * (2 ** -(i + 1))
    return int_part + frac_part
