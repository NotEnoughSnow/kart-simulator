# Define a custom function to convert a decimal number to a binary string
def dec2bin(x, n_int, n_frac):
    # Convert the integer part to binary
    int_part = bin(int(x))[2:]
    # Pad with zeros if necessary
    int_part = int_part.zfill(n_int)
    # Convert the fractional part to binary
    frac_part = ""
    temp = x - int(x)
    for i in range(n_frac):
        temp *= 2
        frac_part += str(int(temp))
        temp -= int(temp)
    # Concatenate the integer and fractional parts
    return int_part + "." + frac_part


# Define a custom function to convert a binary string to a decimal number
def bin2dec(x, n_int, n_frac):
    # Split the binary string into integer and fractional parts
    int_part, frac_part = x.split(".")
    # Convert the integer part to decimal
    int_part = int(int_part, 2)
    # Convert the fractional part to decimal
    frac_part = 0
    for i in range(n_frac):
        frac_part += int(x[n_int + 1 + i]) * (2 ** -(i + 1))
    # Add the integer and fractional parts
    return int_part + frac_part
