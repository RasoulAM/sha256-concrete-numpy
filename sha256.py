# Required libraries
import concrete.numpy as cnp
import numpy as np

# Bitwidth of each slice and number of slices in each 32-bit number.
WIDTH=4
NUM_SLICES=8

assert (WIDTH * NUM_SLICES == 32)

def split_to_slices(x):
    return np.array([(x//2**(i*WIDTH)) % (2**WIDTH) for i in range(NUM_SLICES-1, -1, -1)])

def slices_to_uint32(slices):
    return sum([2**((NUM_SLICES-1-i)*WIDTH)*x for i, x in enumerate(slices)])

def slices_to_hexarray(slices):
    hexes = [hex(slices_to_uint32(word))[2:] for word in slices]
    hexes = ['0'*(8-len(y))+y for y in hexes] #Appending leadning zero to the ones that are less than 8 characters TODO: write better
    result = "".join(hexes)
    return result

# Extracts the [shift] lower bits of a [WIDTH]-bit number and places them at the top
table_extract_low_bits_and_raise = {
    shift : cnp.LookupTable([2**(WIDTH-shift) * (x %2**shift) for x in range(2**WIDTH)])
    for shift in range(WIDTH)
}

# Extracts the [WIDTH-shift] higher bits of a [WIDTH]-bit number
table_extract_high_bits = {
    shift : cnp.LookupTable([x//2**shift  for x in range(2**WIDTH)])
    for shift in range(WIDTH)
}

def right_rotate_list_of_slices(list_to_rotate, amount):
    xx = [x for x in list_to_rotate[-amount:]] + [x for x in list_to_rotate[:-amount]]
    return cnp.array(xx)

def right_shift_list_of_slices(list_to_rotate, amount):
    xx = [x-x for x in list_to_rotate[-amount:]] + [x for x in list_to_rotate[:-amount]]

    return cnp.array(xx)

def left_rotate_list_of_slices(list_to_rotate, amount):
    xx = [x for x in list_to_rotate[amount:]] + [x for x in list_to_rotate[:amount]]
    return cnp.array(xx)

def left_shift_list_of_slices(list_to_rotate, amount):
    xx = [x for x in list_to_rotate[amount:]] + [0 for x in list_to_rotate[:amount]]
    return cnp.array(xx)

def rotate_less_than_width(slices, shift):
    raised_low_bits = table_extract_low_bits_and_raise[shift][slices]
    shifted_raised_low_bits = right_rotate_list_of_slices(raised_low_bits, 1)

    high_bits = table_extract_high_bits[shift][slices]
    return shifted_raised_low_bits + high_bits

def right_rotate(slices, rotate_amount):
    x = rotate_amount // WIDTH
    y = rotate_amount % WIDTH
    if x != 0:    
        rotated_slices = right_rotate_list_of_slices(slices, x)
    else:
        rotated_slices = slices
    if y != 0:
        rotated = rotate_less_than_width(rotated_slices, y)
    else:
        rotated = rotated_slices

    return rotated

def right_shift(slices, shift_amount):
    x = shift_amount // WIDTH
    y = shift_amount % WIDTH
    if x != 0:
        shifted_slices = right_shift_list_of_slices(slices, x)
    else:
        shifted_slices = slices
    if y != 0:
        # shift within slices
        raised_low_bits = table_extract_low_bits_and_raise[y][shifted_slices]
        shifted_raised_low_bits = right_shift_list_of_slices(raised_low_bits, 1)
        high_bits = table_extract_high_bits[y][shifted_slices]
        result = shifted_raised_low_bits + high_bits
    else:
        result = shifted_slices
    return result

extract_one_bit_carry = cnp.LookupTable([x // (2 ** WIDTH) for x in range(2**(WIDTH+1))])
extract_slice_from_two_operand_sum = cnp.LookupTable([x % (2 ** WIDTH) for x in range(2**(WIDTH+1))])

def add_two_32_bits(slices):
    added = np.sum(slices, axis=0)

    for i in range(NUM_SLICES):
        results = extract_slice_from_two_operand_sum[added]
        if i < NUM_SLICES-1:
            carries = extract_one_bit_carry[added]
            added = left_shift_list_of_slices(carries, 1) + results

    return results

extract_two_bit_carry = cnp.LookupTable([x // (2 ** WIDTH) for x in range(2**(WIDTH+2))])
extract_slice_from_four_operand_sum = cnp.LookupTable([x % (2 ** WIDTH) for x in range(2**(WIDTH+2))])

def add_four_32_bits(slices):
    added = np.sum(slices, axis=0)
    
    # First iteration of the loop is seperated
    carries = extract_two_bit_carry[added]
    results = extract_slice_from_four_operand_sum[added]
    shifted_carries = left_shift_list_of_slices(carries, 1)
    added = shifted_carries + results

    for i in range(1,NUM_SLICES):
        results = extract_slice_from_two_operand_sum[added]
        
        # In the last iteration, carries need not be calculated
        if i != NUM_SLICES-1: 
            carries = extract_one_bit_carry[added]
            shifted_carries = left_shift_list_of_slices(carries, 1)
            added = shifted_carries + results

    return results

# Used in the expansion

def s0(w):
    return right_rotate(w, 7) ^ right_rotate(w, 18) ^ right_shift(w, 3)

def s1(w):
    return right_rotate(w, 17) ^ right_rotate(w, 19) ^ right_shift(w, 10)

# Used in main loop

def S0(a_word):
    return right_rotate(a_word, 2) ^ right_rotate(a_word, 13) ^ right_rotate(a_word, 22)

def S1(e_word):
    return right_rotate(e_word, 6) ^ right_rotate(e_word, 11) ^ right_rotate(e_word, 25)

def Ch(e_word, f_word, g_word):
    return (e_word & f_word) ^ ((2**WIDTH-1 - e_word) & g_word)

def Maj(a_word, b_word, c_word):
    return (a_word & b_word) ^ (a_word & c_word) ^ (b_word & c_word)

def main_loop(args, w_i_plus_k_i):
    a, b, c, d, e, f, g, h = args
    temp1 = add_four_32_bits(np.array([h,S1(e),Ch(e, f, g), w_i_plus_k_i]))
    temp2 = add_two_32_bits(np.array([S0(a), Maj(a, b, c)]))
    new_a = add_two_32_bits(np.array([temp1, temp2]))
    new_e = add_two_32_bits(np.array([d, temp1]))
    return np.array([new_a, a, b, c, new_e, e, f, g])

# (150*8+1+271+64)%512 = 0
PAD = 2**7 # 10000000 (8bit)
ZERO_PAD = 33
LENGTH_BIG_ENDIAN=[0,0,0,0,0,0,4,176] # 150 * 8 in big endian
TOTAL_LENGTH = 1536
def padding(data):
    to_pad = [PAD] + [0 for _ in range(ZERO_PAD)] + LENGTH_BIG_ENDIAN
    result = [x for x in data] + [x for x in to_pad]
    return cnp.array(result)

splitting_tables = [
    cnp.LookupTable([(x//2**(i*WIDTH))%(2**WIDTH) for x in range(2**8)]) for i in range(8//WIDTH)
]

def break_down_data(data):
    all_slices = []
    for i in range(8//WIDTH-1, -1, -1):
        all_slices += [[x for x in splitting_tables[i][data]]]
    all_slices = cnp.array(all_slices).transpose().reshape((48,NUM_SLICES))
    return all_slices


K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]
H = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]

def sha256(data):
    h_slices = cnp.zeros((len(H), NUM_SLICES))
    k_slices = cnp.zeros((len(K), NUM_SLICES))

    for i in range(len(H)):
        h_slices[i] += split_to_slices(H[i])

    for i in range(len(K)):
        k_slices[i] += split_to_slices(K[i])
    
    padded_data = padding(data)
    slices_data = break_down_data(padded_data) # (48, 8)
    for chunk_iter in range(0, 3):
        
        # Initializing the variables
        chunk = slices_data[chunk_iter*16:(chunk_iter+1)*16]
        w = cnp.zeros((64, NUM_SLICES))
        # Starting the main loop and expansion
        args = h_slices
        for j in range(0, 64):
            if j<16:
                w[j] = chunk[j]
            else:
                w[j] = add_four_32_bits(np.array([w[j-16], s0(w[j-15]), w[j-7], s1(w[j-2])]))
            w_i_k_i = add_two_32_bits(np.array([w[j], k_slices[j]]))
            args = main_loop(args,w_i_k_i)
        
        # Accumulating the results
        for j in range(8):
            h_slices[j] = add_two_32_bits(np.array([h_slices[j], args[j]]))
    return h_slices

# Compilation of the circuit should take a few minutes
compiler = cnp.Compiler(sha256, {"data": "encrypted"})
circuit = compiler.compile(
    inputset=[
        np.random.randint(0, 2 ** 8, size=(150,))
        for _ in range(100)
    ],
    configuration=cnp.Configuration(
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location=".keys",
    ),
    verbose=False,
)

# WARNING: This takes a LONG time
text = (
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    b"Curabitur bibendum, urna eu bibendum egestas, neque augue eleifend odio, et sagittis viverra."
)
assert len(text) == 150
encrypted_evaluation = circuit.encrypt_run_decrypt(text)

print("Encrypted Evaluation: ", slices_to_hexarray(encrypted_evaluation))
print("    Plain Evaluation: ", slices_to_hexarray(sha256(text)))