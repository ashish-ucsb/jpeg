import numpy as np
import itertools
import huffman
from bitarray import bitarray
import collections
import ast
import itertools
import cv2

"""Scale image to 0-255 pixel values if required"""
scaleImage = lambda img: img*255 if (np.max(img) <= 1) else img

"""Zero center image"""
centerImage = lambda img: img - 128

"""Decenter zero centered image"""
decenterImage = lambda img: img + 128

"""Makes 'x' y-divisible"""
yDivisible = lambda x,y: 0 if (x%y==0) else y - (x%y)

"""Pad the image if required"""
padImage = lambda img, block_h, block_w: np.pad(img, [(0, yDivisible(img.shape[0], block_h)),
                                                      (0, yDivisible(img.shape[1], block_w))], mode='constant')

"""Unpad the padded image"""
unpadImage = lambda img, h, w: img[:h,:w]

def getTopNElements(a, N):
    """Takes numpy matrix a & keeps N elements in their original position acc. to their abs magnitude"""
    if (N > a.size or N < 0):
        raise Exception("N must be between 0 & {}, N:{}".format(a.size,N))
    idx = abs(a).ravel().argsort()[:-N]
    idx = np.stack(np.unravel_index(idx, a.shape)).T
    for i in idx:
        a[i[0]][i[1]] = 0
    return a

def blockDCT(new_shape, block_shape, img, N=64):
    """Apply block DCT to image"""
    (new_h, new_w) = new_shape
    (block_h, block_w) = block_shape
    dct = np.zeros(new_shape)
    for i,j in itertools.product(range(0,new_h, block_h), range(0,new_w, block_w)):
        block = cv2.dct(img[i:i+block_h, j:j+block_w])
        dct[i:i+block_h, j:j+block_w] = getTopNElements(block, N)
    return dct

def blockIDCT(new_shape, block_shape, img):
    """Apply inverse block DCT"""
    (new_h, new_w) = new_shape
    (block_h, block_w) = block_shape
    recon = np.zeros(new_shape)
    for i,j in itertools.product(range(0,new_h,block_h), range(0,new_w,block_w)):
        recon[i:i+block_h, j:j+block_w] = cv2.idct(img[i:i+block_h, j:j+block_w])
    return recon

"""Quantization matrix"""
Q = np.array([
    [ 16,  11,  10,  16,  24,  40,  51,  61],
    [ 12,  12,  14,  19,  26,  58,  60,  55],
    [ 14,  13,  16,  24,  40,  57,  69,  56],
    [ 14,  17,  22,  29,  51,  87,  80,  62],
    [ 18,  22,  37,  56,  68, 109, 103,  77],
    [ 24,  35,  55,  64,  81, 104, 113,  92],
    [ 49,  64,  78,  87, 103, 121, 120, 101],
    [ 72,  92,  95,  98, 112, 100, 103,  99]
])

def quantize(x, block_shape, quality=1.0):
    """Quantize matrix x using quantization matrix Q"""
    (h, w) = x.shape
    (block_h, block_w) = block_shape
    for i,j in itertools.product(range(0,h,block_h), range(0,w,block_w)):
        x[i:i+block_h, j:j+block_w] = np.round((x[i:i+block_h, j:j+block_w]/(Q/quality)))
    return x

def dequantize(x, block_shape):
    """Dequantize matrix x using quantization matrix Q"""
    (h, w) = x.shape
    (block_h, block_w) = block_shape
    for i,j in itertools.product(range(0,h,block_h), range(0,w,block_w)):
        x[i:i+block_h, j:j+block_w] = np.multiply(x[i:i+block_h, j:j+block_w],Q)
    return x

def zigzagIndices(shape):
    """Returns zigzag indices of a given shape"""
    (h,w) = shape
    a = (np.arange(h*w)).reshape(h,w)
    if h>=w:
        a = np.concatenate([np.diagonal(a[::-1,:], i)[::(2*((i+1) % 2)-1)] for i in range(1-h, h)])
    else:
        a = np.concatenate([np.diagonal(a[::-1,:], i)[::(2*((i+1) % 2)-1)] for i in range(1-w, w)])
    idx = np.stack(np.unravel_index(a, shape)).T
    return idx

def zigzagImage(img, block_shape):
    """Returns zigzag scanned matrix"""
    (h, w) = img.shape
    (block_h, block_w) = block_shape
    z = np.zeros((int((h/block_h)*(w/block_w)), block_h*block_w))
    ctr = 0
    idx = zigzagIndices(block_shape)
    for i,j in itertools.product(range(0,h,block_h), range(0,w,block_w)):
        block = img[i:i+block_h, j:j+block_w]
        z[ctr] = [block[m,n] for m,n in idx]
        ctr+=1
    return z.flatten()

def unzigzagImage(scanned, new_shape, block_shape):
    """Returns original matrix from zigzag string"""
    scanned = scanned.reshape((-1,64))
    (new_h, new_w) = new_shape
    (block_h, block_w) = block_shape
    y = np.zeros(new_shape)
    ctr1 = 0
    idx = zigzagIndices(block_shape)
    for i,j in itertools.product(range(0, new_h, block_h), range(0, new_w, block_w)):
        block = np.zeros(block_shape)
        ctr2 = 0
        for m,n in idx:
            block[m,n] = scanned[ctr1][ctr2]
            ctr2 += 1
        ctr1 += 1
        y[i:i+8, j:j+8] = block     
    return y

def runEncode(a):
    """Run length encode a given stream"""
    l = []
    ctr = 0
    for i in a:
        if (i!=0):
            l.extend((ctr,i))
            ctr=0
        else:
            ctr+=1
    l.extend((0,0))
    return np.array(l)

def runDecode(a,ln):
    l = []
    for i,j in a.reshape((-1,2)):
        if ((i,j) == (0,0)):
            break
        l.extend([0]*i)
        l.append(j)
    if (len(l) < ln):
        l.extend([0]*(ln-len(l)))
    return np.array(l)

charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'

def pack64(bin_string):
    """Converts a binary string into base64 string"""
    chunks = [bin_string[i:i+6] for i in range(0, len(bin_string), 6)]
    last_chunk_length = len(chunks[-1])
    decimals = [int(chunk, 2) for chunk in chunks]
    decimals.append(last_chunk_length)
    ascii_string = ''.join([charset[i] for i in decimals])
    return ascii_string

def unpack64(ascii_string):
    """Converts a base 64 string into binary string"""
    decimals = [charset.index(char) for char in ascii_string]
    last_chunk_length, last_decimal = decimals.pop(-1), decimals.pop(-1)
    bin_string = ''.join([bin(decimal)[2:].zfill(6) for decimal in decimals])
    bin_string += bin(last_decimal)[2:].zfill(last_chunk_length)
    return bin_string

"""Converts dictionary keys to bitarray keys"""
bitarrayDict = lambda dt: dict((key, bitarray(value)) for key,value in dt.items())

def huffmanEncode(data):
    """Returns huffman encoded data along with codebook"""
    codebook = huffman.codebook(collections.Counter(data).items())
    encoded = bitarray()
    encoded.encode(bitarrayDict(codebook), data)
    encoded = pack64(str(encoded)[10:-2])
    return encoded, codebook

def huffmanDecode(encoded, codebook):
    """Decodes huffman encoded data given a codebook"""
    if type(codebook) is str:
        codebook = ast.literal_eval(codebook)
    encoded = bitarray(unpack64(encoded))
    return encoded.decode(bitarrayDict(codebook))