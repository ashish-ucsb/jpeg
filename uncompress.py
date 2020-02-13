import myjpeg
import numpy as np
import cv2
import itertools
import optparse
import ast

parser = optparse.OptionParser()
parser.add_option('-i', '--image', action="store", dest="img", help="myJPEG image")
options, args = parser.parse_args()

f = open(options.img, "r")
data = f.read()
f.close()

# Seperate the data
data = ast.literal_eval(data)
block_shape = (block_h, block_w) = data['block_shape']
org_shape = (h, w) = data['org_shape']
new_shape = (new_h, new_w) = data['new_shape']
codebook = data['codebook']
encoded = data['encoded']

# Decode the stream using codebook
decoded = myjpeg.huffmanDecode(encoded, codebook)

# Runlength decode the stream
decoded = myjpeg.runDecode(np.array(decoded), new_h*new_w)

# unzigzag stream
unzigzag = myjpeg.unzigzagImage(np.array(decoded), new_shape, block_shape)

# Dequantize the dct matrix
dequant = myjpeg.dequantize(unzigzag, block_shape)

# Reconstruct the image
recon = myjpeg.blockIDCT(new_shape, block_shape, dequant)

# Unpad the image back to its original size
recon = myjpeg.unpadImage(recon, h, w)

# Decenter the image
recon = myjpeg.decenterImage(recon)

fname = options.img.split('.')[0] + "_.jpg"
cv2.imwrite(fname, recon)
print("Image successfully reconstructed")
print("Image saved as {}".format(fname))