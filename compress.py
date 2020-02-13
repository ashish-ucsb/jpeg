import myjpeg
import numpy as np
import cv2
import itertools
import optparse

parser = optparse.OptionParser()
parser.add_option('-i', '--image', action="store", dest="img", help="Image location")
parser.add_option('-n', '--num', type="int", action="store", dest="num", help="Number of coefficients to keep")
parser.add_option('-q', '--qual', type="float", action="store", dest="qual", help="Quality 0 to 1")
options, args = parser.parse_args()

N = options.num 			# Top N elements in each block
quality = options.qual 		# Image quality 
block_shape = (block_h, block_w) = (8,8)
org_img = cv2.imread(options.img, 0).astype(float)

# Scale image to 0-255 pixel values
org_img = myjpeg.scaleImage(org_img)
org_shape = (h,w) = org_img.shape

# zero center & pad the image
img = myjpeg.centerImage(org_img)
img = myjpeg.padImage(img, 8, 8)
new_shape = (new_h, new_w) = img.shape

dct = myjpeg.blockDCT(new_shape, block_shape, img, N)

# Quantize the image
quant = myjpeg.quantize(dct, block_shape, quality)

# Zigzag Scan the image
scanned = myjpeg.zigzagImage(quant, block_shape)

# Runlength encode the scanned stream
scanned = myjpeg.runEncode(scanned)

# Huffman encode the image
encoded, codebook = myjpeg.huffmanEncode(scanned.astype(int))

# Prepare data to write to file
data = {
	"block_shape": (block_h, block_w),
    "org_shape": (h,w),
    "new_shape": (new_h,new_w),
    "codebook": codebook,
    "encoded": encoded   
}

# write JPEG encoded data
fname = options.img.split('.')[0] + ".myJPEG"
f = open(fname, "w")
f.write(str(data))
f.close()

print("Data successfully written to {}".format(fname))