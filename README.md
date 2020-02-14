# Baseline JPEG implementation

Your code should take the following as input parameters:  
- image file name  
- number of coefficients retained, a number between 1-64  
  
and the output should include:  
- the RMSE  
- image reconstructed from the retained coefficients.  

Your code will :  
- read the input image,  
- subtract 128 from each pixel value,  
- partition the image into blocks of 8x8 pixels (zero padding if needed to fill up the boundary blocks),  
- compute the DCT of each 8x8 block,  
- threshold the DCT coefficients to retain only the number of coefficients N as specified and zero out the remaining coefficients,  
- and then reconstruct the 8x8 blocks and put the reconstructed picture together.  

Extra Credit :  

- Huffman code the coefficients (including zig-zag scanning, creating the run-level pairs, and huffman coding these pairs), and construct your version of a JPEG image
with extension .myJPEG.  
- You should also have a read function for this file extension.  
- Compare and contrast with the standard JPEG compression.  
- The *.myJPEG image should be self-contained and should have all the data embedded in it for reconstruction (including the huffman table as needed).  
- For this part of the assignment, you may use the standard (default) JPEG quantization matrix and you may assume that this is not part of
the JPEG image.

___

## Usage

- Install required packages from `requirement.txt`
`pip install -r requirements.txt`
- Compress a given image:  
`python compress.py -i zelda.png -n 64 -q 0.95`  
 -i image to compress  
 -n coeffs to keep, 1 to 64  
 -q Image quality 0.0 to 1.0  
- Uncompress MYJPEG image:  
`python uncompress.py -i zelda.myJPEG`  
	-i image to uncompress
