#! /usr/bin/env python

'''
----------------------------------------------------------------------------
RGB Image to MSX Converter

Original Graphic routine To convert RGB images To MSX1 video, 
made in BlitzBasic by Leandro Correia (www.leandrocorreia.com)

python 2.7 converted by MsxKun (uses PIL library)

python + numpy and gimp plugin by Sepulep
----------------------------------------------------------------------------
'''

#from os.path import basename, splitext, dirname, abspath

import os
import sys
import numpy
import time

from multiprocessing import Pool

try:
  from gimpfu import *
  HAS_GIMP=True
except ImportError:
  HAS_GIMP=False

MSX_PALETTE = [ 0,0,0,0,0,0,36,219,36,109,255,109,36,36,255,73,109,255,182,36,36,73,219,
                  255,255,36,36,255,109,109,219,219,36,219,219,146,36,146,36,219,73,182,182,
                  182,182,255,255,255]


'''
---------------------------------------------------------------------------- 
Calculate Distance Function
----------------------------------------------------------------------------

Convert two RGB color values into Lab and uses the CIEDE2000 formula to calculate 
the distance between them.



This function first converts RGBs to XYZ and then to Lab.

'''

def rgb2cielab(r,g,b):
    r = r/255.0
    g = g/255.0
    b = b/255.0   
    
    r = ((r+0.055)/1.055)**2.4*(r > 0.04045) + (r/12.92)*(r <= 0.04045)
    g = ((g+0.055)/1.055)**2.4*(g > 0.04045) + (g/12.92)*(g <= 0.04045)
    b = ((b+0.055)/1.055)**2.4*(b > 0.04045) + (b/12.92)*(b <= 0.04045)
    
    r = r*100.0
    g = g*100.0
    b = b*100.0
     
    x = r*0.4124 + g*0.3576 + b*0.1805
    y = r*0.2126 + g*0.7152 + b*0.0722
    z = r*0.0193 + g*0.1192 + b*0.9505
  
    x = x/95.047
    y = y/100.000
    z = z/108.883
  
    x = x**(1.0/3.0)*(x > 0.008856) + ((7.787*x) + (16.0/116.0))*(x <= 0.008856)
    y = y**(1.0/3.0)*(y > 0.008856) + ((7.787*y) + (16.0/116.0))*(y <= 0.008856)
    z = z**(1.0/3.0)*(z > 0.008856) + ((7.787*z) + (16.0/116.0))*(z <= 0.008856)
  
    ''' Converts XYZ to Lab... '''
  
    l = (116*y)-16.0
    a = 500.0*(x-y)
    b = 200.0*(y-z)
    return l,a,b

# tested against http://www2.ece.rochester.edu/~gsharma/ciede2000/dataNprograms/ciede2000testdata.txt
def dist2000(l1,a1,b1,l2,a2,b2):  
    ''' ...and then calculates distance between Lab colors, using the CIEDE2000 formula. '''
  
    dl = l2-l1
    hl = l1+dl*0.5
    sqb1 = b1*b1
    sqb2 = b2*b2
    c1 = (a1*a1+sqb1)**0.5
    c2 = (a2*a2+sqb2)**0.5
    hc7 = ((c1+c2)*0.5)**7
    trc = (hc7/(hc7+6103515625.0))**0.5
    t2 = 1.5-trc*0.5
    ap1 = a1*t2
    ap2 = a2*t2
    c1 = (ap1*ap1+sqb1)**0.5
    c2 = (ap2*ap2+sqb2)**0.5
    dc = c2-c1
    hc = c1+dc*0.5
    hc7 = hc**7.0
    trc = (hc7/(hc7+6103515625.0))**0.5
    h1 = numpy.arctan2(b1,ap1)
    h1 = (h1+numpy.pi*2.0)*(h1<0) + h1*(h1>=0)
    h2 = numpy.arctan2(b2,ap2)
    h2 = (h2+numpy.pi*2.0)*(h2<0) + h2*(h2>=0)
    hdiff = h2-h1
    hh = h1+h2
  
    hh= hh *(numpy.abs(hdiff)<=numpy.pi) + \
           (hh + 2*numpy.pi) *(numpy.abs(hdiff)>numpy.pi)*(hh<2*numpy.pi) + \
           (hh - 2*numpy.pi) *(numpy.abs(hdiff)>numpy.pi)*(hh>=2*numpy.pi)

    #~ hh=(numpy.abs(hdiff)<=numpy.pi)*hh+(hh+2*numpy.pi)*(numpy.abs(hdiff)>numpy.pi)
  
    hdiff= hdiff *(numpy.abs(hdiff)<=numpy.pi) + \
           (hdiff - 2*numpy.pi) *(h2-h1 > numpy.pi) + \
           (hdiff + 2*numpy.pi) *(h2-h1 < -numpy.pi)

    #~ hdiff=(numpy.abs(hdiff)<=numpy.pi)*(hdiff-2*numpy.pi)+ \
          #~ (hdiff)*(numpy.abs(hdiff)>numpy.pi)*(h2>h1) + \
          #~ (hdiff+2*numpy.pi)*(numpy.abs(hdiff)>numpy.pi)*(h2<=h1)
    
    hh = hh*0.5
    t2 = 1-0.17*numpy.cos(hh-numpy.pi/6)+0.24*numpy.cos(hh*2)
    t2 = t2+0.32*numpy.cos(hh*3+numpy.pi/30.0)
    t2 = t2-0.2*numpy.cos(hh*4-numpy.pi*63/180.0)
    dh = 2*(c1*c2)**0.5*numpy.sin(hdiff*0.5)
    sqhl = (hl-50)*(hl-50)
    fl = dl/(1+(0.015*sqhl/(20.0+sqhl)**0.5))
    fc = dc/(hc*0.045+1.0)
    fh = dh/(t2*hc*0.015+1.0)
    #~ dt = 30*numpy.exp(-(36.0*hh-55.0*numpy.pi**2.0)/(25.0*numpy.pi*numpy.pi))
    #~ dt = 30*numpy.exp(-((36.0*hh-55.0*numpy.pi)**2.0)/(25.0*numpy.pi*numpy.pi))
    dt = 30*numpy.exp(-((hh/numpy.pi*180-275)/25.0)**2)
    r = -2*trc*numpy.sin(2.0*dt*numpy.pi/180.0)
  
    return (fl*fl+fc*fc+fh*fh+r*fc*fh)**0.5 

 
def calcdist2000(r1, g1, b1, r2, g2, b2):

    l1,a1,b1=rgb2cielab(r1,g1,b1)
    l2,a2,b2=rgb2cielab(r2,g2,b2)

    return dist2000(l1,a1,b1,l2,a2,b2)
  
class RGB2MSX(object):
    def __init__(self,tolerance=100, palette=MSX_PALETTE, detail_weight=0, nproc=1): # possible: ~20% faster with multiple blocks per pass

        self.tolerance=numpy.clip(tolerance,0,100)*1.01  # *101 because differences can be slightly larger than 100..
        self.palette=palette
        self.detail_weight=numpy.clip(detail_weight,0.,1.)
        self.nproc=nproc
        
        msxr = palette[0::3]
        msxg = palette[1::3]
        msxb = palette[2::3]
        
        toner=[]
        toneg=[]
        toneb=[]        
        tone_weight=[]
        combinations=[]
                
        for cor1 in range(1,16):
          for cor2 in range(cor1,16):
              combinations.append([cor1,cor2])
              toner.append([msxr[cor1], msxr[cor2], (msxr[cor1]+msxr[cor2])/2])
              toneg.append([msxg[cor1], msxg[cor2], (msxg[cor1]+msxg[cor2])/2])
              toneb.append([msxb[cor1], msxb[cor2], (msxb[cor1]+msxb[cor2])/2])
              cd=calcdist2000(msxr[cor1],msxg[cor1],msxb[cor1],msxr[cor2],msxg[cor2],msxb[cor2])
              tone_weight.append([0,0,cd])
        
        toner=numpy.array(toner)
        toneg=numpy.array(toneg)
        toneb=numpy.array(toneb)
    
        tone_weight=numpy.array(tone_weight)
        self.no_dither_tones=tone_weight>self.tolerance
        self.combinations=numpy.array(combinations)
    
        self.tr=numpy.reshape(numpy.outer(toner,numpy.ones(8)),toner.shape+(8,))
        self.tg=numpy.reshape(numpy.outer(toneg,numpy.ones(8)),toner.shape+(8,))
        self.tb=numpy.reshape(numpy.outer(toneb,numpy.ones(8)),toner.shape+(8,))
  
        dither=numpy.ones((8,2),dtype="int32")
        dither[::2,0]=0
        dither[1::2,1]=0
        self.dither=dither

    def detail(self, data):
        lum=numpy.sum(data,axis=2)/3

        detail=numpy.zeros_like(lum)
        
        detail[1:-1,1:-1]=numpy.maximum( numpy.abs(lum[:-2,1:-1]-lum[1:-1,1:-1]),numpy.abs(lum[1:-1,1:-1]-lum[2:,1:-1])) + \
                          numpy.maximum( numpy.abs(lum[1:-1,:-2]-lum[1:-1,1:-1]),numpy.abs(lum[1:-1,1:-1]-lum[1:-1,2:]))

        detail=detail/2.

        return detail


    def _convert(self,data, progress_callback=None):
  
        Nx=data.shape[0]
        Ny=data.shape[1]
        assert data.shape[2] == 3
        
        detail=self.detail(data)
        
        tl,ta,tb=rgb2cielab(self.tr,self.tg,self.tb)        
        
        result=numpy.zeros((Nx,Ny), dtype=numpy.int8)  # result array with indexed pic
        patarray=numpy.zeros((Nx//8, Ny), dtype=numpy.int8) # pattern array
        colarray=numpy.zeros((Nx//8, Ny), dtype=numpy.int8) # color array
        
        y=0
        
        p,o=numpy.indices((self.tr.shape[0], self.tr.shape[2]))
        
        while y<Ny:
            if progress_callback and callable(progress_callback):
              progress_callback(y/(1.*Ny))

            x=0
            
            # parallelize?
            while x<Nx:
                octetr=data[x:x+8,y,0]
                octetg=data[x:x+8,y,1]
                octetb=data[x:x+8,y,2]
                octetdetail=detail[x:x+8,y]
                
                octetl,octeta,octetb=rgb2cielab(octetr,octetg,octetb)

                alldistances=dist2000(tl,ta,tb, octetl,octeta,octetb) * (1.+octetdetail*self.detail_weight)
                alldistances[self.no_dither_tones]=999.
                best_color_for_each_tone=numpy.argmin(alldistances,axis=1)
                best_distance=alldistances[p,best_color_for_each_tone,o]
                best_distance=numpy.sum(best_distance,axis=1) #
                best_tone=numpy.argmin(best_distance)
        
                best=best_color_for_each_tone[best_tone,:]        
                dithered= best==2
                best[dithered]=self.dither[dithered,y%2]
                
                result[x:x+8,y]=self.combinations[best_tone][best]
                patarray[x//8,y]=numpy.packbits(numpy.uint8(best))
                colarray[x//8,y]=self.combinations[best_tone][0]+16*self.combinations[best_tone][1]
        
                x+=8
            y+=1
        
        #  shuffle patarray and colarray 
        flat=numpy.arange(8*(Nx//8)*(Ny//8)) # truncates to multiple of 8!
        patbuffer=numpy.uint8(patarray[ (flat//8)%(Nx//8) , 8*(flat//(Nx))+flat%8])
        colbuffer=numpy.uint8(colarray[ (flat//8)%(Nx//8) , 8*(flat//(Nx))+flat%8])

        return result, patbuffer, colbuffer

    def convert(self,data, progress_callback=None):
        if self.nproc<=1:
            return self._convert(data, progress_callback)
        Nx=data.shape[0]
        Ny=data.shape[1]
        pool=Pool(self.nproc)
        
        N=pool._processes
        n=8
        
        result=numpy.zeros((Nx,Ny), dtype=numpy.int8)  # result array with indexed pic
        patbuffer=numpy.zeros( (Nx//8) * Ny, dtype=numpy.uint8) # pattern array
        colbuffer=numpy.zeros( (Nx//8) * Ny, dtype=numpy.uint8) # color array
        
        requests=[ pool.apply_async( pool_wrap,  (data[:,i*n:(i+1)*n,:],)) for i in range(Ny//n) ]

        for i, request in enumerate(requests):
            (r,p,c)=request.get()
            if progress_callback and callable(progress_callback):
                progress_callback(i/(1.*len(requests)))
            low=i*n
            high=(i+1)*n
            result[:,low:high]=r
            patbuffer[ (Nx//8)*low : (Nx//8)*high]=p
            colbuffer[ (Nx//8)*low : (Nx//8)*high]=c
        
        return result,patbuffer,colbuffer
            
def pool_wrap(data):        
    global rgb2msx
    return rgb2msx._convert(data)

# test
def write_vram(patbuffer,colbuffer, outputfile="test.sc2"):
    header = chr(0xFE)+chr(0)*2+chr(0xFF)+chr(0x37)+chr(0)*2

    if len(patbuffer)!=6144 or len(colbuffer)!=6144:
      print("warning: not a valid buffer size")

    nambuf = bytearray()
    for i in range(256):
        nambuf.append(i)
    sprbuf = bytearray(1280)

    f = open(outputfile,"wb")
    f.write(header)
    f.write(patbuffer.tobytes())
    f.write(nambuf)
    f.write(nambuf)
    f.write(nambuf)
    f.write(sprbuf)
    f.write(colbuffer.tobytes())
    f.close()

import argparse

def optionparser():
    parser = argparse.ArgumentParser(description='convert image to MSX 16 color')
    parser.add_argument('input_file', type= str, help='input file name')
    parser.add_argument('-o ', dest='output_file', type= str, help='output file name', default="msx.sc2")
    parser.add_argument('-d ', dest='dither_tolerance', type= int, help='dither tolerance value (0-100, smaller means less dithering)', default=100)
    parser.add_argument('-l ', dest='detail_weight', type= float, help='level of detail adjustment weight (0-1)', default=0)
    parser.add_argument('-n', dest='nproc', type= int, help='numper of processes to use (default=1)', default=1)
    return parser

def standalone():
    global rgb2msx
    from PIL import Image

    parser=optionparser()
    args = parser.parse_args()

    inputfile = args.input_file
    
    im = Image.open(inputfile)       # Can be many different formats.
    im.show()
    data = numpy.array(im)
    data = numpy.transpose(data, (1,0,2))
    
    rgb2msx=RGB2MSX(tolerance=args.dither_tolerance, detail_weight=args.detail_weight, nproc=args.nproc)
    
    result,pat,col=rgb2msx.convert(data)
    
    write_vram(pat,col, outputfile=args.output_file)
  
    im=Image.fromarray(result.T, 'P')
    im.putpalette(rgb2msx.palette)
    im.show()

def gimp2msx(image, layer, dither_threshold=100, detail_weight=0, scale=False, writeVRAM=False, dirname="", filename="", nproc=1):
    global rgb2msx

  # in place if indexed and size match?
    xoffset=0
    yoffset=0
  
    width=layer.width
    height=layer.height
        
    if scale and (width!=256 or height!=192) or image.base_type!=RGB:    
        _image=gimp.Image(layer.width, layer.height, RGB)
        layer=pdb.gimp_layer_new_from_drawable(layer, _image)
        _image.add_layer(layer,0)
  
        if layer.width>4/3.*layer.height:
            height=192
            width=(192*layer.width)//layer.height
        else:
            width=256
            height=(256*layer.height)//layer.width
        _image.scale(width,height)      
        if width>256:
            xoffset=(width-256)//2
            width=256
        if height>192:
            yoffset=(height-192)//2
            height=192
    
    region=layer.get_pixel_rgn(xoffset, yoffset, width, height, False)

    data=numpy.frombuffer(region[:,:], dtype=numpy.uint8)
    data=data.reshape((height,width,3))
    data = numpy.transpose(data, (1,0,2))
        
    rgb2msx=RGB2MSX(tolerance=dither_threshold,detail_weight=detail_weight, nproc=nproc)
    
    pdb.gimp_progress_init("Converting to MSX image",None)
    result,pat,col=rgb2msx.convert(data, progress_callback=pdb.gimp_progress_update)
    palette=rgb2msx.palette
    
    new_image = gimp.Image(width, height, INDEXED)
    
    pdb.gimp_image_set_colormap(new_image, len(palette), palette)  
    
    layer = gimp.Layer(new_image, "MSX 16 color", width, height, INDEXED_IMAGE,
                             100, NORMAL_MODE)
    region=layer.get_pixel_rgn(0,0,width,height, True)
  
    region[:,:]=numpy.uint8(result.T).tobytes()
    new_image.add_layer(layer, 0)
  
    pdb.gimp_display_new(new_image)

    if writeVRAM:
        if dirname is None:
            dirname=""
        write_vram(pat,col, outputfile=os.path.join(dirname,filename))

    if writeVRAM and (width!=256 or height!=192):
        pdb.gimp_message("written file probably invalid: the size of the image is not 256x192, rescale or enable scaling")


def do_gimp():
    register(
      "python_fu_sample",
      "convert to MSX",
      "Convert image to MSX 16 color image",
      "FIP",
      "FIP",
      "april 2019",
      "<Image>/Filters/MSX/RGB2MSX",
      "",      # Create a new image, don't work on an existing one
      [ 
      (PF_INT, "dither_tolerance", "dither threshold (0-100, lower means less dither)", 100),
      (PF_FLOAT, "detail_weight", "weight given to detail in adjustment for detail level (0-1)", 0),
      (PF_BOOL, "scale", "Scale image to 256x192?", True),
      (PF_BOOL, "writeVRAM", "Write MSX VRAM image?", True),
      (PF_DIRNAME, "dirname", "Save directory:", ""),
      (PF_STRING, "filename", "MSX VRAM file name:", "msx.sc2"), 
      (PF_INT, "nproc", "number of processors to use", 1),
      ],
      [],
      gimp2msx)
    
    main()
  
  
if __name__=="__main__":
    if HAS_GIMP:
      do_gimp()
    else:
      standalone()
