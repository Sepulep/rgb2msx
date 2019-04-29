## RGB to MSX screen 2 converter ##

converts 256x192 images to msx screen 2 VRAM dump

### standalone ###

Usage info:
```
chmod +x rgb2msx.py
./rgb2msx.py --help
```

for example:
```
./rgb2msx.py chimp.bmp
```

### GIMP plugin ###

Copy the rgb2msx.py file to the plugin directory 
(e.g. ~/.config/GIMP/2.10/plug-ins)

### file format ###

The scripts writes a file with a dump of the MSX screen 2 VRAM, directly loadable 
with the following MSX Basic program:
```
10 screen 2
20 BLOAD "msx.sc2", S
30 goto 30
```
If you think a different format is useful, let me know in an issue.

### Parallel mode ###

The ```-n<nproc>``` option allows parallel computation. 
This has been tested on linux. 
