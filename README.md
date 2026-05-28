This sample illustrates how to get images from the blaze camera using the Python Harvester module.

Information about the Harvester module can be found here: https://github.com/genicam/harvesters

Harvester 1.3.5 or above is required.

The requirements for the `simple_grab.py` sample are documented in the `requirements.txt` file. This can be found in the same folder as this readme.

To install the requirements, execute:

```bash
pip3 install -r requirements.txt
```

How to Execute the Sample
=========================

Windows
-------

* Navigate to the folder this readme is located in.
* Execute the sample program: `python simple_grab.py`

Linux
-----

* Navigate to the folder this readme is located in.
* Execute the sample program: `python3 simple_grab.py`

If you have installed the pylon Camera Software Suite and the pylon Supplementary Package for blaze to a location other than **/opt/pylon**, you have to adapt the path where Harvesters should load 
the GenTL producer for blaze cameras from. In the sample code search for occurrences of 
`/opt/pylon/lib/gentlproducer/gtl` and adjust the path.


Troubleshooting
================

* The `import cv2` statement fails on Windows: "ImportError: DLL load failed: The specified module could not be found"

Windows N and KN editions don't contain the Media Feature Pack that is required by OpenCV.
If you are using a Windows N or KN edition, the Media Feature Pack can be installed manually: 
https://support.microsoft.com/en-us/help/3145500/media-feature-pack-list-for-windows-n-editions
(Source: https://pypi.org/project/opencv-python/)
