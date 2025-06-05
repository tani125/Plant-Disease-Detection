import distutils.core
import os
import sys
from subprocess import call

def install_detectron2():
    
    call(['git', 'clone', 'https://github.com/facebookresearch/detectron2'])
    
    
    dist = distutils.core.run_setup("./detectron2/setup.py")
    call(['pip', 'install'] + dist.install_requires)
    
    
    sys.path.insert(0, os.path.abspath('./detectron2'))

if __name__ == "__main__":
    install_detectron2()
