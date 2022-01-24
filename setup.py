from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

dist_metrics = Extension('dist_metrics',
                         sources=['./dist_metrics.pyx'])
_hdbscan_tree = Extension('_hdbscan_tree',
                          sources=['./_hdbscan_tree.pyx'])
_d_heap_queue = Extension('_d_heap_queue',
                          sources=['./_d_heap_queue.pyx'])
_hdbscan_linkage = Extension('_hdbscan_linkage',
                             sources=['./_hdbscan_linkage.pyx'])
setup(ext_modules=cythonize([dist_metrics, _hdbscan_tree, _d_heap_queue, _hdbscan_linkage]))