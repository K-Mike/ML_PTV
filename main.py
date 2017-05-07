from pyspark import SparkContext
from pyspark import SparkConf

import numpy as np
# from scipy.misc import imread
import os
import sys


# os.environ['PYSPARK_PYTHON'] = '/shared/anaconda/bin/python'


DATA_DIR = '/hdfs/user/lkozinkin/ptv'


# def process_image_pmc(image):
#     from pmc import PMC
#
#     pmc_obj = PMC(ksize=(7, 7), threshold_rel=0.6)
#     coordinates = pmc_obj.get_positions(image)
#
#     return coordinates

# def install_packages(x):
#     import os
#     import pip
#
#     basedir = os.path.abspath(os.path.dirname(__file__))
#
#     try:
#         pip.main(['install', 'scipy', '-t', basedir])
#     except:
#         return
#
#     import scipy as sp
#
#     return ' '.join(sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]))

sc = SparkContext()


from pmc import PMC


print(PMC())

# from scipy.signal import correlate2d
# import scipy

# import os
# import pip
#
# basedir = os.path.abspath(os.path.dirname(__file__))
# pip.main(['install', 'scipy', '-t', basedir])

# from scipy.misc import imread

# rdd = sc.parallelize(np.arange(10))
# res = rdd.map(install_packages).distinct().collect()
# res2 = rdd.map(lambda x: sys.executable).distinct().collect()
# print(res)

# print(os.environ)
# print(os.listdir(DATA_DIR))
