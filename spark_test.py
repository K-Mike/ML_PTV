from pyspark import SparkContext
from pyspark import SparkConf

import os
import sys


# PROJECT_DIR = '/home/lkozinkin/ptv'


# py_files = [
#     '/home/lkozinkin/ptv/PTV/particle_detection/pmc.py',
#     '/home/lkozinkin/ptv/PTV/particle_detection/subpixel_peak.py',
#     '/home/lkozinkin/ptv/PTV/tracking/relaxation.py'
# ]

# py_files = [
#     'hdfs:ptv/pmc.py',
#     'hdfs:ptv/subpixel_peak.py',
#     'hdfs:ptv/relaxation.py'
# ]

# for root, dirs, files in os.walk(PROJECT_DIR):
#     for file in files:
#         if file.endswith('.py'):
#             py_files.append(os.path.join(root, file))
#
sc = SparkContext()
# sc.addPyFile('pmc.py')
# sc.addPyFile('relaxation.py')
# print(py_files)

# from pmc import PMC
# import numpy
# import scipy
# print(scipy)

import six
print(six)

# import skimage
# print(skimage)

# import matplotlib
# print(matplotlib)

print('Import successful')

# try:
#     from PTV.particle_detection import PMC
#
# except ImportError as e:
#     print("Error importing Spark Module", e)
#
# try:
#     # conf = SparkConf()
#     # conf.setMaster(SPARK_HOME)
#     # conf.setAppName("First_Remote_Spark_Program")
#     # sc = SparkContext(conf=conf)
#     sc = SparkContext()
#     # print("connection succeeded with Master", conf)
#     data = [1, 2, 3, 4, 5]
#     distData = sc.parallelize(data)
#
# except Exception:
#     print("unable to connect to remote server")

# pmc_obj = PMC()
# print(pmc_obj)
print('Hello, World!')
