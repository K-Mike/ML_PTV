# This Python file uses the following encoding: utf-8
from pyspark import SparkContext
from pyspark import SparkConf

# from io import BytesIO
# import numpy as np
# from scipy.misc import imread
import os
import sys


# os.environ['PYSPARK_PYTHON'] = '/shared/anaconda/bin/python'


# DATA_DIR = 'ptv/data'


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


# from pmc import PMC
#
#
# print(PMC())


def process_pmc(rdd):
    import numpy as np
    from io import BytesIO
    from pmc import PMC

    img_label = rdd[0].rsplit('/', 1)[1][:-6]

    with BytesIO(rdd[1]) as buffer:
        image = np.load(buffer)

    detector = PMC(ksize=(9, 9), threshold_rel=0.6)
    points = detector.get_positions(image)

    return img_label, points


def process_relaxation(rdd):
    from relaxation import Relaxation

    img_label = rdd[0]

    frame_a = rdd[1][0]
    frame_b = rdd[1][1]

    predictor = Relaxation()
    predictor.fit(frame_a, frame_b)
    vector_field = predictor.predict()

    return img_label, vector_field


rdd_frame_a = sc.binaryFiles('ptv/data/*a.npy').map(process_pmc)
rdd_frame_b = sc.binaryFiles('ptv/data/*b.npy').map(process_pmc)

rdd_paired_frames = rdd_frame_a.join(rdd_frame_b)

res = rdd_paired_frames.map(process_relaxation)
res.saveAsTextFile('ptv/result')


# test = sc.binaryFiles('ptv/data/*a.npy').map(lambda rdd: rdd[0].rsplit('/', 1)[1][:-6]).saveAsTextFile('ptv/logs')
# print(test)

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
