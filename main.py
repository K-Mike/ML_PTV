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


def process_pmc(rdd, kernel_size=(5, 5), threshold_rel=0.5):
    import numpy as np
    from io import BytesIO
    from pmc import PMC

    img_label = rdd[0].rsplit('/', 1)[1][:-6]

    with BytesIO(rdd[1]) as buffer:
        image = np.load(buffer)

    detector = PMC(ksize=kernel_size, threshold_rel=threshold_rel)
    points = detector.get_positions(image)

    return img_label, points


def process_relaxation(rdd, R_n=30.0, R_p=30.0, R_c=1, A=0.3, B=3.0, epoch_n=1, max_velocity=0):
    import numpy as np
    from relaxation import Relaxation

    img_label = rdd[0]

    frame_a = rdd[1][0]
    frame_b = rdd[1][1]

    predictor = Relaxation(R_n=R_n, R_p=R_p, R_c=R_c, A=A, B=B, epoch_n=epoch_n)
    predictor.fit(frame_a, frame_b)
    vector_field = predictor.predict()

    # velocity filtering
    if max_velocity:
        vec_field_filtered = []

        for particle in vector_field:
            if np.sqrt((particle[2] - particle[0])**2 + (particle[3] - particle[1])**2) < max_velocity:
                vec_field_filtered.append(particle)

        vector_field = np.vstack(vec_field_filtered)

    return img_label, vector_field


rdd_frame_a = sc.binaryFiles('ptv/data/*a.npy').map(lambda rdd: process_pmc(rdd, threshold_rel=0.45))
rdd_frame_b = sc.binaryFiles('ptv/data/*b.npy').map(lambda rdd: process_pmc(rdd, threshold_rel=0.5))

rdd_paired_frames = rdd_frame_a.join(rdd_frame_b)

res = rdd_paired_frames.map(lambda rdd: process_relaxation(rdd, R_n=60.0, R_p=30.0,
                                                           R_c=10, A=0.3, B=3.0,
                                                           epoch_n=6, max_velocity=20))
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
