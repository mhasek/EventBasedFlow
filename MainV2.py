import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import cv2
import scipy.io as sio

os.chdir('/home/mhasek/PycharmProjects/EBOF')

fname = open('events_all_test.p', 'rb')

events = pickle.load(fname)

time_window_len = 1.
overlap_len = 0
spatial_window_len = 3
t0 = events['XYPOLtms'][0][0,3]

xy = events['XYPOLtms'][0][:,0:2]
t = events['XYPOLtms'][0][:,3:4] - t0



for i, item in enumerate(events['XYPOLtms'][1:-1]):
	time1 = time.time()

	ts = events['t'][i] - t0
	xy = np.vstack((xy,item[:,0:2]))
	t = np.vstack((t, item[:, 3:4] - t0 ))

	time2 = time.time()
	print time2 - time1

xyt= {}
xyt['xy'] = xy
xyt['t'] = t
sio.matlab.savemat('xyt.mat',xyt)
