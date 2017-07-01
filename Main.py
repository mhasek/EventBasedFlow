import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import cv2

os.chdir('/home/mhasek/PycharmProjects/EBOF')

fname = open('events_all_test.p', 'rb')

events = pickle.load(fname)

time_window_len = 1.
overlap_len = 0
spatial_window_len = 3

t0 = events['t'][0]

for i, item in enumerate(events['XYPOLtms']):
	time1 = time.time()

	ts = events['t'][i] - t0
	xy = item[:, 0:2]
	t = item[:, 3] - t0
	treal = item[:,3]
	M = item.shape[0]

	j = 0
	while j < M:

		tcurr = t[j]
		evntidx = abs(t-tcurr) < time_window_len/2.  # define window time
		im = np.zeros((180, 240))
		currev = np.int_(xy[evntidx, :])
		tcurrev = treal[evntidx]
		clr = 1
		idxlist = np.argwhere(evntidx == 1)

		for indx in idxlist:
			currevidx = (np.abs(currev[:, 0] - xy[indx, 0]) < spatial_window_len) & (np.abs(currev[:, 1] - xy[indx, 1]) < spatial_window_len)
			currev_local = currev[currevidx]
			tcurrev_local = tcurrev[currevidx]

			# estimate plane
			# len = tcurrev_local.size
			# X = np.hstack((currev_local, tcurrev_local.reshape(len, 1), np.ones((len, 1))))		# visualization of events
			# X0 = np.mean(X, 0)
			# X_bar = ( X - np.tile(X0, (len, 1)) )
			# R1 = np.sum((X_bar[:,0:3] * X_bar[:, 0:1])**2, 0)
			# R2 = np.sum((X_bar[:,0:3] * X_bar[:, 1:2])**2, 0)
			# R3 = np.sum((X_bar[:,0:3] * X_bar[:, 2:3])**2, 0)
			#
			# W = np.vstack((R1,R2,R3))

			# print W

			im[currev_local[:, 1], currev_local[:, 0]] = clr
			clr += 10

		cv2.imshow('im', im)
		cv2.waitKey(1)
		tcurr += 0.5 - overlap_len
		jnew = np.argmin(abs(t-tcurr))

		# print j, jnew

		if jnew == j and overlap_len != time_window_len/2.:
			break
		elif overlap_len == time_window_len/2.:
			j += 1
		else:
			j = jnew


	time2 = time.time()
	print time2 - time1