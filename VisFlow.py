import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import cv2
from tqdm import tqdm
import scipy.io as sio

fname = open('/home/mhasek/PycharmProjects/data/events_all_test.p', 'rb')

events = pickle.load(fname)

time_window_len = 1.
spatial_window_len = 3
nr = 180
nc = 240
M = len(events['t'])

cnt = 0
th1 = 1e-5
th2 = 0.05

for event in events['XYPOLtms']:
	t = event[:, 3:4]
	xy = np.int_(event[:, 0:2])
	N = t.size
	spatial_temporal_w_idx = []

	imx = np.zeros((nr, nc))
	imy = np.zeros((nr, nc))

	for i in np.arange(N):
		eventidx = (abs(xy[:,0] - xy[i,0]) < spatial_window_len) & (abs(xy[:, 1] - xy[i, 1]) < spatial_window_len) & \
				(abs(t[:,0]-t[i,0]) < time_window_len/2.)

		if np.sum(eventidx) > 1:

			Current_event = np.hstack((xy[eventidx, :], t[eventidx,:]))

			print Current_event.shape

			xyt_bar = np.mean(Current_event, 0)
			xyt_mean_cent = Current_event - xyt_bar

			W = np.vstack((np.sum(xyt_mean_cent*xyt_mean_cent[:,0:1],0),np.sum(xyt_mean_cent*xyt_mean_cent[:,1:2],0),np.sum(xyt_mean_cent*xyt_mean_cent[:,2:3],0)))
			(E,V) = np.linalg.eig(W)
			PI = np.vstack((V[:,0:1],-np.sum(xyt_mean_cent*V[:,0])))

			err = 1e6
			S = Current_event.shape[0]

			while err > th1:
				errs = np.abs(np.matmul(PI.T, np.hstack((Current_event,np.ones((S,1)))).T))
				idx = errs.T < th2
				Current_event = Current_event[idx,:]

				S = Current_event.shape[0]
				Current_event = Current_event.reshape(S,3)

				xyt_bar = np.mean(Current_event, 0)
				xyt_mean_cent = Current_event - xyt_bar

				W = np.vstack((np.sum(xyt_mean_cent * xyt_mean_cent[:, 0:1], 0), np.sum(xyt_mean_cent * xyt_mean_cent[:, 1:2], 0), np.sum(xyt_mean_cent * xyt_mean_cent[:, 2:3], 0)))

				(E, V) = np.linalg.eig(W)
				PInew = np.vstack((V[:, 0:1], -np.sum(xyt_mean_cent * V[:, 0])))

				err = np.sqrt(np.sum((PInew - PI)**2))
				print err

			imx[xy[i, 1], xy[i, 0]] = -PI[2]/PI[0]
			imy[xy[i, 1], xy[i, 0]] = -PI[2]/PI[1]

		cv2.imshow('imx',imx)
		cv2.imshow('imy',imy)
		cv2.waitKey(1)

	cnt += 1

