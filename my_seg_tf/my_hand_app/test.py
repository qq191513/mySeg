import cv2
path ="/home/mo/work/seg_caps/my_hand_app/data/ddd.avi"
cap = cv2.VideoCapture(path)

if False == cap.isOpened():
	print('open video failed')
else:
	print('open video succeeded')






