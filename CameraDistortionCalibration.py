

import numpy as np
import cv2
import glob


# check calling or debug
if(len(sys.argv)>1):
    # argument
    # Ratate matrxi from WCS to Tool Center Point System (TCP) 
    row = sys.argv[0]
    # Ratate matrxi from TCP to Panel TCP 
    RE2 =  sys.argv[1]
    
else:
    row = 8
    col = 15    




# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((row *col,3), np.float32)
objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.bmp')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (col,row ),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (col,row ), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# ret: 重投影誤差
# mtx: 相機內參矩陣
# dist: 相機畸變參數
# rvecs: 



print('mtx')
print(mtx)
print('dist')
print(dist)


img = cv2.imread('V630SB3250001__Cam0_Distortion_20230620190740_LHDR_G2Flx7.36500_MaxGray_34.623.bmp')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)