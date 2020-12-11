# NGUYEN Anh Vu
# Face Detection Algorithm (HaarCascade) is provided by OpenCV.org, https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
# Face Recognition Algorithm is implemented by NGUYEN Anh Vu, based on the research paper of P. Belhumeur, J. Hespanha, and D. Kriegman.
# 'Eigenfaces vs.Fisherfaces: recognition using class specific linear projection', IEEE Trans. Pattern Anal. Mach. Intell., vol. 19, no. 7, pp. 711-720, 1997.

import cv2
import numpy as np

# Useful coefficients
# Height
H = 192
# Width
W = 168
# Number of class
NoC = 30
# Number of images per class
NoI = 15
# Dimension of each image
dimImg = 32256
# 50 largest eigen vectors
nb = 50

# Load HaarCascade Frontal Face File
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video Capture
cap = cv2.VideoCapture(0)

# Load matrices created from Matlab
M = np.loadtxt('meanImg.txt')
vectorL = np.loadtxt('vectorL.txt', delimiter=',')
PMeC = np.loadtxt('PMeC.txt', delimiter=',')
A = np.loadtxt('ImDatabase.txt', delimiter=',')
Wopt = np.loadtxt('Wopt.txt', delimiter=',')
Wopt_transposed = np.transpose(Wopt)

font = cv2.FONT_HERSHEY_SIMPLEX
Info = np.genfromtxt('Info.txt', dtype='str')

lastI = 0
#index = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    inputImg = np.zeros((H, W))
    posX = 0
    posY = 0

    for (x, y, w, h) in faces:
        posX = x
        posY = y
        cv2.rectangle(img, (x, y), (x+w, y+h), (20, 255, 57), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (192, 192))
        inputImg = roi_gray[:, 12:180]
        # Take picture
        #cv2.imwrite('vu_'+str(index)+'.png', inputImg)
        #print(index)
        #index = index + 1
        #
        cv2.imshow('ROI', inputImg)

    # Show Face Detection
    # Show information
    cv2.putText(img, Info[lastI], (posX, posY-10), font, 0.5, (20, 255, 57), 1, cv2.LINE_AA)
    cv2.imshow('img', img)

    # Face Recognition with method "Fisherfaces"
    inputImg = inputImg/255
    inputImg_db = np.reshape(inputImg, (H*W, 1), order="F")

    # Test
    # inputImg_db = np.copy(A[:, 0])

    for x in range(H*W):
        inputImg_db[x] = inputImg_db[x] - M[x]

    inputCoeff = np.matmul(np.transpose(inputImg_db), vectorL)
    inputCoeff = np.transpose(inputCoeff)
    ProjInputCoeff = np.matmul(Wopt_transposed, inputCoeff)

    # Result
    comp_Coeff = np.zeros(NoC)
    tmp_res = 0

    for i in range(NoC):
        for j in range(nb):
            tmp_res = tmp_res + (ProjInputCoeff[j] - PMeC[j, i]) **2
        comp_Coeff[i] = np.sqrt(tmp_res)
        tmp_res = 0

    MinCoeff = np.amin(comp_Coeff)
    I = np.argmin(comp_Coeff)
    print('MinCoeff = ', MinCoeff, ' ', ' I = ', I)
    lastI = I

    # Show Database Image
    Org = np.multiply(A[:, I*NoI], 255)
    # Org = np.round(Org)
    imgRes = np.array(Org, dtype=np.uint8)
    imgResReshaped = np.reshape(imgRes, (H, W), order="F")
    cv2.imshow('Database Image', imgResReshaped)




    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
