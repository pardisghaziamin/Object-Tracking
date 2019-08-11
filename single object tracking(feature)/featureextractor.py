import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

detector= cv2.xfeatures2d.SIFT_create()
#detector=cv2.SIFT()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})


#sift = cv2.xfeatures2d.SIFT_create()

location=[]
MIN_MATCH_COUNT=10

for i in range(500,575):
    path='frame000'+("%03d"%i)+'.png'
    location.append(path)

images = [cv2.imread(file) for file in glob.glob("/home/mmm/itu/opencvtracker/feature/imges/*.png")]

#frame_name = location[1]
#img = cv2.imread(frame_name)
cv2.imshow("frame_name", images[0])
box = cv2.selectROI(img=images[0], fromCenter=False, showCrosshair=False)
imCrop = images[0][int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
trainImg=cv2.cvtColor(imCrop,cv2.COLOR_BGR2GRAY)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
#kp_1, desc_1 = sift.detectAndCompute(imCrop1, None)

cv2.destroyAllWindows()
QueryImgBGR=images[1]
imagesgray=cv2.cvtColor(images[1],cv2.COLOR_BGR2GRAY)
QueryImg=imagesgray
queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
matches=flann.knnMatch(queryDesc,trainDesc,k=2)

goodMatch=[]
for m,n in matches:
    if(m.distance<0.75*n.distance):
        goodMatch.append(m)
if(len(goodMatch)>MIN_MATCH_COUNT):
    tp=[]
    qp=[]
    for m in goodMatch:
        tp.append(trainKP[m.trainIdx].pt)
        qp.append(queryKP[m.queryIdx].pt)
    tp,qp=np.float32((tp,qp))
    H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
    h,w=trainImg.shape
    trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
    queryBorder=cv2.perspectiveTransform(trainBorder,H)
    cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    x1=queryBorder[0][0][0]
    x2=queryBorder[0][2][0]
    y1=queryBorder[0][0][1]
    y2=queryBorder[0][1][1]
    prev=QueryImg[int(y1):int(y2), int(x1):int(x2)]
    

else:
    print("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
cv2.imshow('result',QueryImgBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(2,len(images)):
    cv2.destroyAllWindows()
    trainKP,trainDesc=detector.detectAndCompute(prev,None)
    QueryImgBGR=images[i]
    imagesgray=cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
    QueryImg=imagesgray
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        trainBorder=[]
        queryBorder=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        x1=queryBorder[0][0][0]
        x2=queryBorder[0][2][0]
        y1=queryBorder[0][0][1]
        y2=queryBorder[0][1][1]
        prev=QueryImg[int(y1):int(y2), int(x1):int(x2)]
    else:
        print("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
    cv2.imshow('result',QueryImgBGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#if cv2.waitKey(10)==ord('q'):
    #break


#result = cv2.drawMatches(imCrop, kp_1, images[1], kp_2, good_points, None)
#cv2.imshow("result", result)
#cv2.waitKey(0)

#for i in range(0,len(images)):
    #cv2.destroyAllWindows()
    #cv2.imshow("frame_name", images[i])
    #cv2.waitKey(0)

    #box = cv2.selectROI("frame_name", img=images[i], fromCenter=False, showCrosshair=False)
    #imCrop = images[i][int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]

    #match_feature(img1=images[i],img2=imCrop)

def match_feature(img1,img2):

    kp1,des1=orb.detectAndCompute(img1,None)
    kp2,des2=orb.detectAndCompute(img2,None)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    matches=bf.match(des1,des2)
    matches=sorted(matches,key=lambda x:x.distance)
    img_out=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)

    plt.imshow(img_out)
    plt.show()


