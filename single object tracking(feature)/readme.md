
# **Single object Tracking using Feature extraction**

## Purpose :

we are going to track object; actually in this case, object is car.


![
](https://lh3.googleusercontent.com/ZfnbX3tESAV0JrPYWeisgvqBNj_ZQUhGLMPP4svVY3Qr4aCAgI35Iicg-2yc7jPina5ttodolyU "car")

## First

Import libraries:

-   we need  **opencv**  for working with images
-   we need  **numpy**  for working with arrays
-   we need  **matplotlib**  for draw any line on the image  
 `import numpy`  
 `import cv2`
  `from matplotlib import pyplot as plt`

## Second
This function returns a list of files that match the given pattern in pathname parameter. The pathname can be absolute or relative. It can also include wild cards like *

    import glob
    images = [cv2.imread(file) for file in glob.glob("/home/mmm/itu/opencvtracker/feature/imges/*.png")]

Following code above reads all file in current directory with ‘.png’ extension.

## Third

let user select the box on the first image and crop it :

    box = cv2.selectROI(img=images[0], fromCenter=False, showCrosshair=False)
    
	imCrop = images[0][int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
## Forth
and next we used the feature extractor to detect features and stored them in two variables one is trainKP which is the list of key points / coordinates of the features, and other in trainDesc which is list of descriptions of the corresponding key points

    trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

We will need these to to find visually similar objects in our next frames,
##  Start The Main LOOP!!
Now that we are done with all the preparation, we can start the main loop to start doing the main work

![
](https://lh3.googleusercontent.com/xXKUGyKk_E7RpGnj8iryo3xV9W00p4vR6hcmrvRADZHaPIATpYx_d4B07sRCrLyDKpXEwmZXH8k "m")


    for i in range(2,len(images)):
	    cv2.destroyAllWindows()
	    trainKP,trainDesc=detector.detectAndCompute(prev,None)
	    QueryImgBGR=images[i]
	    imagesgray=cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
	    QueryImg=imagesgray
	    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
	    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
In the above code, we first read next frames one by one, then converted it to gray scale, then we extracted the features like we did in the training image,after that we used the **_flann_** feature matcher to match the features in both images, and stored the matches results in **_matches_** variable

here flann is using knn to match the features with k=2, so we will get 2 neighbors
## Five
After this we have to filter the matches to avoid false matches

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)

In the above code we created an empty list named **_goodMatch_** and the we are checking distance from the most neatest neighbor m and the next neatest neighbor n, and we are considering the match is a good match if the distance from point _**“m”**_is less the 70% of the distance on point “_**n**_” and appending that point to _**“goodMatch”**_

## Six

We also need to make sure that we have enough feature matches to call these a match, for that we are going to set a threshold “_**MIN_MATCH_COUNT**_” and if the number of matches are greater than then value then only we are going to consider them as match

    MIN_MATCH_COUNT=10
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

So in the above code we first check if number of matched features are more than the minimum number of threshold then we are going to do further operation,

Now we created two empty to get the coordinates of the matched features from the training image as well as from the query image, and converted that to numpy lists

the we used cv2.findHomography(tp,qp,cv2.RANSAC,3.0) to find transformation constant to translate points from training points to query image points,

now we want to draw border around the object so we want to get the coordinates of the border corners from the training image, which are (0,0), (0,h-1), (w-1,h-1),(w-1,0) where h,w is the height and width of the training image

Now using the transformation constant “H” that we got from earlier we will translate the coordinates from training image to query image,

finally we are using “cv2.polylines()” to draw the borders in the query image

Lastly if the number of features are less that the minimum match counts then we are going to print it in the screen in the else part
## Result

Now lets display the image, and close the window if loop ends

    cv2.imshow('result',QueryImgBGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
![
](https://lh3.googleusercontent.com/ZfnbX3tESAV0JrPYWeisgvqBNj_ZQUhGLMPP4svVY3Qr4aCAgI35Iicg-2yc7jPina5ttodolyU "b")