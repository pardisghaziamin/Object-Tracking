from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

# initializing global and local variables
#global trackers, tracker
location = []
name_array = []
ref_pnt = [ ]
label = str()
i = 0
k = 0
FRAMES_PATH = ".//frames//"

# place arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="goturn", help="OpenCV object tracker type")
args = vars(ap.parse_args())

# tracking objects (best one is "medianflow")
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    "goturn": cv2.TrackerGOTURN_create
}

# define a multi-tracker object and medianflow tracker
trackers = cv2.MultiTracker_create()
tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()

# store the frames to the location array
while (i < 10):
    path = FRAMES_PATH + 'frame00000' + str(i) + '.png'
    i = i + 1
    location.append(path)
while (i < 20):
    path = FRAMES_PATH + 'frame0000' + str(i) + '.png'
    i = i + 1
    location.append(path)

# read first image
frame_name = location[1]

img = cv2.imread(frame_name)
cv2.imshow(frame_name, img)

print("Image: ", format(frame_name))


# FUNCTION action after the mouse click
def click_event(event, x, y, flags, param):
    global ref_pnt

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pnt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        ref_pnt.append((x, y))


# FUNCTION Next Frame
def change_image(num):
    cv2.destroyAllWindows()

    # update the frame
    frame_name = location[num]

    img = cv2.imread(frame_name)
    cv2.imshow(frame_name, img)

    print("Image: ", format(frame_name))


# FUNCTION the box is selected
def rectangle_box(img, boxes, success, name_array):
    global returning_array, box_array

    if success:
        box_array = []
        returning_array = []

        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=(0,255,0), thickness=2)

            box_array.append(box)

        # put text on the box
        for i in range(0, len(box_array)):
            (x, y, w, h) = [int(v) for v in box_array[i]]
            cv2.putText(img=img, text=name_array[i], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
            print("{}: ".format(name_array[i]), x, y, w, h)

            file_array = []
            file_array.append(x)
            file_array.append(y)
            file_array.append(w)
            file_array.append(h)

            returning_array.append(file_array)

    else:
        print("nothing...")

    #return returning_array, box_array

# general Loop
while True:
    cv2.destroyAllWindows()
    img = cv2.imread(frame_name)

    (success, boxes) = trackers.update(img)

    # draw the rectangle if success is true
    rectangle_box(img=img, boxes=boxes, success=success, name_array=name_array)

    cv2.imshow(frame_name, img)
    print("Image: ", format(frame_name))

    key = cv2.waitKey(0) & 0xFF

    # select Roi
    if key == ord('s'):
        box = cv2.selectROI(frame_name, img=img, fromCenter=False, showCrosshair=False)

        # read label name for the box and append it to the array
        name = input('input label name: ')
        name_array.append(name)

        tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()

        # add the selected ROI to the tracking object
        trackers.add(tracker, img, box)

    # change image and save data to the text file
    elif key == ord('d'):
        output_path = ".//outputs//" + str(frame_name[11:-4]) + ".txt"

        # save to the text file
        with open(output_path, "w") as text_file:
            for i in range(0, len(box_array)):
                text_file.write('\n')
                text_file.write("%s %s" % (name_array[i], returning_array[i]))
            text_file.close()

        # change image
        k = k + 1
        frame_name = location[k]
        img = cv2.imread(frame_name)

    # re update the boxes
    elif key == ord('f'):
        cv2.destroyAllWindows()
        img = cv2.imread(frame_name)
        trackers = cv2.MultiTracker_create()
        tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
        box = cv2.selectROI(frame_name, img=img, fromCenter=False, showCrosshair=False)
        trackers.add(tracker, img, box)

    # if left mouse button is clicked the deleter specific selected box
    elif key == ord('r'):
        cv2.setMouseCallback(frame_name, click_event)

        if len(ref_pnt) == 2:
            print("clicked")
        else:
            print(len(ref_pnt))

    # break the loop
    elif key == ord('q'):
        break

# close all windows
cv2.destroyAllWindows()
