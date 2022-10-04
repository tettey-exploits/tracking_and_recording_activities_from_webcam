import cv2 as cv

# Get handle on webcam
cap = cv.VideoCapture(0)

# Import cascade classifier
faceCascade = cv.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
# video_name = input("Please enter a name of the video your want to save to: ")

# Prepare video recorder
video_recorder = cv.VideoWriter('video.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
video_recorder2 = cv.VideoWriter('video2.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
# Create tracker object
tracker = cv.TrackerCSRT_create()

# Tracker state flag
tracking_ongoing = False

# Variable to keep count of frames
frame_counter = 0


# For every frame of interest, apply face detection algorithm and track detected faces
def get_object(img):
    # Write text on screen
    faces = faceCascade.detectMultiScale(img, 1.1, 5)  # Apply facial classifier to image of interest

    # Loop through faces and draw bounding boxes around them
    for (x, y, w, h) in faces:
        # cv.circle(frame, (x + w // 2, y + h // 2), 2, (255, 0, 0), 2)
        # cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        break  # Because only one face is of interest to us

    if len(faces) > 0:
        tracker.init(frame, (x, y, w, h))
        return True
    else:
        return False


while True:

    _, frame = cap.read()
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if not tracking_ongoing or frame_counter > 60:
        tracking_ongoing = get_object(imgGray)
        message = "Searching"
        frame_counter = 0

    if tracking_ongoing:
        message = "Tracking"
        video_recorder2.write(frame)
        (success, box) = tracker.update(frame)

        if success:
            cv.circle(frame, (box[0] + box[1] // 2, box[2] + box[3] // 2), 2, (255, 0, 0), 2)
        else:
            tracking_ongoing = False

    cv.putText(frame, "Status: " + str(message), (40, 440), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    video_recorder.write(frame)  # Write frame to disk
    cv.imshow("Frame", frame)  # Display video on screen
    frame_counter = frame_counter + 1
    if cv.waitKey(1) == "s":
        break

cap.release()
video_recorder.release()

# Closes all the frames
cv.destroyAllWindows()

print("The video was successfully saved")
