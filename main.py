import cv2

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg") #loads configuration
model = cv2.dnn_DetectionModel(net)  #model for detection
model.setInputParams(size = (320, 320), scale = 1/255)

#LOAD CLASSES 
classes  = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
print('objects list')
print(classes)


cap = cv2.VideoCapture(0) # capture webcam video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#window creation
cv2.namedWindow('Frame')



while True: 
    ret, frame = cap.read() #read capture webcam video
    
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        print(x, y, w, h)
        class_name = classes[class_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (204, 51, 99), 3) #frame capture configuration
        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (204, 51, 99), 2) #text for the frame capture
            
    print("class ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)
    
    cv2.imshow("Frame", frame) #opens capture
    key = cv2.waitKey(1) #holds cv2.imshow open
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()