"""
Description
@file dynamicTrafficManagement.py
The dataset for this model is taken from the Universitetsboulevarden,
Aalborg, Denmark
This file calculate the density of the traffic and set the value of the timer
for each four lane
"""

"""
# Import statements #
"""
import cv2
import numpy as np
from sklearn.externals import joblib
import threading
import time
import yolo_main
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# importing linearRegression model pickle file#
model = joblib.load("timePredictionModel.cpickle")

# Calculating the frame number from the given time as an argument#
def calcFrame(x, y):
    frame_time = int((x * 60 + y) * 35)
    return frame_time

congestionTest = 0
x = 2
i=j=k=l=0
def process(frame,lane):
    global x
    if lane == 1:
        if x == 2:
            x=1
        else:
            x=2
    vidClone = frame.copy()
    #cv2.imwrite('image.png',vidClone)
    filepath = 'TrafficImages/lane'+str(lane)+'/image'+str(x)+'.png'
    
    print("entered for processing")
    #Finding the roi#
    roi=np.zeros((frame.shape[0],frame.shape[1]),"uint8")
    cv2.rectangle(roi, (65, 60), (241, 187), 255, -1)
    frame=cv2.bitwise_and(frame,frame,mask=roi)
    cv2.imwrite(filepath,frame)
    global congestionTest
    congestionTest=frame.copy()
    congestionTest=cv2.cvtColor(congestionTest,cv2.COLOR_BGR2GRAY)
    #Yolo Logic#
    num=yolo_main.detect(frame)
    print("detected vehicles",num)
    arr=np.array(num)
    arr=arr.reshape(-1,1)

    #Obtaining the time#
    time = int(model.predict(arr))
    if(time<10):
        time=10
    print("predicted time is",time)

    
    print("completed processing")
    return int(time)

#Main function for input and output displaying#

def get_frame():

    global i,j,k,l
    vid1 = cv2.VideoCapture('TrafficDataSet.mp4')
    vid2 = cv2.VideoCapture('TrafficDataSet.mp4')
    vid3 = cv2.VideoCapture('TrafficDataSet.mp4')
    vid4 = cv2.VideoCapture('TrafficDataSet.mp4')
    flag1=[0,0]
    flag2=[0,0]
    flag3=[0,0]
    flag4=[0,0]
    _, frame1 = vid2.read()
    temp = np.zeros(frame1.shape,"uint8")
    
    li=[[2,37],[2,52],[4,1],[7,30],[8,51],[10,8],[11,8],[12,21],[14,6],[15,34]]
    index=0
    red_img=cv2.imread("traffic_lights/red.png")
    yellow_img=cv2.imread("traffic_lights/yellow.png")
    green_img=cv2.imread("traffic_lights/green.png")

    #------------_DEBUG3-------------------#
    red_img=cv2.resize(red_img,(100,200),None)
    yellow_img=cv2.resize(yellow_img,(100,200),None)
    green_img=cv2.resize(green_img,(100,200),None)

    while True:
 
        # setting the video frame for different lanes#
        #For lane1 #

        lane1_start_time = calcFrame(li[index][0],li[index][1] )
        print("index",index)
        vid1.set(1, lane1_start_time)
        _, frame1 = vid1.read()
        
        
        index=(index+1)%9
        #For lane2 #
        lane2_start_time = calcFrame(li[index][0],li[index][1])
        print("index",index)
        vid2.set(1, lane2_start_time)
        _, frame2 = vid2.read()

        index=(index+1)%9
        #For lane3#
        lane3_start_time = calcFrame(li[index][0],li[index][1])
        print("index",index)
        vid3.set(1, lane3_start_time)
        _, frame3 = vid3.read()

        index=(index+1)%9
        #For lane4#
        lane4_start_time = calcFrame(li[index][0],li[index][1])
        print("index",index)
        vid4.set(1, lane4_start_time)
        _, frame4 = vid4.read()

        index=(index+1)%9
        # display window. fWin is the final Video#
        # st0 = np.hstack((temp, frame1, temp))
        # st1 = np.hstack((frame4, temp, frame2))
        # st2 = np.hstack((temp, frame3, temp))
        # fWin = np.vstack((st0, st1, st2))

        #------------DEBUG1-----------#
        #print(temp.shape,st0.shape,red_img.shape)
        next_predected_time = 0
        if next_predected_time == 0:
            predected_time = int(process(frame1,1))
        else:
            predected_time = int(next_predected_time)

        #print("predicted time is",predected_time)
        t0 = time.clock()
        t0 = time.time()

        while (time.time()-t0<=predected_time):
            rem_time=predected_time-(time.time()-t0)
            print("frame 1")
            _, frame1 = vid1.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, temp, frame2))
            st2 = np.hstack((temp, frame3, temp))

            if rem_time<7:
                testing = np.hstack((yellow_img,red_img,red_img,red_img))
            else:
                testing = np.hstack((green_img,red_img,red_img,red_img))

              #------------DEBUG2-----------#
            
            testing=cv2.resize(testing,(st0.shape[1],st0.shape[0]+200),None)

            fWin = np.vstack((st0,st1,st2,testing))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 1:', (x-280, y-50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(int(rem_time)), (x - 200, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))


            imgencode=cv2.imencode('.jpg',fWin)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')


            #cv2.imshow("frame", fWin)
            #cv2.waitKey(1)
            
            if int(rem_time)==5:
                print("processing frame 2")
                next_predected_time=(process(frame2,2))
            print(rem_time)
            if int(rem_time)==4:
                roi=np.zeros((frame1.shape[0],frame1.shape[1]),"uint8")
                cv2.rectangle(roi, (65, 60), (241, 187), 255, -1)
                frame1=cv2.bitwise_and(frame1,frame1,mask=roi) 
                print("Checking for congestion")
                filepath = 'TrafficImages/lane1/congestionCheck.png'
                cv2.imwrite(filepath,frame1)
                congestionCheck=frame1.copy()
                congestionCheck=cv2.cvtColor(congestionCheck,cv2.COLOR_BGR2GRAY)
                if np.mean(cv2.bitwise_xor(congestionCheck,congestionTest)<4):
                    flag1[i]=1
                    i=(i+1)%2
                if flag1[0]==flag1[1] and flag1[1]==1:
                    print("Unwanted Congestion detected")
                    pass


        predected_time=next_predected_time


        #For Frame2#
        t0 = time.clock()
        t0 = time.time()
        next_predected_time = 0
        while (time.time() - t0 <= predected_time):
            rem_time = predected_time - (time.time() - t0)
            print("frame 2")
            ret2, frame2 = vid2.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, temp, frame2))
            st2 = np.hstack((temp, frame3, temp))
            

            #-- Traffic Lights --#
            if rem_time<7:
                testing = np.hstack((red_img,yellow_img,red_img,red_img))
            else:
                testing = np.hstack((red_img,green_img,red_img,red_img))

            
            testing=cv2.resize(testing,(st0.shape[1],st0.shape[0]+200),None)
            fWin = np.vstack((st0,st1,st2,testing))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 2:', (x - 280, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(int(rem_time)), (x - 200, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            
            imgencode=cv2.imencode('.jpg',fWin)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

            
            if int(rem_time) == 5:
                print("processing frame3")
                next_predected_time = (process(frame3,3))
            if int(rem_time)==4:
                roi=np.zeros((frame2.shape[0],frame2.shape[1]),"uint8")
                cv2.rectangle(roi, (65, 60), (241, 187), 255, -1)
                frame2=cv2.bitwise_and(frame2,frame2,mask=roi) 
                print("Checking for congestion")
                filepath = 'TrafficImages/lane2/congestionCheck.png'
                cv2.imwrite(filepath,frame2)
                congestionCheck=frame2.copy()
                congestionCheck=cv2.cvtColor(congestionCheck,cv2.COLOR_BGR2GRAY)
                if np.mean(cv2.bitwise_xor(congestionCheck,congestionTest)<4):
                    flag1[j]=1
                    j=(j+1)%2  
                if flag2[0]==flag2[1] and flag2[1]==1:
                    print("Unwanted Congestion detected")
                    pass       
            print(rem_time)


        predected_time=next_predected_time


        #For Frame3#
        t0 = time.clock()
        t0 = time.time()
        next_predected_time = 0
        while (time.time() - t0 <= predected_time):

            rem_time = predected_time - (time.time() - t0)
            print("frame 3")
            ret2, frame3 = vid3.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, temp, frame2))
            st2 = np.hstack((temp, frame3, temp))

            if rem_time<7:
                testing = np.hstack((red_img,red_img,yellow_img,red_img))
            else:
                testing = np.hstack((red_img,red_img,green_img,red_img))

            testing=cv2.resize(testing,(st0.shape[1],st0.shape[0]+200),None)
            fWin = np.vstack((st0, st1, st2,testing))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 3:', (x - 280, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(int(rem_time)), (x - 200, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            imgencode=cv2.imencode('.jpg',fWin)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

            
            if int(rem_time) == 5:
                print("processing frame4")
                next_predected_time = process(frame4,4)
            if int(rem_time)==4:
                roi=np.zeros((frame3.shape[0],frame3.shape[1]),"uint8")
                cv2.rectangle(roi, (65, 60), (241, 187), 255, -1)
                frame3=cv2.bitwise_and(frame3,frame3,mask=roi) 
                print("Checking for congestion")
                filepath = 'TrafficImages/lane3/congestionCheck.png'
                cv2.imwrite(filepath,frame3)
                congestionCheck=frame3.copy()
                congestionCheck=cv2.cvtColor(congestionCheck,cv2.COLOR_BGR2GRAY)
                if np.mean(cv2.bitwise_xor(congestionCheck,congestionTest)<4):
                    flag1[k]=1
                    k=(k+1)%2     
                if flag3[0]==flag3[1] and flag3[1]==1:
                    print("Unwanted Congestion detected")
                    pass        
            print(rem_time)


        predected_time=next_predected_time

        #For Frame4#
        t0 = time.clock()
        t0 = time.time()
        next_predected_time = 0
        while (time.time() - t0 <= predected_time):
            rem_time = predected_time - (time.time() - t0)
            print("frame 4")
            ret2, frame4 = vid4.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, temp, frame2))
            st2 = np.hstack((temp, frame3, temp))
            if rem_time<7:
                testing = np.hstack((red_img,red_img,red_img,yellow_img))
            else:
                testing = np.hstack((red_img,red_img,red_img,green_img))

            testing=cv2.resize(testing,(st0.shape[1],st0.shape[0]+200),None)
            fWin = np.vstack((st0, st1, st2,testing))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 4:', (x - 280, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(int(rem_time)), (x - 200, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            imgencode=cv2.imencode('.jpg',fWin)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

            rem_time = predected_time - (time.time() - t0)
            if int(rem_time) == 5:
                print("processing frame 1")
                next_predected_time = process(frame1,1)
            if int(rem_time)==4:
                roi=np.zeros((frame4.shape[0],frame4.shape[1]),"uint8")
                cv2.rectangle(roi, (65, 60), (241, 187), 255, -1)
                frame4=cv2.bitwise_and(frame4,frame4,mask=roi) 
                print("Checking for congestion")
                filepath = 'TrafficImages/lane4/congestionCheck.png'
                cv2.imwrite(filepath,frame2)
                congestionCheck=frame4.copy()
                congestionCheck=cv2.cvtColor(congestionCheck,cv2.COLOR_BGR2GRAY)
                if np.mean(cv2.bitwise_xor(congestionCheck,congestionTest)<4):
                    flag1[l]=1
                    l=(l+1)%2
                if flag4[0]==flag4[1] and flag4[1]==1:
                    print("Unwanted Congestion detected")
                    pass
            print(rem_time)

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
