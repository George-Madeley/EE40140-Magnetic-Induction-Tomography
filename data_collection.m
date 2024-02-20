% Getting then printing a list of available webcams
camList = webcamlist;
disp(camList);

% Connect to Webcam
cam = webcam('HD Pro Webcam C920');
preview(cam);