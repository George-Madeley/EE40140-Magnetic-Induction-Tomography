% Setting Directory for snapshots
savepath = 'D:\FYP\snapshots';

% Creating naming pattern for images
nametemplate = 'snapshot_%05d.png';

% Getting then printing a list of available webcams
camList = webcamlist;
disp(camList);

% Connect to Webcam
cam = webcam('HD Pro Webcam C920');

for imNum = 1 : 5
    % Take snapshot
    img = snapshot(cam);
    
    % Create filename
    filename = sprintf(nametemplate, imNum);
    filename = fullfile(savepath, filename);

    % Save image
    imwrite(img, filename);
end

% Disconnecting from Webcam and clearing variable
clear cam;