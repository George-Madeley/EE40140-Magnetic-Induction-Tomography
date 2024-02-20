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

    % Create a random id for the image
    imId = randi([0, 999999]);
    
    % Create filename
    filename = sprintf(nametemplate, imId);
    filename = fullfile(savepath, filename);

    % Assume the file with the same name already exists. Check if a file
    % with the same name acutally exists. If a file already exists with the
    % same name, create a new name an repeat. Else, exist out of the while
    % loop.
    fileExist = true;
    while fileExist
        if isfile(filename)
            % Change the file id
            imId = randi([0, 999999]);

            % Create filename
            filename = sprintf(nametemplate, imId);
            filename = fullfile(savepath, filename);
        else
            % A file does not exist therefore no overwrite will occur
            fileExist = false;
        end
    end

    % Save image
    imwrite(img, filename);
end

% Disconnecting from Webcam and clearing variable
clear cam;