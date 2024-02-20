webcamName = camSettings();
captureImages('D:\FYP\snapshots', webcamName);

function webcamName = camSettings()
    % Getting then printing a list of available webcams
    camList = webcamlist;
    disp(camList);

    % Ask to enter is camera name and check if that name is valid. Else,
    % try again.
    webcamName = "";
    validWebcam = false;
    while ~validWebcam
        % Ask to enter camera name
        disp("Enter Desired Webcam.")
        webcamName = input(">?", "s");
        validWebcam = ismember(webcamName, camList);
        if ~validWebcam
            disp("The name you have entered is not in the list of available webcams.")
        end
    end

    % Connect to Webcam
    cam = webcam(webcamName);

    % display webcam properties
    disp("Webcam Properties:")
    disp(cam)
end

function captureImages(savepath, webcamName)
    % Validate arguments
    arguments
        savepath (1,1) string = 'D:\FYP\snapshots'
        webcamName (1,1) string = 'HD Pro Webcam C920'
    end

    % Creating naming pattern for images
    nametemplate = 'snapshot_%05d.png';
    
    % Connect to Webcam
    cam = webcam(webcamName);
    
    % Create an endless session
    endSession = false;
    
    while ~endSession
    
        % Wait for a button or mouse press before continuing code execution
        w = waitforbuttonpress;
    
        % If the mouse has been clicked, end the current session.
        if w == 0
            disp("SESSION OVER!");
            break;
        end
    
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
    
        disp(filename);
    end
    
    % Disconnecting from Webcam and clearing variable
    clear cam;
end

