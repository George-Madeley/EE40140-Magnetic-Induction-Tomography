function [cam, webcamName] = cameraSettings()
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

    % Display image
    preview(cam)

    % display webcam properties
    disp("Webcam Properties:")
    disp(cam)
end