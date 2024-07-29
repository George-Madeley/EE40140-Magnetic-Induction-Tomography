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

    % Check for saved camera settings
    if isfile("./cameraSettings.json")
        cameraSettingsJSON = jsondecode(fileread("cameraSettings.json"));
        if cameraSettingsJSON.Name == webcamName
            camProperties = cameraSettingsJSON.Properties;
            fields = fieldnames(camProperties);
            for i = 1:length(fields)
                key = fields{i};
                value = camProperties.(key);
                cam.(key) = value;
            end
        end
    end

    % display webcam properties
    disp("Webcam Properties:")
    disp(cam)
end