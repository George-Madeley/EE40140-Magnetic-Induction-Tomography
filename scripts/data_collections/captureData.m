function captureData()
    % CSV filepath
    CSVfilepath = 'C:\Users\Georg\OneDrive\Documents\Education\University\Semester 8\EE40150 Final Year Project\Code';
    CSVfilename = fullfile(CSVfilepath, 'input_data.csv');

    % If the CSV file does not exist, create a new file and write the
    % headers
    if ~isfile(CSVfilename)
        prefix = 'sensor_';
        headers = {'filepath' 'shape'};
        length = numel(headers);
        N = 1:120;
        for i = 1:numel(N)
            headers{i + length} = [prefix int2str(N(i))];
        end
        writecell(headers, CSVfilename);
    end

    % Check if the camera object exists. If it does not, a connection to
    % the camera needs to be created.
    camExists = exist('cam', 'var');
    if ~camExists
        [cam, ~] = cameraSettings();
    end

    % Create an endless session
    endSession = false;

    % Set sensor properties
    frequency = 20000;
    gain = 7;
    avg = 3;

    % Measure background readings
    sensorData = zeros(120,1);
    sensorData = sensorData + captureMultiFrames16(frequency,gain,avg);
    bb = mean(sensorData, 2);
    bb(bb > 2e4) = 0;
    
    while ~endSession
        % Wait for a button or mouse press before continuing code execution
        w = waitforbuttonpress;
    
        % If the mouse has been clicked, end the current session.
        if w == 0
            disp("SESSION OVER!");
            break;
        end

        for i = 1:5
            tic
            % Read data from sensors
            sensorData = zeros(120,1);
            sensorData = sensorData + captureMultiFrames16(frequency,gain,avg);
            cc = mean(sensorData, 2);
            cc(cc > 2e4) = 0;
            dv = (cc-bb) ./ 1;
    
            % capture the image
            filename = captureImages('..\..\images', cam);
    
            % create the row of data
            row = {filename 'square'};
            length = numel(row);
            for i = 1:numel(dv)
                row{i + length} = [string(dv(i))];
            end
            toc
    
            % Write the row of data to the CSV file
            writecell(row, CSVfilename, 'WriteMode', 'append');
        end
        disp("NEXT!");
    end
end