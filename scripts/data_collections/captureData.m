function captureData()
    % CSV filepath
    CSVfilepath = 'C:\Users\Georg\OneDrive\Documents\Education\University\Semester 8\EE40150 Final Year Project\Code';
    CSVfilename = fullfile(CSVfilepath, 'data.csv');

    % If the CSV file does not exist, create a new file and write the
    % headers
    if ~isfile(CSVfilename)
        headers = {'bb_filename' 'cc_filename' 'shape' 'sample' 'gain' 'avg' 'freq' 'rtime' 'dtime'};

        % Create headers for the background readings
        prefix = 'bb_';
        length = numel(headers);
        N = 1:120;
        for i = 1:numel(N)
            headers{i + length} = [prefix int2str(N(i))];
        end

        % Create headers for the actual readings
        prefix = 'cc_';
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
    
    % Ask for shape of object
    disp("Enter Object Shape:");
    shape = input(">?", "s");

    % Ask for sample ID
    disp("Enter Sample ID (i.e., A, B, C etc):")
    sampleId = input(">?", "s");

    % Create an endless session
    endSession = false;

    % Set sensor properties
    frequency = 20000;
    gain = 7;
    avg = 5;

    % Measure background readings
    sensorData = zeros(120,1);
    sensorData = sensorData + captureMultiFrames16(frequency, gain, avg);
    bb = mean(sensorData, 2);
    bb(bb > 2e4) = 0;

    % capture the image
    bb_filename = captureImages('..\..\images', cam);
    
    while ~endSession
        % Wait for a button or mouse press before continuing code execution
        w = waitforbuttonpress;
    
        % If the mouse has been clicked, end the current session.
        if w == 0
            disp("SESSION OVER!");
            break;
        end

        for i = 1:5
            % start time
            rtime = tic;

            % Read data from sensors
            sensorData = zeros(120,1);
            sensorData = sensorData + captureMultiFrames16(frequency,gain,avg);
            cc = mean(sensorData, 2);
            cc(cc > 2e4) = 0;
    
            % capture the image
            cc_filename = captureImages('..\..\images\original', cam);

            % End time
            dtime = toc;
    
            % create the row of data
            row = {bb_filename cc_filename shape sampleId gain avg frequency rtime dtime};

            % Add background data to row
            length = numel(row);
            for i = 1:numel(bb)
                row{i + length} = [string(bb(i))];
            end

            % Add sample data to row
            length = numel(row);
            for i = 1:numel(cc)
                row{i + length} = [string(cc(i))];
            end
    
            % Write the row of data to the CSV file
            writecell(row, CSVfilename, 'WriteMode', 'append');
        end
        disp("NEXT!");
        beep;
    end
end