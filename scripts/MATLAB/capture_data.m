function capture_data()
    % CSV filepath
    CSVfilepath = 'C:\Users\Georg\OneDrive\Documents\Education\University\Semester 8\EE40150 Final Year Project\Code';
    CSVfilename = fullfile(CSVfilepath, 'data.csv');

    % If the CSV file does not exist, create a new file and write the
    % headers
    if ~isfile(CSVfilename)
        headers = {'filepath' 'shape'};
        writecell(headers, CSVfilename);
    end

    % Check if the camera object exists. If it does not, a connection to
    % the camera needs to be created.
    camExists = exist('cam', 'var');
    if ~camExists
        [cam, ~] = cam_settings();
    end

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
        
        % capture the image
        filename = capture_images('D:\FYP\snapshots', cam);

        % create the row of data
        row = {filename 'square'};

        % Write the row of data to the CSV file
        writecell(row, CSVfilename, 'WriteMode', 'append');
    end
end