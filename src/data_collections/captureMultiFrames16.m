%---------Function to capture single frame of data----------

function[data] = captureMultiFrames16(f, gain, nFrms)
    %disp("Frequency: " + num2str(f) + "   Scan: " + num2str(i) + "   Gain: " + num2str(gain))
    system(char(strcat('MIT_Multi_Frame', {' '}, num2str(f), {' '}, num2str(gain), {' '}, num2str(nFrms)))); 
    load data.csv;
    data=reshape(data,120,nFrms);
end
    
