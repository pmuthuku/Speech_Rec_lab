a=audiorecorder(16000,16,1);
ip=input('Hit r and enter on keyboard to talk ','s');
spkr=1;
if(strcmp(ip,'r'))
    
    pause(3);
    disp('Speak Now');
    record(a);
    pause(1.5);
    x=getaudiodata(a);
    if(log10(sum(x.*x))<-4)
        stop(a);
        disp('Recording stopped');
    end
    
    while (spkr)
        
        h=getaudiodata(a);
        x=h(end-8000:end,1);
        if(log10(sum(x.*x))<-5)
            stop(a)
            disp('End Detected');
            spkr=0;
        end
        
    end
else
    disp('Hit r to talk');
end
x=getaudiodata(a);
wavwrite(x,16000,16,'record.wav');
