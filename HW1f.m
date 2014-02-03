wsz=100;
hsz=60;
fs=16000;
wtype='h1';
nmfil=40;
ncep=13;
lfr=0;
hfr=0.5;



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

%x=wavread('record.wav');
psz = (length(x)-wsz)/hsz;
zpad = ceil(psz)*hsz + wsz - length(x);
x = [x;zeros( zpad, 1)];

if strcmp(wsz,'r')
    wt=ones(wsz,1);
elseif strcmp(wtype,'h1')
    wt=hanning(wsz);
elseif strcmp(wtype,'h2')
    wt=hamming(wsz);
else
    wt=hanning(wsz);
end

mf = zeros(wsz, (length(x)-wsz)/hsz);
for i=1:(length(x)-wsz)/hsz+1
    
    xcurr=x((i-1)*hsz+1:(i-1)*hsz+wsz,1);
    mf(:,i)=wt.*xcurr;
    
end

mfc=fft(mf,wsz);
mfc=mfc(1:floor(size(mfc,1)/2)+1,:);
mfc1=(abs(mfc)).*(abs(mfc));
mfc1=mfc1(2:end,:);
[melfb,mn,mx]=melbankm(nmfil,wsz,fs,lfr,hfr,'tz');
th=max(mfc1(:))*1e-20;
mfc2=log(max(melfb*mfc1,th));

%vlml=imagesc();
mfcc1=dct(mfc2);
mfcc1=mfcc1';
mfcc=mfcc1(:,1:ncep);





