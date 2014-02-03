clear; clc;

h = figure;
mag_spec=load('mag_spec.data');
imagesc(flipud(mag_spec'));
title('Magnitude Spectrum');
print(h, '-depsc', 'Mag_spec.eps');
clear mag_spec;


h = figure;
mag_spec=load('mel_spec.data');
imagesc(flipud(mag_spec'));
title('Mel Spectrum');
print(h, '-depsc', 'Mel_spec.eps');
clear mag_spec;


h = figure;
mag_spec=load('mel_log_spec.data');
imagesc(flipud(mag_spec'));
title('Mel Log Spectrum');
print(h, '-depsc', 'Mel_log_spec.eps');
clear mag_spec;
