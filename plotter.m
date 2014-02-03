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

h = figure;
mag_spec=load('recomp_cep.data');
mel_log_spec = idct(mag_spec',76);
mel_log_spec = mel_log_spec(1:39,:);
%mel_log_spec = -mel_log_spec;
imagesc(flipud(mel_log_spec));
title('Recomputed Mel Log Spectrum');
print(h, '-depsc', 'recomp_Mel_log_spec.eps');
clear mag_spec;
