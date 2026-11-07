%DESIGNPLOTS The front end's DSP design, as asked for in the course report: filter specification,
%   magnitude/phase response, pole-zero plot, Goertzel resonator response and the cost of the
%   Goertzel bank against an FFT. Run: designPlots
fs = 16000;

% --- High-pass filter: specification -> coefficients -> response ---------------------------
% Spec: Butterworth (maximally flat), order 2, fc = 100 Hz (-3 dB), fs = 16 kHz.
[b, a] = butter(2, 100 / (fs / 2), 'high');
fprintf('HPF coefficients  b = [%s]   a = [%s]\n', num2str(b, '%.6f '), num2str(a, '%.6f '));
figure('Name', 'High-pass filter', 'Color', 'w');
subplot(2, 2, [1 3]); [h, f] = freqz(b, a, 4096, fs);
semilogx(f(2:end), 20 * log10(abs(h(2:end)))); grid on; xlim([10 8000]); ylim([-60 3]);
xline(100, '--', 'f_c = 100 Hz'); xline(520, ':', '520 Hz alarm'); xlabel('Hz'); ylabel('dB');
title('Magnitude response');
subplot(2, 2, 2); plot(f, unwrap(angle(h)) * 180 / pi); grid on; xlabel('Hz'); ylabel('degrees');
xlim([0 1000]); title('Phase response');
subplot(2, 2, 4); zplane(b, a); title('Pole-zero plot: double zero at z = 1');

% --- Goertzel resonator ---------------------------------------------------------------------
% s[n] = x[n] + 2cos(w) s[n-1] - s[n-2]  ->  H(z) = 1 / (1 - 2cos(w) z^-1 + z^-2):
% poles at exp(+-jw) ON the unit circle (a marginally stable resonator, read out once per frame).
N = 320; f0 = 3100; w = 2 * pi * f0 / fs;
figure('Name', 'Goertzel resonator', 'Color', 'w');
subplot(1, 2, 1); zplane(1, [1 -2 * cos(w) 1]); title(sprintf('Goertzel pole-zero, %d Hz', f0));
subplot(1, 2, 2); hold on;
ff = linspace(2000, 4200, 2000); n = (0:N - 1).'; win = hann(N);
for fc = [3000 3100 3200]
    r = abs(sum(win .* exp(1j * 2 * pi * (ff - fc) .* n / fs), 1));
    plot(ff, 20 * log10(r / max(r)));
end
grid on; ylim([-60 3]); xlabel('Hz'); ylabel('dB'); title('Probe responses (N = 320, Hann)');
legend('3.0 kHz', '3.1 kHz', '3.2 kHz');

% --- Cost: Goertzel bank vs. FFT --------------------------------------------------------------
P = 1:64; nfft = 512;
figure('Name', 'Cost', 'Color', 'w');
plot(P, P * N, P, repmat(nfft * log2(nfft), size(P))); grid on;
xlabel('number of frequencies needed'); ylabel('multiplies per 20 ms frame');
legend('Goertzel bank (N per probe)', '512-point FFT (~N log_2 N)', 'Location', 'northwest');
title('The Goertzel bank is cheaper below ~14 probes; with the gate''s FFT already computed, its bins can be reused');
