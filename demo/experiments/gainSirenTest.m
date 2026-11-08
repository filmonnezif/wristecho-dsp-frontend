function T = gainSirenTest()
%GAINSIRENTEST Does a fixed gain before the front end help low-level sirens wake the gate?
%   "17 dB" in the demo is the gate threshold Delta at the relaxed budget: how far a band's level must rise
%   above that band's own tracked noise floor (a relative level in dB). It is not dBFS, not SNR and not a
%   gain. A constant gain g raises a band's level and, once tracked, its floor by the same g dB, so the gate
%   statistic should not change; this experiment checks that, and what the gain does to the classifier.
%
%   Gain is applied to every sample of the input (sound + room noise) before the front end, then clipped to
%   [-1, 1]; recordings, classifier (EfficientAT + head) and thresholds (relaxed budget: Delta = 17 dB, H = 0)
%   are identical across gains. Sirens: the 8 ESC-50 test-fold siren clips at 10, 5 and 0 dB SNR.
%   Negatives: quiet room, microwave, truck beeper, 1 Hz beeper at 10 dB. Reports, per gain: siren trigger
%   coverage (share of windows overlapping the siren that the gate keeps awake), siren recall, gate wakes
%   outside any siren, false alerts on negatives, classifier calls and clipped samples.
%   Writes gainSirenTest.mat next to this file. Runs in about 3 min.
fs = 16000;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
api = WristEchoDemo('api');
P = api.defaults();
P.final.thresholds = P.budgets.thresholds(3, :); P.final.gate_delta = P.budgets.gate_delta(3);
P.final.gate_hang = P.budgets.gate_hang(3); P.delta = P.final.gate_delta;
gains = [0 3 6 12]; snrs = [10 5 0];
sirens = {'5-117118-A-42', '5-117120-A-42', '5-117122-A-42', '5-133989-A-42', '5-133989-B-42', ...
          '5-150409-A-42', '5-160551-A-42', '5-184323-A-42'};
negatives = {'Quiet room (nothing happens)', 'Microwave beep (look-alike)', ...
             'Truck reversing beeper (look-alike)', 'Continuous 1 Hz beeper 3.1 kHz (look-alike)'};
rows = [];
t0 = tic;
for i = 1:numel(sirens)
    f = fullfile(tempdir, ['wristecho_' sirens{i} '.wav']);
    if ~isfile(f)
        websave(f, ['https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/' sirens{i} '.wav']);
    end
    [y, f0] = audioread(f); s = resample(mean(y, 2), fs, f0);
    for snr = snrs
        [x, onset] = api.makeScene(s, snr, fs);
        stop = numel(x) / fs - 4;                              % makeScene: 4 s of room after the sound
        for g = gains
            rows = [rows; run(x, g, onset, stop, 'siren', snr)]; %#ok<AGROW>
        end
    end
end
for i = 1:numel(negatives)
    [x, ~] = api.makeScene(negatives{i}, 10, fs);
    for g = gains
        rows = [rows; run(x, g, NaN, NaN, 'negative', 10)]; %#ok<AGROW>
    end
end
T = struct2table(rows);
save(fullfile(here, 'gainSirenTest.mat'), 'T', 'gains', 'snrs', 'P');
fprintf('\n%5s | %8s %8s | %9s %9s | %10s %11s | %9s\n', 'gain', 'coverage', 'recall', 'wakes out', 'wakes neg', ...
    'false alrt', 'clf calls', 'clipped');
for g = gains
    sr = T(T.gain == g & T.kind == "siren", :); ng = T(T.gain == g & T.kind == "negative", :);
    fprintf('%+4d dB | %7.1f%% %7.1f%% | %9d %9d | %10d %11d | %8.4f%%\n', g, 100 * mean(sr.coverage), ...
        100 * mean(sr.recall), sum(sr.wakesOutside), sum(ng.wakesOutside), sum(ng.alerts), ...
        sum(T.calls(T.gain == g)), 100 * sum(T.clipped(T.gain == g)) / sum(T.samples(T.gain == g)));
end
for snr = snrs
    fprintf('  %2d dB SNR: recall by gain %s | coverage %s\n', snr, ...
        mat2str(arrayfun(@(g) round(100 * mean(T.recall(T.gain == g & T.snr == snr & T.kind == "siren"))), gains)), ...
        mat2str(arrayfun(@(g) round(100 * mean(T.coverage(T.gain == g & T.snr == snr & T.kind == "siren"))), gains)));
end
fprintf('(%.0f s)\n', toc(t0));

    function r = run(x, g, onset, stop, kind, snr)
        y = x * 10 ^ (g / 20);
        clipped = sum(abs(y) > 1);
        y = max(min(y, 1), -1);
        R = api.frontEnd(y, fs, P);
        tw = R.tw; ts = tw - P.win / fs;                     % window end / start (s)
        if isnan(onset)
            over = false(size(tw));
        else
            over = tw > onset & ts < stop;
        end
        hit = false;
        for k = 1:size(R.alerts, 1)
            hit = hit || (R.alerts{k, 2} == "SIREN" && R.alerts{k, 1} >= onset && R.alerts{k, 1} <= stop + 1);
        end
        r = struct('kind', string(kind), 'snr', snr, 'gain', g, 'coverage', mean(R.awake(over)), 'recall', hit, ...
            'wakesOutside', sum(R.awake(~over)), 'alerts', size(R.alerts, 1) * strcmp(kind, 'negative'), ...
            'calls', sum(R.awake), 'clipped', clipped, 'samples', numel(y));
        if ~any(over), r.coverage = NaN; end
    end
end
