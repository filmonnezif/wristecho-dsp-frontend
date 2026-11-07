function WristEchoDemo(mode, extra)
%WRISTECHODEMO Live demo of the DSP wake-up front end ("Listen Before You Classify").
%   WristEchoDemo('check') runs every sound without the UI and prints what the proposed system and
%   always-on YAMNet report (WristEchoDemo('check', urls) adds extra recordings by URL).
%   Pick a sound, set the room noise and the front-end parameters, press Play. The app mixes the
%   sound into room noise, runs the DSP front end and YAMNet exactly as in the paper, and replays
%   the result in real time: spectrogram, gate statistic vs. threshold, when YAMNet is awake,
%   Goertzel probe energies, YAMNet's category scores and the alert.
%
%   Requires Audio Toolbox, Signal Processing Toolbox and Deep Learning Toolbox (YAMNet via
%   audioPretrainedNetwork). Sound files are downloaded once into tempdir (see demoSounds).

fs = 16000;
P = defaults();
if nargin && strcmp(mode, 'check')
    if nargin < 2, extra = {}; end
    checkAll(P, fs, extra);
    return
end
S = struct('net', [], 'classes', [], 'scene', [], 'res', [], 'stop', false);

fig = uifigure('Name', 'Listen Before You Classify - live demo', 'Position', [60 60 1280 760], 'Color', [1 1 1]);
g = uigridlayout(fig, [1 2], 'ColumnWidth', {300, '1x'}, 'Padding', [10 10 10 10]);

% --- Controls -------------------------------------------------------------------------------
c = uigridlayout(g, [19 1], 'RowHeight', repmat({'fit'}, 1, 19), 'RowSpacing', 6);
uilabel(c, 'Text', 'Sound', 'FontWeight', 'bold');
names = demoSounds();
ddSound = uidropdown(c, 'Items', names, 'Value', names{2});
uilabel(c, 'Text', 'Room noise (SNR of the sound, dB)');
slSnr = uislider(c, 'Limits', [-5 30], 'Value', 10, 'MajorTicks', -5:5:30);
cbGate = uicheckbox(c, 'Text', 'Wake-up gate on (off = YAMNet on every window)', 'Value', true);
uilabel(c, 'Text', 'Gate threshold \Delta (dB above the noise floor)', 'Interpreter', 'tex');
slDelta = uislider(c, 'Limits', [5 30], 'Value', P.delta, 'MajorTicks', 5:5:30);
uilabel(c, 'Text', 'Hangover H (s)');
slHang = uislider(c, 'Limits', [0 3], 'Value', P.hang, 'MajorTicks', 0:1:3);
cbDsp = uicheckbox(c, 'Text', 'DSP alarm detector (Goertzel + T-3)', 'Value', true);
cbStrict = uicheckbox(c, 'Text', 'Strict T-3 (needs the 1.5 s pause)', 'Value', false);
cbVeto = uicheckbox(c, 'Text', 'Siren veto (fire needs fire score > siren score)', 'Value', true);
bPlay = uibutton(c, 'Text', 'Play', 'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', [0 .62 .45], 'FontColor', 'w');
bStop = uibutton(c, 'Text', 'Stop');
lamp = uilamp(c, 'Color', [.6 .6 .6]);
lblAwake = uilabel(c, 'Text', 'YAMNet: asleep', 'FontSize', 14);
lblSaved = uilabel(c, 'Text', 'Classifier compute saved: -', 'FontSize', 14, 'FontWeight', 'bold', 'WordWrap', 'on');
alert = uilabel(c, 'Text', 'no alert', 'FontSize', 26, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'BackgroundColor', [.93 .93 .93], 'FontColor', [.4 .4 .4]);
lblLog = uilabel(c, 'Text', '', 'WordWrap', 'on', 'FontSize', 11);

% --- Plots ----------------------------------------------------------------------------------
p = uigridlayout(g, [3 2], 'RowHeight', {'1.3x', '1x', '1x'}, 'ColumnWidth', {'2x', '1x'});
axSpec = uiaxes(p); axSpec.Layout.Column = [1 2]; title(axSpec, 'Spectrogram');
axGate = uiaxes(p); axGate.Layout.Column = [1 2]; title(axGate, 'Gate statistic (max band rise above its floor)');
axProbe = uiaxes(p); title(axProbe, 'Goertzel probes (share of frame energy)');
axYam = uiaxes(p); title(axYam, 'YAMNet category scores');

bPlay.ButtonPushedFcn = @(~, ~) onPlay();
bStop.ButtonPushedFcn = @(~, ~) setStop();

    function setStop()
        S.stop = true;
    end

    function onPlay()
        S.stop = false;
        bPlay.Enable = 'off';
        lblLog.Text = 'Preparing...';
        drawnow;
        if isempty(S.net)
            [S.net, S.classes] = audioPretrainedNetwork("yamnet");
        end
        P.delta = slDelta.Value; P.hang = slHang.Value; P.gate = cbGate.Value;
        P.dsp = cbDsp.Value; P.strict = cbStrict.Value; P.veto = cbVeto.Value;
        [x, onset] = makeScene(ddSound.Value, slSnr.Value, fs);
        R = frontEnd(x, fs, P, S.net, S.classes);
        animate(x, onset, R);
        bPlay.Enable = 'on';
    end

    function animate(x, onset, R)
        tf = (1:numel(R.nov)) * P.frame / fs;
        tw = R.tw;
        % static parts: full spectrogram and gate statistic, a cursor moves over them
        [~, fsp, tsp, ps] = spectrogram(x, hann(512), 384, 512, fs);
        db = 10 * log10(ps + 1e-14);
        imagesc(axSpec, tsp, fsp / 1000, db); axis(axSpec, 'xy');
        colormap(axSpec, 'turbo'); clim(axSpec, [prctile(db(:), 5), max(db(:))]); ylabel(axSpec, 'kHz');
        hold(axSpec, 'on'); cs = xline(axSpec, 0, 'w', 'LineWidth', 1.5);
        xline(axSpec, onset, '--w', 'sound starts'); hold(axSpec, 'off');
        cla(axGate); hold(axGate, 'on');
        area(axGate, tw, R.awake * max(40, max(R.novw)), 'FaceColor', [0 .62 .45], 'FaceAlpha', .15, 'EdgeColor', 'none');
        plot(axGate, tf, R.nov, 'Color', [.4 .4 .4]);
        yline(axGate, P.delta, '--k', sprintf('\\Delta = %.0f dB', P.delta));
        cg = xline(axGate, 0, 'r', 'LineWidth', 1.5);
        for k = 1:size(R.alerts, 1)
            xline(axGate, R.alerts{k, 1}, '-', R.alerts{k, 2}, 'Color', R.alerts{k, 3}, 'LineWidth', 2);
        end
        hold(axGate, 'off'); xlim(axGate, [0 tf(end)]); ylim(axGate, [0 max(40, max(R.novw))]);
        xlabel(axGate, 'time (s)'); ylabel(axGate, 'dB');
        saved = 100 * (1 - mean(R.awake));
        lblSaved.Text = sprintf('Classifier compute saved: %.0f %%  (%d of %d windows classified)', ...
            saved, sum(R.awake), numel(R.awake));
        lblLog.Text = R.log;
        alert.Text = 'no alert'; alert.BackgroundColor = [.93 .93 .93]; alert.FontColor = [.4 .4 .4];
        player = audioplayer(x / max(abs(x)) * 0.9, fs);
        play(player); t0 = tic;
        k = 0;
        while toc(t0) < numel(x) / fs && ~S.stop
            t = toc(t0);
            cs.Value = t; cg.Value = t;
            kw = find(tw <= t, 1, 'last');
            if ~isempty(kw) && kw ~= k
                k = kw;
                if R.awake(k)
                    lamp.Color = [0 .8 .3]; lblAwake.Text = 'YAMNet: AWAKE';
                else
                    lamp.Color = [.6 .6 .6]; lblAwake.Text = 'YAMNet: asleep';
                end
                bar(axYam, categorical(R.cats, R.cats), R.catScore(k, :) .* R.awake(k), 'FaceColor', [.2 .4 .8]);
                ylim(axYam, [0 1]); yline(axYam, 0.3, '--', 'threshold');
                fr = max(1, min(numel(R.nov), round(t * fs / P.frame)));
                bar(axProbe, R.probes / 1000, max(R.probeShare(max(1, fr - 24):fr, :), [], 1), 'FaceColor', [.84 .37 0]);
                ylim(axProbe, [0 1]); yline(axProbe, R.ratio, '--', 'tone share'); xlabel(axProbe, 'kHz');
                a = find([R.alerts{:, 1}] <= t, 1, 'last');
                if ~isempty(a)
                    alert.Text = upper(R.alerts{a, 2}); alert.BackgroundColor = R.alerts{a, 3}; alert.FontColor = 'w';
                end
            end
            drawnow limitrate;
            pause(0.03);
        end
        stop(player);
    end
end

% =================================================================================================
function P = defaults()
% The paper's settings: F4 at the 5 % budget (Delta = 18 dB, H = 0 s) and detector v4.
P = struct('frame', 320, 'nfft', 512, 'hop', 8000, 'win', 15600, 'nbands', 16, 'bandLo', 100, 'bandHi', 7600, ...
    'smooth', 0.06, 'rise', 3, 'delta', 18, 'hang', 0, 'gate', true, 'dsp', true, 'strict', false, 'veto', true, ...
    'probeLo', 2800, 'probeHi', 3600, 'ratio', 0.15, 'square', [520 1560 2600], 'thr', 0.30);
end

function R = frontEnd(x, fs, P, net, classes)
%FRONTEND The DSP front end and YAMNet on audio x (column), as in the paper.
% Gate path: HPF -> 20 ms frames -> |FFT|^2 in 16 log bands -> one-pole smoothing -> floor -> novelty
[b, a] = butter(2, 100 / (fs / 2), 'high');           % spec: Butterworth, order 2, fc = 100 Hz
y = filter(b, a, x);
n = floor(numel(y) / P.frame);
fr = reshape(y(1:n * P.frame), P.frame, n).';          % frames x samples
win = hann(P.frame).';
X = abs(fft(fr .* win, P.nfft, 2)) .^ 2;
X = X(:, 1:P.nfft / 2 + 1);
f = (0:P.nfft / 2) * fs / P.nfft;
edges = logspace(log10(P.bandLo), log10(P.bandHi), P.nbands + 1);
B = zeros(n, P.nbands);
for k = 1:P.nbands
    B(:, k) = sum(X(:, f >= edges(k) & f < edges(k + 1)), 2);
end
al = exp(-P.frame / fs / P.smooth);
E = 10 * log10(filter(1 - al, [1 -al], B) + 1e-10);    % one-pole low-pass per band, in dB
F = zeros(size(E)); fl = min(E(1:min(50, n), :), [], 1); rise = P.rise * P.frame / fs;
for t = 1:n
    fl = min(E(t, :), fl + rise);                       % floor: follows down at once, rises <= 3 dB/s
    F(t, :) = fl;
end
R.nov = max(E - F, [], 2);
nw = floor((numel(x) - P.win) / P.hop) + 1;
R.tw = (P.win + (0:nw - 1) * P.hop) / fs;              % time at which each window ends
R.novw = zeros(1, nw);
for k = 1:nw
    e = floor((P.win + (k - 1) * P.hop) / P.frame);
    R.novw(k) = max(R.nov(max(1, e - P.hop / P.frame + 1):e));
end
trig = R.novw > P.delta;
R.awake = conv(double(trig), ones(1, round(P.hang / (P.hop / fs)) + 1));
R.awake = R.awake(1:nw) > 0;
if ~P.gate, R.awake = true(1, nw); end

% YAMNet on every window (only the awake ones are used; computing all lets the app compare)
R.cats = {'Fire alarm', 'Siren', 'Door'};
lab = {{"Smoke detector, smoke alarm", "Fire alarm", "Alarm"}, ...
       {"Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Emergency vehicle"}, ...
       {"Knock", "Doorbell", "Ding-dong"}};
veto = ismember(string(classes), ["Ringtone", "Telephone", "Telephone bell ringing"]);
R.catScore = zeros(nw, 3); vs = zeros(nw, 1);
for k = 1:nw
    seg = x((k - 1) * P.hop + (1:P.win));
    s = mean(predict(net, yamnetPreprocess(seg, fs)), 1);
    for j = 1:3
        R.catScore(k, j) = max(s(ismember(string(classes), string(lab{j}))));
    end
    vs(k) = max(s(veto));
end

% Alarm path: Goertzel resonator bank -> tone track -> T-3 pattern
R.probes = P.probeLo:50:P.probeHi; R.ratio = P.ratio;
xf = reshape(x(1:n * P.frame), P.frame, n).' .* win;
total = sum(xf .^ 2, 2) * P.frame / 2 + 1e-12;
Eg = goertzelBank(xf, [R.probes P.square], fs);
R.probeShare = Eg(:, 1:numel(R.probes)) ./ total;
piezo = max(Eg(:, 1:numel(R.probes)), [], 2); sq = sum(Eg(:, numel(R.probes) + 1:end), 2);
loud = 10 * log10(total / P.frame) > -75;
on = ((piezo ./ total > P.ratio & envOk(piezo)) | (sq ./ total > 0.35 & envOk(sq))) & loud;
dspT = t3Times(on, P.frame / fs, P.strict);
if ~P.dsp, dspT = []; end

R.alerts = alertRule(R.tw, R.awake, R.catScore, vs, dspT, P.veto, P.thr, R.cats);
F0 = alertRule(R.tw, true(1, nw), R.catScore, vs, [], false, P.thr, R.cats);
R.f0 = F0;
logTxt = "This system:" + newline;
for k = 1:size(R.alerts, 1)
    logTxt = logTxt + sprintf("  %.1f s  %s\n", R.alerts{k, 1}, R.alerts{k, 2});
end
if isempty(R.alerts), logTxt = logTxt + "  no alert" + newline; end
logTxt = logTxt + "Always-on YAMNet (no gate, no DSP, no veto):" + newline;
for k = 1:size(F0, 1)
    logTxt = logTxt + sprintf("  %.1f s  %s\n", F0{k, 1}, F0{k, 2});
end
if isempty(F0), logTxt = logTxt + "  no alert" + newline; end
R.log = logTxt + sprintf("\nGate + DSP cost: 0.5 %% of one YAMNet inference per window.");
end

function A = alertRule(tw, awake, catScore, vs, dspT, veto, thr, cats)
%ALERTRULE The alert rule: score >= 0.30 in N classified windows in a row (fire, siren: 2; door: 1),
% a run survives skipped windows up to 2 s apart, 20 s cooldown per category, most urgent first;
% DSP alarm detections join under the same cooldown.
need = [2 2 1]; runs = [0 0 0]; last = -inf(1, 3); lastCls = -inf;
col = {[.85 .1 .1], [.1 .3 .85], [.95 .6 0]};
A = cell(0, 3);
for k = 1:numel(tw)
    t = tw(k);
    for td = dspT(dspT <= t & dspT > t - 0.5)            % DSP detections in the last hop
        if td - last(1) >= 20
            last(1) = td; A(end + 1, :) = {td, "FIRE ALARM (DSP)", col{1}}; %#ok<AGROW>
        end
    end
    if ~awake(k), continue; end
    if t - lastCls > 2.0 + 1e-6, runs(:) = 0; end
    lastCls = t;
    ok = catScore(k, :) >= thr;
    ok(1) = ok(1) && vs(k) < catScore(k, 1);
    if veto, ok(1) = ok(1) && catScore(k, 1) > catScore(k, 2); end
    runs = (runs + 1) .* ok;
    j = find(runs >= need & t - last >= 20, 1);
    if ~isempty(j)
        last(j) = t; runs(j) = 0;
        A(end + 1, :) = {t, upper(string(cats{j})) + " (YAMNet)", col{j}}; %#ok<AGROW>
    end
end
end

function E = goertzelBank(frames, freqs, fs)
%GOERTZELBANK Power of DFT bins via s[n] = x[n] + 2cos(w) s[n-1] - s[n-2] (2nd-order IIR resonator).
c = 2 * cos(2 * pi * freqs / fs);                       % 1 x probes
s1 = zeros(size(frames, 1), numel(freqs)); s2 = s1;
for k = 1:size(frames, 2)
    s0 = frames(:, k) + c .* s1 - s2;
    s2 = s1; s1 = s0;
end
E = s1 .^ 2 + s2 .^ 2 - c .* s1 .* s2;
end

function ok = envOk(e)
%ENVOK "On" only within 8 dB of the running 1.5 s peak (reverberant tails decay below it).
db = 10 * log10(e + 1e-12);
ok = db > movmax(db, [74 0]) - 8;
end

function hits = t3Times(on, dt, strict)
%T3TIMES Times at which an ISO 8201 T-3 pattern completes: 3 pulses of 0.3-0.8 s, gaps 0.3-0.8 s.
on = on(:).';
d = diff([0 on 0]); st = find(d == 1); en = find(d == -1);
for k = 1:numel(st) - 1                                  % bridge dropouts up to 100 ms
    if st(k + 1) - en(k) <= 5, on(en(k):st(k + 1) - 1) = true; end
end
on = medfilt1(double(on), 3) > 0.5;                      % drop blips
d = diff([0 on 0]); st = find(d == 1); en = find(d == -1);
p = [st(:) en(:)] * dt; hits = [];
for i = 1:size(p, 1) - 2
    q = p(i:i + 2, :); len = q(:, 2) - q(:, 1); gap = q(2:3, 1) - q(1:2, 2);
    if all(len >= 0.3 & len <= 0.8) && all(gap >= 0.3 & gap <= 0.8)
        if ~strict
            hits(end + 1) = q(3, 2); %#ok<AGROW>
        else
            nxt = inf; if i + 3 <= size(p, 1), nxt = p(i + 3, 1); end
            prv = inf; if i > 1, prv = q(1, 1) - p(i - 1, 2); end
            if nxt - q(3, 2) >= 1.0 && prv >= 1.0 && numel(on) * dt >= q(3, 2) + 1.0
                hits(end + 1) = q(3, 2) + 1.0; %#ok<AGROW>
            end
        end
    end
end
end

function checkAll(P, fs, extra)
[net, classes] = audioPretrainedNetwork("yamnet");
items = [demoSounds(), extra(:).'];
for i = 1:numel(items)
    for snr = [20 10 0]
        if startsWith(items{i}, 'http')
            [~, b, e] = fileparts(items{i}); f = fullfile(tempdir, ['wristecho_' b e]);
            if ~isfile(f), websave(f, items{i}); end
            [y, f0] = audioread(f); s = resample(mean(y, 2), fs, f0);
            [x, ~] = makeScene(s, snr, fs);
        else
            [x, ~] = makeScene(items{i}, snr, fs);
        end
        R = frontEnd(x, fs, P, net, classes);
        [~, nm] = fileparts(items{i});
        fprintf('%-28s %3d dB | saved %3.0f%% | system: %-32s | always-on: %s\n', nm(1:min(end, 28)), snr, ...
            100 * (1 - mean(R.awake)), joinAlerts(R.alerts), joinAlerts(R.f0));
    end
end
end

function t = joinAlerts(A)
if isempty(A), t = '-'; else, t = char(strjoin(reshape(string(A(:, 2)), 1, []), ', ')); end
end

function [x, onset] = makeScene(name, snr, fs)
%MAKESCENE 4 s of room noise, the sound (at most 12 s) at the chosen SNR, then 4 s of room noise.
if ischar(name) || isstring(name), s = demoSounds(name, fs); else, s = name; name = ''; end
s = s(1:min(end, 12 * fs));
fr = reshape(s(1:floor(numel(s) / 320) * 320), 320, []);
r = sqrt(mean(fr .^ 2)); act = r > max(r) * 10 ^ (-35 / 20);
rmsAct = sqrt(mean(r(act) .^ 2));
onset = 4;
n = round((onset + numel(s) / fs + 4) * fs);
rng(1);
bg = pinknoise(n, 1);
bg = bg / rms(bg) * 10 ^ (-45 / 20);                    % room at -45 dBFS
x = bg;
i = round(onset * fs);
x(i + (1:numel(s))) = x(i + (1:numel(s))) + s / rmsAct * 10 ^ (-45 / 20) * 10 ^ (snr / 20);
if contains(name, 'Quiet'), x = bg; end
x = max(min(x, 1), -1);
end
