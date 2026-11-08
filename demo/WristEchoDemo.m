function WristEchoDemo(mode, extra)
%WRISTECHODEMO Live demo of the paper's system ("Listen Before You Classify").
%   WristEchoDemo            the app: pick a sound, set the room noise and the front end, press Play.
%   WristEchoDemo('check')   runs every sound without the UI and prints what the system and today's app
%                            (always-on YAMNet) report (WristEchoDemo('check', urls) adds recordings by URL).
%   WristEchoDemo('selftest') checks the imported EfficientAT model against the Python export.
%   The app mixes the sound into room noise, runs the DSP front end and the classifier exactly as in the
%   paper, and replays the result in real time: spectrogram, gate statistic vs. threshold, when the
%   classifier is awake, Goertzel probe energies, the class scores and the alert.
%   Classifier: "EfficientAT + head" is the paper's final system (log-mel computed here, EfficientAT-mn10 CNN imported from ONNX,
%   the trained head, per-class smoothing and thresholds at 0.1 false alarms/h per class, from
%   model/final_system.mat); "YAMNet" is today's app rule (score >= 0.30).
%
%   Requires Audio Toolbox, Signal Processing Toolbox, Deep Learning Toolbox (YAMNet via
%   audioPretrainedNetwork) and the Deep Learning Toolbox Converter for ONNX Model Format. Sound files are
%   downloaded once into tempdir (see demoSounds).

fs = 16000;
P = defaults();
if nargin && strcmp(mode, 'selftest')
    selfTest();
    return
end
if nargin && strcmp(mode, 'check')
    if nargin < 2, extra = {}; end
    checkAll(P, fs, extra);
    return
end
S = struct('scene', [], 'res', [], 'stop', false);

fig = uifigure('Name', 'Listen Before You Classify - live demo', 'Position', [60 60 1280 760], 'Color', [1 1 1]);
g = uigridlayout(fig, [1 2], 'ColumnWidth', {300, '1x'}, 'Padding', [10 10 10 10]);

% --- Controls -------------------------------------------------------------------------------
c = uigridlayout(g, [24 1], 'RowHeight', repmat({'fit'}, 1, 24), 'RowSpacing', 5);
uilabel(c, 'Text', 'Sound', 'FontWeight', 'bold');
names = demoSounds();
ddSound = uidropdown(c, 'Items', names, 'Value', names{2});
uilabel(c, 'Text', 'Room noise (SNR of the sound, dB)');
slSnr = uislider(c, 'Limits', [-5 40], 'Value', 10, 'MajorTicks', -5:5:40);
uilabel(c, 'Text', 'Classifier', 'FontWeight', 'bold');
ddClf = uidropdown(c, 'Items', {'EfficientAT + head (paper)', 'YAMNet (today''s app)'}, 'Value', 'EfficientAT + head (paper)');
uilabel(c, 'Text', 'False-alarm budget (EfficientAT; per hour per class)');
ddBudget = uidropdown(c, 'Items', {'0.1: strict (paper''s choice)', '0.3', '1: relaxed (more recall, more false alerts)'}, ...
    'ItemsData', 1:3, 'Value', 1);
cbGate = uicheckbox(c, 'Text', 'Wake-up gate on (off = classifier on every window)', 'Value', true);
cbTwo = uicheckbox(c, 'Text', 'Two noise floors (catches sounds that swell slowly)', 'Value', true);
uilabel(c, 'Text', 'Gate threshold \Delta (dB above the noise floor)', 'Interpreter', 'tex');
slDelta = uislider(c, 'Limits', [5 30], 'Value', P.delta, 'MajorTicks', 5:5:30);
cbDsp = uicheckbox(c, 'Text', 'DSP alarm detector (Goertzel + T-3)', 'Value', true);
cbStrict = uicheckbox(c, 'Text', 'Strict T-3 (needs the 1.5 s pause)', 'Value', false);
cbVeto = uicheckbox(c, 'Text', 'Siren veto (alarm needs alarm score > siren score)', 'Value', true);
bPlay = uibutton(c, 'Text', 'Play', 'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', [0 .62 .45], 'FontColor', 'w');
bStop = uibutton(c, 'Text', 'Stop');
lamp = uilamp(c, 'Color', [.6 .6 .6]);
lblAwake = uilabel(c, 'Text', 'Classifier: asleep', 'FontSize', 14);
lblSaved = uilabel(c, 'Text', 'Classifier compute saved: -', 'FontSize', 14, 'FontWeight', 'bold', 'WordWrap', 'on');
alert = uilabel(c, 'Text', 'no alert', 'FontSize', 26, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'BackgroundColor', [.93 .93 .93], 'FontColor', [.4 .4 .4]);
lblLog = uilabel(c, 'Text', '', 'WordWrap', 'on', 'FontSize', 11);

% --- Plots ----------------------------------------------------------------------------------
p = uigridlayout(g, [3 2], 'RowHeight', {'1.3x', '1x', '1x'}, 'ColumnWidth', {'2x', '1x'});
axSpec = uiaxes(p); axSpec.Layout.Column = [1 2]; title(axSpec, 'Spectrogram');
axGate = uiaxes(p); axGate.Layout.Column = [1 2]; title(axGate, 'Gate statistic (max band rise above its floor)');
axProbe = uiaxes(p); title(axProbe, 'Goertzel probes (share of frame energy)');
axYam = uiaxes(p); title(axYam, 'Class scores (dots: thresholds)');

bPlay.ButtonPushedFcn = @(~, ~) onPlay();
bStop.ButtonPushedFcn = @(~, ~) setStop();
ddClf.ValueChangedFcn = @(~, ~) onClf();
ddBudget.ValueChangedFcn = @(~, ~) onBudget();

    function setStop()
        S.stop = true;
    end

    function onClf()  % each classifier (and budget) has its own calibrated gate threshold
        if startsWith(ddClf.Value, 'EfficientAT'), slDelta.Value = P.final.gate_delta; else, slDelta.Value = P.deltaYamnet; end
    end

    function onBudget()  % thresholds and gate settings chosen on tuning data at that budget (paper, Table I)
        i = ddBudget.Value;
        P.final.thresholds = P.budgets.thresholds(i, :);
        P.final.gate_delta = P.budgets.gate_delta(i); P.final.gate_hang = P.budgets.gate_hang(i);
        onClf();
    end

    function onPlay()
        S.stop = false;
        bPlay.Enable = 'off';
        lblLog.Text = 'Preparing (the first run loads the models)...';
        drawnow;
        P.delta = slDelta.Value; P.gate = cbGate.Value; P.dsp = cbDsp.Value; P.strict = cbStrict.Value;
        P.veto = cbVeto.Value; P.twoFloor = cbTwo.Value; P.clf = ddClf.Value;
        [x, onset] = makeScene(ddSound.Value, slSnr.Value, fs);
        R = frontEnd(x, fs, P);
        animate(x, onset, R);
        bPlay.Enable = 'on';
    end

    function animate(x, onset, R)
        tf = (1:numel(R.nov)) * P.frame / fs;
        tw = R.tw;
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
                    lamp.Color = [0 .8 .3]; lblAwake.Text = 'Classifier: AWAKE';
                else
                    lamp.Color = [.6 .6 .6]; lblAwake.Text = 'Classifier: asleep';
                end
                cats = categorical(R.cats, R.cats);
                bar(axYam, cats, R.catScore(k, :) .* R.awake(k), 'FaceColor', [.2 .4 .8]);
                hold(axYam, 'on'); plot(axYam, cats, R.thr, 'r.', 'MarkerSize', 14); hold(axYam, 'off');
                ylim(axYam, [0 1]);
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
% The paper's final system: two-floor gate, alarm detector, siren veto, EfficientAT-mn10 + head with the
% settings exported from the paper's run (model/final_system.mat). YAMNet mode: today's app rule, with the
% two-floor gate calibrated for YAMNet at the 5 % budget (Delta = 19 dB).
P = struct('frame', 320, 'nfft', 512, 'hop', 8000, 'win', 15600, 'nbands', 16, 'bandLo', 100, 'bandHi', 7600, ...
    'smooth', 0.06, 'rise', 3, 'slowRise', 0.3, 'slowOffset', 6, 'twoFloor', true, 'delta', 19, 'hang', 0, ...
    'gate', true, 'dsp', true, 'strict', false, 'veto', true, 'probeLo', 2800, 'probeHi', 3600, 'ratio', 0.15, ...
    'square', [520 1560 2600], 'clf', 'EfficientAT + head (paper)', 'deltaYamnet', 19);
P.cats = {'Alarm', 'Siren', 'Car horn', 'Baby crying', 'Door'};
P.need = [2 2 1 2 1];
P.final = load(fullfile(fileparts(mfilename('fullpath')), 'model', 'final_system.mat'));
P.budgets = load(fullfile(fileparts(mfilename('fullpath')), 'model', 'budgets.mat'));
P.delta = P.final.gate_delta;
end

function R = frontEnd(x, fs, P)
%FRONTEND The DSP front end and the classifier on audio x (column), as in the paper.
% Gate path: HPF -> 20 ms frames -> |FFT|^2 in 16 log bands -> one-pole smoothing -> two floors -> novelty
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
F = zeros(size(E)); Sl = F; fl = min(E(1:min(50, n), :), [], 1); sl = fl;
rise = P.rise * P.frame / fs; slow = P.slowRise * P.frame / fs;
for t = 1:n
    fl = min(E(t, :), fl + rise);                       % fast floor: follows down at once, rises <= 3 dB/s
    sl = min(E(t, :), sl + slow);                       % slow floor: rises <= 0.3 dB/s (remembers the quiet room)
    F(t, :) = fl; Sl(t, :) = sl;
end
R.nov = max(E - F, [], 2);
if P.twoFloor, R.nov = max(R.nov, max(E - Sl, [], 2) - P.slowOffset); end
nw = floor((numel(x) - P.win) / P.hop) + 1;
R.tw = (P.win + (0:nw - 1) * P.hop) / fs;              % time at which each window ends
R.novw = zeros(1, nw);
for k = 1:nw
    e = floor((P.win + (k - 1) * P.hop) / P.frame);
    R.novw(k) = max(R.nov(max(1, e - P.hop / P.frame + 1):e));
end
final = startsWith(P.clf, 'EfficientAT');
hang = 0; if final, hang = P.final.gate_hang; end
R.awake = conv(double(R.novw > P.delta), ones(1, round(hang / (P.hop / fs)) + 1));
R.awake = R.awake(1:nw) > 0;
if ~P.gate, R.awake = true(1, nw); end

% Classifiers on every window (only the awake ones are used; computing all lets the app compare)
R.cats = P.cats;
[yamScore, vs] = yamnetScores(x, fs, P, nw);
if final
    R.catScore = finalScores(x, fs, P, nw);
    R.thr = P.final.thresholds(:).';
else
    R.catScore = yamScore;
    R.thr = 0.30 * ones(1, 5);
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

vsUse = vs; if final, vsUse = zeros(nw, 1); end         % the ringtone veto belongs to YAMNet's labels
R.alerts = alertRule(R.tw, R.awake, R.catScore, vsUse, dspT, P.veto, R.thr, R.cats, P.need);
R.f0 = alertRule(R.tw, true(1, nw), yamScore, vs, [], true, 0.30 * ones(1, 5), R.cats, P.need);
logTxt = "This system:" + newline;
for k = 1:size(R.alerts, 1)
    logTxt = logTxt + sprintf("  %.1f s  %s\n", R.alerts{k, 1}, R.alerts{k, 2});
end
if isempty(R.alerts), logTxt = logTxt + "  no alert" + newline; end
logTxt = logTxt + "Today's app (YAMNet always on, 0.30):" + newline;
for k = 1:size(R.f0, 1)
    logTxt = logTxt + sprintf("  %.1f s  %s\n", R.f0{k, 1}, R.f0{k, 2});
end
if isempty(R.f0), logTxt = logTxt + "  no alert" + newline; end
R.log = logTxt + sprintf("\nGate + DSP: 0.5 %% of one YAMNet inference per window.");
end

function [catScore, vs] = yamnetScores(x, fs, P, nw)
%YAMNETSCORES Stock YAMNet: per class the best label score; vs = best ringtone/telephone score (veto).
persistent net classes
if isempty(net), [net, classes] = audioPretrainedNetwork("yamnet"); end
lab = {{"Smoke detector, smoke alarm", "Fire alarm", "Alarm"}, ...
       {"Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Emergency vehicle"}, ...
       {"Vehicle horn, car horn, honking", "Air horn, truck horn"}, {"Baby cry, infant cry"}, ...
       {"Knock", "Doorbell", "Ding-dong"}};
veto = ismember(string(classes), ["Ringtone", "Telephone", "Telephone bell ringing"]);
catScore = zeros(nw, 5); vs = zeros(nw, 1);
for k = 1:nw
    s = mean(predict(net, yamnetPreprocess(x((k - 1) * P.hop + (1:P.win)), fs)), 1);
    for j = 1:5
        catScore(k, j) = max(s(ismember(string(classes), string(lab{j}))));
    end
    vs(k) = max(s(veto));
end
end

function catScore = finalScores(x, fs, P, nw)
%FINALSCORES The paper's final classifier: EfficientAT-mn10 (ONNX) -> embedding + its own class logits ->
% trained head (softmax) -> per class: head probability, own score or their mean -> causal mean over k windows.
M = P.final;
x32 = resample(x, 2, 1);
head = zeros(nw, 5); own = zeros(nw, 5);
for k = 1:nw
    [logits, emb] = mn10(x32(2 * (k - 1) * P.hop + (1:2 * P.win)));
    sc = 1 ./ (1 + exp(-logits));
    for j = 1:5
        if iscell(M.cat_label_idx), idx = M.cat_label_idx{j}; else, idx = M.cat_label_idx(j, :); end
        own(k, j) = max(sc(idx));
    end
    feat = emb;
    if M.stack
        q = min(max(own(k, :), 1e-4), 1 - 1e-4);
        feat = [feat, log(q ./ (1 - q))];
    end
    a = (feat - M.mu) ./ M.sd;
    for l = 0:M.n_layers - 1
        a = a * double(M.(sprintf('W%d', l))) + double(M.(sprintf('b%d', l)));
        if l < M.n_layers - 1, a = max(a, 0); end
    end
    e = exp(a - max(a));
    pr = e / sum(e);
    head(k, :) = pr(1:5);
end
catScore = zeros(nw, 5);
for j = 1:5
    if iscell(M.source), src = M.source{j}; else, src = strtrim(M.source(j, :)); end
    switch strtrim(src)
        case 'head', s = head(:, j);
        case 'stock', s = own(:, j);
        otherwise, s = (head(:, j) + own(:, j)) / 2;
    end
    k = M.smooth(j);
    catScore(:, j) = filter(ones(k, 1), 1, s) ./ min((1:nw).', k);   % causal mean over the last k windows
end
end

function [logits, emb] = mn10(w)
%MN10 EfficientAT-mn10 on one 0.975 s window at 32 kHz. Its log-mel front end is computed here (pre-emphasis,
% reflect padding, 1024-point STFT with its 800-point Hann window, hop 320, 128-band mel filterbank, log); the
% CNN is imported from ONNX (model/mn10.onnx, input 128 mel bands x 98 frames).
persistent net L
if isempty(net)
    d = fullfile(fileparts(mfilename('fullpath')), 'model');
    L = load(fullfile(d, 'logmel.mat'));
    old = cd(tempdir);                                   % the importer writes helper packages to the current folder
    g = dir('+mn10*'); for i = 1:numel(g), rmdir(g(i).name, 's'); end
    net = importNetworkFromONNX(fullfile(d, 'mn10.onnx'), 'Namespace', 'mn10demo');
    cd(old); addpath(tempdir);
end
y = w(2:end) - 0.97 * w(1:end - 1);                      % pre-emphasis
y = [y(513:-1:2); y; y(end - 1:-1:end - 512)];           % reflect padding (n_fft / 2), as torch.stft(center=True)
T = floor((numel(y) - 1024) / 320) + 1;
F = y((0:T - 1) * 320 + (1:1024).') .* L.window(:);      % 1024 x T frames
X = fft(F, 1024);
mel = (log(L.banks * abs(X(1:513, :)) .^ 2 + 1e-5) + 4.5) / 5;   % 128 x T log-mel
[o1, o2] = predict(net, dlarray(single(mel), 'SSCB'));
o1 = double(extractdata(o1)); o2 = double(extractdata(o2));
if numel(o1) == 527, logits = o1(:).'; emb = o2(:).'; else, logits = o2(:).'; emb = o1(:).'; end
end

function selfTest()
%SELFTEST The imported model against the Python export on the stored test window.
M = load(fullfile(fileparts(mfilename('fullpath')), 'model', 'final_system.mat'));
[logits, emb] = mn10(M.test_x32(:));
fprintf('mn10 import: max |logit| error %.2e, max |embedding| error %.2e (should be < 1e-3)\n', ...
    max(abs(logits - M.test_logits(:).')), max(abs(emb - M.test_embedding(:).')));
end

function A = alertRule(tw, awake, catScore, vs, dspT, veto, thr, cats, need)
%ALERTRULE The alert rule: score >= threshold in N classified windows in a row (alarm, siren, baby: 2;
% horn, door: 1), a run survives skipped windows up to 2 s apart, 20 s cooldown per class, most urgent
% first; DSP alarm detections join under the same cooldown.
nc = numel(cats); runs = zeros(1, nc); last = -inf(1, nc); lastCls = -inf;
col = {[.85 .1 .1], [.1 .3 .85], [.55 .35 .1], [.6 .2 .7], [.95 .6 0]};
A = cell(0, 3);
for k = 1:numel(tw)
    t = tw(k);
    for td = dspT(dspT <= t & dspT > t - 0.5)            % DSP detections in the last hop
        if td - last(1) >= 20
            last(1) = td; A(end + 1, :) = {td, "ALARM (DSP)", col{1}}; %#ok<AGROW>
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
        A(end + 1, :) = {t, upper(string(cats{j})), col{j}}; %#ok<AGROW>
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
        R = frontEnd(x, fs, P);
        [~, nm] = fileparts(items{i});
        fprintf('%-28s %3d dB | saved %3.0f%% | system: %-30s | today''s app: %s\n', nm(1:min(end, 28)), snr, ...
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
