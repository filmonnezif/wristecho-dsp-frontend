function out = demoSounds(name, fs)
%DEMOSOUNDS The demo's sound bank. demoSounds() lists the names; demoSounds(name, fs) returns
%   the sound as a mono column at fs. Real recordings are downloaded once into tempdir; the
%   alarm and beeper are synthesised. Sources and licenses:
%     Freesound 680073 "Smoke alarm.wav" (CC0), 874698 "Fire Alarm Panel Repeated Beeps" (CC0),
%     388044 "Microwave beep.wav" (CC0), 197166 "Truck Reverse Beeper.wav" (CC0);
%     ESC-50 5-150409-A-42 and 5-117118-A-42 (sirens), 5-235644-A-30 (door knock), 5-179860-A-43 (car horn),
%     5-151085-A-20 (baby crying) (CC BY-NC, K. J. Piczak).
bank = {
    'Quiet room (nothing happens)',              'none', ''
    'Smoke alarm (real recording)',              'url',  'https://cdn.freesound.org/previews/680/680073_4627588-hq.mp3'
    'Fire-alarm panel beeps (real, YAMNet misses)', 'url', 'https://cdn.freesound.org/previews/874/874698_16682330-hq.mp3'
    'Smoke alarm 520 Hz (ISO 8201, synthetic)',  'synth', '520'
    'Siren (real, fades in)',                    'url',  'https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/5-117118-A-42.wav'
    'Siren 2 (real; at 10 dB the gate can miss it)', 'url', 'https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/5-150409-A-42.wav'
    'Door knock (real recording)',               'url',  'https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/5-235644-A-30.wav'
    'Car horn (real recording)',                 'url',  'https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/5-179860-A-43.wav'
    'Baby crying (real recording)',              'url',  'https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/5-151085-A-20.wav'
    'Microwave beep (look-alike)',               'url',  'https://cdn.freesound.org/previews/388/388044_7282645-hq.mp3'
    'Truck reversing beeper (look-alike)',       'url',  'https://cdn.freesound.org/previews/197/197166_389864-hq.mp3'
    'Continuous 1 Hz beeper 3.1 kHz (look-alike)', 'synth', 'beeper'
    };
if nargin == 0
    out = bank(:, 1).';
    return
end
k = find(strcmp(bank(:, 1), name), 1);
switch bank{k, 2}
    case 'none'
        out = zeros(round(4 * fs), 1);
    case 'synth'
        t = (0:round(10 * fs) - 1).' / fs;
        if strcmp(bank{k, 3}, '520')                    % T-3: 3 x (0.5 s on, 0.5 s off), 1.5 s pause
            ph = 2 * pi * 520 * t; tone = zeros(size(t));
            for h = 1:2:15, tone = tone + sin(h * ph) / h; end
            cyc = mod(t, 4); env = (cyc < 3) & (mod(cyc, 1) < 0.5);
        else                                             % continuous beeper: 0.5 s on, 0.5 s off
            tone = sin(2 * pi * 3100 * t); env = mod(t, 1) < 0.5;
        end
        out = 0.3 * tone .* conv(double(env), ones(80, 1) / 80, 'same');
    case 'url'
        [~, base, ext] = fileparts(bank{k, 3});
        file = fullfile(tempdir, ['wristecho_' base ext]);
        if ~isfile(file), websave(file, bank{k, 3}); end
        [y, f0] = audioread(file);
        out = resample(mean(y, 2), fs, f0);
end
end
