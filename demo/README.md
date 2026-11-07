# Live demo (MATLAB)

Runs in MATLAB R2024a or later, desktop or MATLAB Online, with Audio, Signal Processing and Deep
Learning Toolboxes.

```matlab
repo = gitclone("https://github.com/filmonnezif/wristecho-dsp-frontend");  % or download the repo
cd(fullfile(repo, "demo"))
WristEchoDemo      % the live app
designPlots        % HPF spec/response/pole-zero, Goertzel resonator, Goertzel vs FFT cost
```

Pick a sound, set the room noise and the front end, press **Play**. The app runs the paper's
front end (gate at the 5 % budget: Δ = 18 dB, H = 0 s; alarm detector v4) and YAMNet, then
replays the scene with a moving cursor.

## A three-minute script for the demo
1. **Quiet room**: the gate stays asleep; "Classifier compute saved" is close to 100 %.
2. **Smoke alarm (real recording)**: the gate wakes at the onset, the Goertzel bars light up at
   about 3 kHz, the alert fires. Move the noise slider to 0 dB and play again.
3. **Smoke alarm 520 Hz (ISO 8201)**: YAMNet does not recognise it; the DSP detector does.
   Untick "DSP alarm detector" to show the miss.
4. **Siren**: untick "Siren veto": YAMNet raises a FIRE ALARM on the siren; tick it again.
5. **Microwave / truck beeper**: YAMNet says fire, the DSP detector stays quiet.
6. **Continuous 1 Hz beeper**: the DSP detector fires; tick "Strict T-3" and it stops (the
   trade-off discussed in the paper).
7. Change Δ and H live: a lower Δ wakes the classifier more often (less saving), a higher Δ can
   miss quiet sounds.

Sounds: Freesound 680073, 874698, 388044, 197166 (CC0); ESC-50 siren and knock (CC BY-NC,
K. J. Piczak); the 520 Hz alarm and the beeper are synthesised.
