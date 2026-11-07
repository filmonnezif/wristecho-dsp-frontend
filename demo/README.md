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
Checked with `WristEchoDemo('check')`, which prints what every sound does at 20 / 10 / 0 dB.

1. **Quiet room**: the gate stays asleep, 100 % of the classifier's compute is saved, no alert.
2. **Smoke alarm (real recording), 10 dB**: the gate wakes at the onset, YAMNet raises the fire
   alarm, 54 % of the compute is saved. Slide the noise to **0 dB**: always-on YAMNet now misses
   it, the Goertzel/T-3 detector still catches it.
3. **Fire-alarm panel beeps** and **Smoke alarm 520 Hz**: always-on YAMNet never recognises them;
   the DSP detector does (untick "DSP alarm detector" to show the miss).
4. **Siren (real recording), 10 dB**: the log shows always-on YAMNet raising FIRE ALARM + SIREN,
   this system only SIREN: the siren veto. Untick "Siren veto" to bring the false fire alarm back.
5. **Siren fading in**: the gate can sleep through a sound that rises slowly (the floor rises with
   it). Lower the gate threshold or add a hangover to catch it, and watch the saving drop.
6. **Continuous 1 Hz beeper**: both detectors call it a fire alarm; tick "Strict T-3" and the DSP
   detector stops (YAMNet still fires: a limitation discussed in the paper).
7. **Door knock**: neither system detects these real knocks, the classifier's weak class
   (door recall 16 % in the paper); the gate does wake for them.
8. **Microwave / truck beeper**: look-alikes; at high SNR YAMNet may call them fire alarms, the
   DSP detector stays quiet.

Sounds: Freesound 680073, 874698, 388044, 197166 (CC0); ESC-50 siren and knock (CC BY-NC,
K. J. Piczak); the 520 Hz alarm and the beeper are synthesised.
