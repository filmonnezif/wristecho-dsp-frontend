# Live demo (MATLAB)

Runs in MATLAB R2024a or later, desktop or MATLAB Online, with Audio, Signal Processing and Deep Learning
Toolboxes and the **Deep Learning Toolbox Converter for ONNX Model Format** support package.

```matlab
repo = gitclone("https://github.com/filmonnezif/wristecho-dsp-frontend");  % or: pull(gitrepo(repo))
cd(fullfile(repo, "demo"))
WristEchoDemo('selftest')  % the imported EfficientAT model vs. the Python export (error ~1e-4)
WristEchoDemo              % the live app
WristEchoDemo('check')     % every sound at 20 / 10 / 0 dB: this system vs. today's app (about 1 min)
designPlots                % HPF spec/response/pole-zero, Goertzel resonator, Goertzel vs FFT cost
```

Pick a sound, the room noise, the classifier and the false-alarm budget, press **Play**. The app runs the paper's
system and replays the scene with a moving cursor: spectrogram, gate statistic and threshold, when the
classifier is awake, Goertzel probe energies, the class scores against their thresholds, and the alert. The log
compares it with today's app (YAMNet always on, score >= 0.30).

* **Classifier "EfficientAT + head (paper)"**: the final system. The log-mel front end (pre-emphasis, 1024-point
  STFT, 128-band mel filterbank, log) is computed in MATLAB; the EfficientAT-mn10 CNN is imported from ONNX
  (`model/mn10.onnx`); the trained head, per-class smoothing and thresholds come from `model/final_system.mat`.
* **Classifier "YAMNet (today's app)"**: MATLAB's YAMNet with the app's rule, behind the same gate.
* **False-alarm budget**: the thresholds chosen on tuning data at 0.1 (the paper's choice), 0.3 or 1 false
  alarms per hour per class (`model/budgets.mat`).

## A three-minute script (checked with `WristEchoDemo('check')`)
1. **Quiet room**: the gate never wakes; 100 % of the classifier's compute is saved, no alert.
2. **Fire-alarm panel beeps** and **Smoke alarm 520 Hz**, 10 dB, then 0 dB: today's app never raises them; the
   Goertzel/T-3 detector does, even at 0 dB, where the gate keeps the classifier asleep.
3. **Door knock, 10 dB**: today's app misses it; the EfficientAT head raises DOOR.
4. **Car horn, 20 dB** and **Baby crying, 20 dB**: raised by the system, missed by today's app.
5. **Microwave beep** and **Truck reversing beeper**: look-alikes; nothing fires.
6. **Siren (real, fades in), 10 dB**: the honest limit. At the strict budget the EfficientAT head misses it
   (sirens are its weakest class, 21 % in the paper); switch the classifier to YAMNet and it is caught. Untick
   "Two noise floors" to see the gate sleep through the slow swell.
7. **Continuous 1 Hz beeper**: both systems call it an alarm; tick "Strict T-3" and the DSP detector stops.

Sounds: Freesound 680073, 874698, 388044, 197166 (CC0); ESC-50 sirens, knock, car horn and baby crying (CC BY-NC,
K. J. Piczak); the 520 Hz alarm and the beeper are synthesised.

## Model files
`model/` is produced in Colab by `demo/export_model.py` (EfficientAT-mn10 with its front end as ONNX, the head and
settings) and converted for MATLAB by `demo/onnx_for_matlab.py` (squeeze-excitation blocks as native ops, front end
split off into `logmel.mat`; outputs checked with onnxruntime).
