Hand Gesture Classification from sEMG Using Multi-Domain Features
This repository contains code and resources for hand gesture classification using surface electromyography (sEMG) signals. The project leverages multi-domain feature extraction and ensemble machine learning models to improve recognition accuracy for both intact and amputee participants.

Overview
Accurate hand gesture recognition from sEMG is critical for prosthetic control and human–machine interfaces.
We propose a framework that extracts 456 features per segment across:

- Time-domain features (e.g., RMS, Zero Crossings, Waveform Length)

- Frequency-domain features (e.g., Median Frequency, Band Power)

- Autoregressive (AR) coefficients

- Hjorth parameters (Mobility, Complexity)

These features are evaluated using XGBoost, Random Forest, and LightGBM under intra- and inter-participant settings.

Dataset
- Ninapro DB10 (MeganePro) dataset

- 30 intact participants + 15 amputees

- 12-channel Delsys Trigno Wireless system

- Sampling rate: 1,926 Hz

- 10 distinct hand gestures

Methods
- Preprocessing: Sliding window segmentation (500 ms, 1000 ms, 2000 ms) with 200 ms step size

- Feature extraction: 38 features per channel × 12 channels = 456 features per window

- Classification models: XGBoost, Random Forest, LightGBM

Results
- Intra-participant accuracy:

- Intact subjects: up to 83% (LightGBM, 2000 ms window)

- Amputee subjects: up to 71% (LightGBM, 2000 ms window)

- Inter-participant accuracy: low (25–29% intact, 13–17% amputees), highlighting subject-specific variability.

Key Insights
- Larger windows improve accuracy but increase latency.

- Multi-domain features provide robust representation of sEMG signals.

- Subject-specific calibration remains essential for amputee participants.

🔮 Future Work
- Adaptive algorithms (transfer learning, semi-supervised learning)

- Multimodal sensing (e.g., tactile + thermal fusion)

