# Woof TTS for Android

An offline Android system Text-To-Speech (TTS) engine powered by **Underdog Woof** running on **ONNX Runtime Mobile**.

Woof is a high-accuracy, lightweight English TTS model with two expressive voices: **Juno** and **Mira**. This app packages Woof as a first-class Android `TextToSpeechService`, allowing any Android app (e.g. e-book readers, navigation, accessibility screen readers, assistant apps) to use Woof as their text-to-speech voice.

---

## Features

- **System-Wide TTS Service**: Registers with Android as an official TTS engine (`android.intent.action.TTS_SERVICE`).
- **Two Built-in Voices**:
  - `juno` (Default male voice)
  - `mira` (Female voice)
- **100% Offline & Private**: All ONNX models and text normalization tables are bundled directly inside the app. Zero network permissions (`android.permission.INTERNET` is not even requested).
- **High-Fidelity 24 kHz Audio**: Synthesizes studio-grade 24 kHz mono 16-bit PCM audio via an autoregressive speech transformer + neural vocoder.
- **Advanced English Normalization**: Port of Underdog Woof's Python frontend for numbers, currency, dates, abbreviations, tech jargon, and respelling dictionaries.
- **Settings & Test Screen**: Preview voices, test arbitrary text, and launch Android's system Text-To-Speech settings with one tap.

---

## Architecture

```
                      [ User Text / Client App Request ]
                                      │
                                      ▼
                           WoofTtsService (Android)
                                      │
                         ┌────────────┴────────────┐
                         │ Text Normalizer & Split │
                         └────────────┬────────────┘
                                      │ Tokens (IntArray)
                                      ▼
                                  WoofEngine
                         ┌─────────────────────────┐
                         │   woof_encoder.onnx     │  (Prefix tokens -> conditioning & KV cache)
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │     woof_step.onnx      │  (Autoregressive frame generation loop)
                         └────────────┬────────────┘
                                      │ Latents (T, 32)
                         ┌────────────▼────────────┐
                         │    woof_decoder.onnx    │  (Neural vocoder -> 24 kHz waveform)
                         └────────────┬────────────┘
                                      │ 16-bit PCM
                                      ▼
                          [ SynthesisCallback.audioAvailable ]
```

---

## Building the App

### GitHub Actions & GitHub CLI (`gh`)

The project includes automated CI (`.github/workflows/build.yml`) that runs the unit test suite and compiles both debug and release APKs.

1. **Push commits**:
   ```bash
   git push -u origin main
   ```

2. **Trigger workflow manually (optional)**:
   ```bash
   gh workflow run "Build Android TTS Engine"
   ```

3. **Check build progress**:
   ```bash
   gh run list --workflow="Build Android TTS Engine"
   gh run watch
   ```

4. **Download the compiled APK**:
   ```bash
   gh run download -n woof-tts-debug
   ```

### Local Build (Requires JDK 17+ and Android SDK 34)

```bash
./gradlew assembleDebug
```

The APK will be generated at `app/build/outputs/apk/debug/app-debug.apk`.

---

## Installation & Setup on Android

1. Install the APK on your Android device:
   ```bash
   adb install app-debug.apk
   ```
2. Open the **Woof TTS** app from your launcher to test synthesis and select your default voice (`Juno` or `Mira`).
3. Tap **"Open system TTS settings"** (or navigate to `Settings` → `System` → `Languages & input` → `Text-to-speech output`).
4. Set **Preferred engine** to **Woof TTS**.

---

## License & Credits

- Underdog Woof model and weights © 2026 Conway Research. Licensed under the [Underdog Non-Commercial License 1.0](LICENSE).
- Voice by [Underdog Woof](https://underdog.ai/woof).
