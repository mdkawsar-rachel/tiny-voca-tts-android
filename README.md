# tiny-voca TTS for Android

An offline Android system Text-To-Speech engine powered by **[tiny-voca](https://huggingface.co/n4ze3m/tiny-voca)** (4.5 M parameters, 24 kHz, two American-English voices) running on the **native ggml runtime in llama.cpp** — no ONNX Runtime, no Python, no network.

This is a fork/rebuild of `woof-tts-android` with the Woof model replaced by tiny-voca and the ONNX Runtime stack replaced by a native JNI bridge to llama.cpp.

---

## Why this design

| | Old app (Woof + ONNX Runtime) | This app (tiny-voca + ggml) |
|---|---|---|
| Parameters | 38.24 M | **4.54 M** |
| Model assets | ~40 MB ONNX + 5 MB text tables | **25.8 MB** (two GGUFs, memory-mapped) |
| Generation | autoregressive: encoder + hundreds of `step.run()` calls + 15 M-param neural vocoder | **non-autoregressive**: one graph pass per sentence + 26 k-param source-filter synth |
| Peak memory | ~330 MB of ORT sessions | tens of MB |
| Front end | Kotlin port of Underdog normalizer | dizione port embedded in the GGUF `tts` chat template |
| Licence | Underdog Non-Commercial 1.0 | **Apache-2.0** |

The model is tiny and its voice is not the most natural, so **quality preservation was the hard constraint**. Consequently:

- The **F32 GGUF pair published by ggml-org** is used unchanged. No weight quantization is applied: for 4.5 M parameters the quality cost is not worth the negligible speed gain. (You may quantize the main GGUF to Q8_0 with `llama-quantize` if you need a smaller APK, but keep the F32 mmproj.)
- The **native `tts` chat template is the reference dizione port**. It handles numbers, dates, money, homographs, stress and sentence splitting exactly like the upstream Python pipeline; re-implementing it in Kotlin would risk pronunciation regressions.
- The only post-processing is the model card's **0.99 peak limiter**, which merely prevents 16-bit clipping.
- The waveform comes straight from llama.cpp's `tiny-voca` pipeline as F32 PCM; there is no WAV encode/decode round trip and no resampling.

Performance comes from the architecture and the runtime, not from audio shortcuts: ggml KleidiAI NEON kernels on `arm64-v8a`, mmap'd weights, one shared process-wide engine, and 2–4 CPU threads.

---

## Architecture

```
  Android app / accessibility / e-book reader
                  │  Intent.ACTION_TTS_SERVICE
                  ▼
  TinyVocaTtsService (system TTS engine, 24 kHz mono PCM16)
                  │
                  ▼
  TinyVocaEngine (Kotlin)  ── assets copied once ──► tiny-voca-F32.gguf
                  │                                  mmproj-tiny-voca-F32.gguf
                  │ JNI
                  ▼
  tinyvoca_jni.cpp
                  │  common_chat_get_tts_prompt()   → dizione text → phonemes + stress
                  │  mtmd_helper_gen_audio_*()      → tiny-voca encoder → decoder → synth
                  ▼
  FloatArray PCM @ 24 kHz  →  0.99 peak limit  →  chunk queue  →  SynthesisCallback
```

The native engine is created once per process and reused by both the service and the test Activity. `onStop()` cancels playback; synthesis itself is short (non-autoregressive) and is not interrupted mid-graph.

---

## Building

The build requires Android SDK 34, NDK and CMake 3.22.1. CI does all of this automatically on `ubuntu-latest`.

```bash
git clone --recurse-submodules <this-repo>
cd tiny-voca-tts-android
python tools/download_models.py     # verified download of both GGUFs into app assets
./gradlew assembleDebug             # app/build/outputs/apk/debug/app-debug.apk
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

GitHub Actions (`.github/workflows/android.yml`) checks out the pinned llama.cpp submodule, downloads and checksum-verifies the GGUFs, builds both ABIs (`arm64-v8a`, `x86_64`) and uploads debug/release APKs as artifacts.

---

## Install & use

```bash
adb install app-debug.apk
```

1. Open **tiny-voca TTS** to pick **F2** (female) or **M2** (male) and preview a sentence with live real-time-factor stats.
2. Open system **Settings → Languages & input → Text-to-speech output** and choose **tiny-voca TTS** as the preferred engine.
3. Any app that uses Android's `TextToSpeech` API can now speak through it. No INTERNET permission is requested.

---

## Notes and limitations

- English only, two fixed voices, no voice cloning.
- tiny-voca is **not streaming**: it emits one waveform per utterance. Sentences are split by the native chat template, and the Kotlin layer streams the resulting PCM in 1-second chunks to the audio callback.
- Unknown/foreign words are the main source of mispronunciation (~1 in 3); the upstream dizione front end has the same limitation. Inline CMUdict pronunciations such as `Say {HH AH0 L OW1} to Samir` are supported by the template.
- A sentence without a final mark gets a full stop automatically (reference behaviour).
- `arm64-v8a` uses ggml KleidiAI; `x86_64` is for emulators and uses portable kernels.

---

## Credits & licence

- tiny-voca © 2026 Muhammed Nazeem — Apache-2.0
- dizione © 2026 Muhammed Nazeem — Apache-2.0
- llama.cpp / ggml © ggml authors — MIT
- This project — Apache-2.0 (see `LICENSE` and `NOTICE`)
