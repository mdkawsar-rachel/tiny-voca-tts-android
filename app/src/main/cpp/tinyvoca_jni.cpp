// JNI bridge to llama.cpp's native ggml tiny-voca pipeline.
//
// Performance notes:
//  * The model is loaded once per process (Engine handle) and both GGUFs are
//    memory-mapped by ggml. There is no per-utterance model load.
//  * tiny-voca is NOT autoregressive: step_prompt() encodes each sentence in
//    one graph evaluation and get_output() returns the whole 24 kHz waveform.
//    Consequently there is no token loop and no KV-cache growth per frame.
//  * The dizione text front end (numbers, dates, homographs, g2p) is embedded
//    in the GGUF chat template, so no separate phonemizer data is shipped.
//  * We request raw F32 PCM (not WAV) to avoid an encode/decode round trip.

#include <jni.h>

#include <android/log.h>

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "common.h"
#include "chat.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace {

constexpr const char * kTag = "TinyVocaJNI";

void logi(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, kTag, fmt, args);
    va_end(args);
}

void throwJava(JNIEnv * env, const std::string & message) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", message.c_str());
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) {
        env->ThrowNew(cls, message.c_str());
    }
}

std::string jstringToStd(JNIEnv * env, jstring s) {
    if (s == nullptr) {
        return {};
    }
    const char * chars = env->GetStringUTFChars(s, nullptr);
    if (chars == nullptr) {
        return {};
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(s, chars);
    return result;
}

struct Engine {
    common_init_result_ptr init;
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    mtmd_context * mctx = nullptr;
    common_chat_templates_ptr templates;
    std::mutex mutex;
};

Engine * asEngine(jlong handle) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr) {
        throw std::runtime_error("native engine handle is null");
    }
    return engine;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_tinyvoca_tts_TinyVocaEngine_nativeCreate(JNIEnv * env, jclass, jstring modelPath,
                                              jstring mmprojPath, jint threads) {
    try {
        static std::once_flag backendOnce;
        std::call_once(backendOnce, [] {
            common_init();
            llama_backend_init();
        });

        common_params params;
        params.model.path = jstringToStd(env, modelPath);
        params.mmproj.path = jstringToStd(env, mmprojPath);
        params.embedding = true;  // the audio pipeline consumes hidden states
        params.cpuparams.n_threads = threads > 0 ? threads : 4;
        params.cpuparams_batch.n_threads = threads > 0 ? threads : 4;
        params.n_ctx = 2048;       // sentences are <= 512 phonemes
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.sampling.seed = 1;  // only the source-filter noise uses it

        auto * engine = new Engine();
        engine->init = common_init_from_params(params);
        engine->model = engine->init->model();
        engine->lctx = engine->init->context();
        if (engine->model == nullptr || engine->lctx == nullptr) {
            delete engine;
            throw std::runtime_error("failed to load tiny-voca backbone GGUF");
        }

        mtmd_context_params mtmdParams = mtmd_context_params_default();
        mtmdParams.use_gpu = false;  // pure CPU: no GPU permission needed on Android
        mtmdParams.n_threads = threads > 0 ? threads : 4;
        engine->mctx = mtmd_init_from_file(params.mmproj.path.c_str(), engine->model, mtmdParams);
        if (engine->mctx == nullptr) {
            delete engine;
            throw std::runtime_error("failed to load tiny-voca mmproj GGUF");
        }

        const auto info = mtmd_gen_audio_get_info(engine->mctx);
        if (info.type != MTMD_GEN_AUDIO_TYPE_TINY_VOCA) {
            delete engine;
            throw std::runtime_error("mmproj is not a tiny-voca projection");
        }
        if (info.sample_rate != 24000) {
            delete engine;
            throw std::runtime_error("unexpected tiny-voca sample rate");
        }

        engine->templates = common_chat_templates_init(engine->model, "");
        if (!engine->templates) {
            delete engine;
            throw std::runtime_error("the GGUF has no tts chat template");
        }

        logi("ready: %d voices, %d Hz, %d threads", info.n_voices, info.sample_rate,
             engine->lctx ? params.cpuparams.n_threads : 0);
        return reinterpret_cast<jlong>(engine);
    } catch (const std::exception & e) {
        throwJava(env, std::string("tiny-voca init failed: ") + e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_tinyvoca_tts_TinyVocaEngine_nativeSynthesize(JNIEnv * env, jclass, jlong handle,
                                                  jstring text, jstring voice) {
    try {
        Engine * engine = asEngine(handle);
        std::lock_guard<std::mutex> lock(engine->mutex);

        const std::string input = jstringToStd(env, text);
        const std::string voiceName = jstringToStd(env, voice);
        if (input.empty()) {
            return env->NewFloatArray(0);
        }

        // The tts chat template is the reference dizione port.  g2p is the
        // unknown-word model stored in the mmproj; the template handles text
        // normalization, homographs, stress and sentence splitting.
        std::string prompt;
        try {
            prompt = common_chat_get_tts_prompt(
                    engine->templates.get(), input, "", voiceName,
                    [&](const std::string & word) {
                        char buf[512];
                        return mtmd_gen_audio_g2p(engine->mctx, word.c_str(), buf, sizeof(buf)) > 0
                                       ? std::string(buf)
                                       : std::string();
                    });
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("tts template failed: ") + e.what());
        }
        if (prompt.empty()) {
            prompt = input;
        }

        // tiny-voca has no voice cloning, but the pipeline picks the voice row
        // from a predefined voice bitmap (F2 / M2).
        mtmd_bitmap * voiceBitmap = nullptr;
        if (mtmd_gen_audio_get_info(engine->mctx).n_voices > 0) {
            voiceBitmap = mtmd_gen_audio_get_voice(engine->mctx, voiceName.c_str());
            if (voiceBitmap == nullptr) {
                throw std::runtime_error("unknown tiny-voca voice: " + voiceName);
            }
        }

        mtmd_helper::gen_audio gen;
        gen.init(engine->lctx, engine->mctx);
        if (!gen.valid()) {
            if (voiceBitmap) mtmd_bitmap_free(voiceBitmap);
            throw std::runtime_error("context cannot run tiny-voca audio generation");
        }

        mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
        inp.seq_id = 0;
        inp.prompt = prompt.c_str();
        inp.prompt_len = prompt.size();
        inp.speaker_ref = voiceBitmap;
        inp.lang = nullptr;
        inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        inp.stream = false;

        int32_t result = gen.set_input(&inp);
        if (result != 0) {
            if (voiceBitmap) mtmd_bitmap_free(voiceBitmap);
            throw std::runtime_error("tiny-voca rejected the prompt (code " + std::to_string(result) + ")");
        }

        // Encode every sentence.  step_prompt returns the number of prompt rows
        // still to process; each call is one non-causal transformer pass.
        while (true) {
            result = gen.step_prompt(llama_n_batch(engine->lctx));
            if (result < 0) {
                if (voiceBitmap) mtmd_bitmap_free(voiceBitmap);
                throw std::runtime_error("tiny-voca prompt encoding failed");
            }
            if (result == 0) break;
        }

        // The tiny-voca pipeline has no frame-by-frame generation: step_gen()
        // finalizes the segments and immediately reports end-of-speech.
        const float * hNext = nullptr;
        bool stop = false;
        result = gen.step_gen(LLAMA_TOKEN_NULL, nullptr, &hNext, &stop);
        if (result != 0) {
            if (voiceBitmap) mtmd_bitmap_free(voiceBitmap);
            throw std::runtime_error("tiny-voca generation failed");
        }

        int32_t sampleRate = 0;
        const char * data = nullptr;
        size_t dataLen = 0;
        int64_t nSamples = 0;
        result = gen.get_output(&sampleRate, &data, &dataLen, &nSamples);
        if (voiceBitmap) mtmd_bitmap_free(voiceBitmap);
        if (result != 0 || data == nullptr || dataLen == 0) {
            throw std::runtime_error("tiny-voca produced no audio");
        }
        if (sampleRate != 24000 || dataLen % sizeof(float) != 0) {
            throw std::runtime_error("unexpected tiny-voca PCM format");
        }

        const jsize nFloats = static_cast<jsize>(dataLen / sizeof(float));
        jfloatArray out = env->NewFloatArray(nFloats);
        if (out == nullptr) {
            throw std::runtime_error("could not allocate output audio");
        }
        env->SetFloatArrayRegion(out, 0, nFloats, reinterpret_cast<const jfloat *>(data));
        return out;
    } catch (const std::exception & e) {
        throwJava(env, std::string("tiny-voca synthesis failed: ") + e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_tinyvoca_tts_TinyVocaEngine_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr) return;
    {
        std::lock_guard<std::mutex> lock(engine->mutex);
        if (engine->mctx != nullptr) {
            mtmd_free(engine->mctx);
            engine->mctx = nullptr;
        }
    }
    engine->templates.reset();
    engine->init.reset();
    delete engine;
}
