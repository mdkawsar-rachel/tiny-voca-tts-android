package tinyvoca.tts

import android.content.Context
import android.os.Process
import android.os.SystemClock
import java.io.File
import java.util.concurrent.BlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs

/**
 * Native tiny-voca engine backed by llama.cpp / ggml.
 *
 * Why this is faster than the previous ONNX Woof engine:
 *  - tiny-voca is non-autoregressive: one graph pass per sentence instead of
 *    hundreds of autoregressive `step.run()` calls with a growing KV cache.
 *  - ggml keeps the 4.5M weights mapped and uses NEON/KleidiAI kernels on
 *    arm64-v8a; peak RSS is tens of MB instead of several hundred MB.
 *  - the waveform generator is a 25k-parameter source-filter synth, not a
 *    neural vocoder, so there is no heavy decoder stage.
 *
 * Quality is deliberately left untouched: the gguf chat template is the
 * reference dizione port, and the only post-processing is the 0.99 peak limiter
 * recommended by the model card to prevent 16-bit clipping.
 */
class TinyVocaEngine private constructor(context: Context, threads: Int) {
    val sampleRate: Int = 24000
    private val handle: Long
    private val synthLock = Any()

    init {
        val dir = File(context.filesDir, "tinyvoca_models")
        ensureModels(context, dir)
        val model = File(dir, MODEL_NAME).absolutePath
        val mmproj = File(dir, MMPROJ_NAME).absolutePath
        handle = withAudioPriority { nativeCreate(model, mmproj, threads) }
        require(handle != 0L) { "nativeCreate returned a null handle" }
    }

    /** Synthesizes arbitrary text. The native tts template splits it into
     *  sentences and returns the concatenated 24 kHz float waveform. */
    fun synthesize(text: String, voice: String = DEFAULT_VOICE): FloatArray {
        synchronized(synthLock) {
            return limitPeak(nativeSynthesize(handle, text, canonicalVoice(voice)))
        }
    }

    /**
     * Synthesizes [text] and streams it into [queue] in ~1 s chunks.
     * tiny-voca produces the whole utterance in one native call; chunking here
     * only keeps the Android SynthesisCallback playback queue healthy.
     */
    fun enqueueSpeech(
        text: String,
        voice: String,
        queue: BlockingQueue<FloatArray>,
        synthDone: AtomicBoolean,
        synthErr: AtomicReference<Throwable?>,
        isActive: () -> Boolean,
        synthMs: AtomicLong? = null
    ) {
        try {
            if (!isActive()) return
            val t0 = SystemClock.elapsedRealtime()
            val audio = synthesize(text, voice)
            synthMs?.set(SystemClock.elapsedRealtime() - t0)
            var off = 0
            while (isActive() && off < audio.size) {
                val n = minOf(CHUNK_SAMPLES, audio.size - off)
                val chunk = audio.copyOfRange(off, off + n)
                off += n
                while (isActive() && !queue.offer(chunk, 40, TimeUnit.MILLISECONDS)) {
                    // backpressure while the callback drains the queue
                }
            }
        } catch (t: Throwable) {
            synthErr.compareAndSet(null, t)
        } finally {
            synthDone.set(true)
            while (isActive() && !queue.offer(FloatArray(0), 20, TimeUnit.MILLISECONDS)) {
                // unblock the consumer on exit
            }
        }
    }

    fun close() {
        synchronized(synthLock) {
            if (handle != 0L) nativeDestroy(handle)
        }
    }

    companion object {
        private const val TAG = "TinyVocaEngine"
        private const val MODEL_NAME = "tiny-voca-F32.gguf"
        private const val MMPROJ_NAME = "mmproj-tiny-voca-F32.gguf"
        private const val CHUNK_SAMPLES = 24000
        const val DEFAULT_VOICE = "F2"
        const val VOICE_F2 = "F2"
        const val VOICE_M2 = "M2"

        init {
            System.loadLibrary("tinyvoca_jni")
        }

        @JvmStatic private external fun nativeCreate(modelPath: String, mmprojPath: String, threads: Int): Long
        @JvmStatic private external fun nativeSynthesize(handle: Long, text: String, voice: String): FloatArray
        @JvmStatic private external fun nativeDestroy(handle: Long)

        /** Accepts the new F2/M2 names plus the old Juno/Mira preferences so an
         *  in-place upgrade keeps the user's selected voice gender. */
        fun canonicalVoice(voice: String?): String = when (voice?.trim()?.lowercase()) {
            "m2", "juno", "male" -> VOICE_M2
            else -> VOICE_F2
        }

        /** Model-card limiter: tiny-voca peaks above 1.0 about once in twenty
         *  sentences. This is the exact scaling the reference implementation
         *  applies before writing 16-bit samples; normal audio is unchanged. */
        private fun limitPeak(input: FloatArray): FloatArray {
            var peak = 0f
            for (v in input) {
                if (v.isNaN() || v.isInfinite()) continue
                val a = abs(v)
                if (a > peak) peak = a
            }
            if (peak <= 0.99f) return input
            val scale = 0.99f / peak
            for (i in input.indices) {
                val v = input[i]
                input[i] = if (v.isNaN() || v.isInfinite()) 0f else v * scale
            }
            return input
        }

        private fun ensureModels(context: Context, dir: File) {
            dir.mkdirs()
            val apkStamp = File(context.applicationInfo.sourceDir).let { "${it.length()}:${it.lastModified()}" }
            val stampFile = File(dir, ".stamp")
            val stampValid = stampFile.exists() && runCatching { stampFile.readText().trim() == apkStamp }.getOrDefault(false)
            var updated = false

            for (name in listOf(MODEL_NAME, MMPROJ_NAME)) {
                val out = File(dir, name)
                val expectedLen = try {
                    context.assets.openFd("models/$name").use { it.length }
                } catch (_: Exception) {
                    -1L
                }
                val valid = stampValid && if (expectedLen > 0) {
                    out.exists() && out.length() == expectedLen
                } else {
                    out.exists() && out.length() > 1024 * 1024
                }
                if (valid) continue
                updated = true
                val tmp = File(dir, "$name.tmp")
                context.assets.open("models/$name").use { input ->
                    tmp.outputStream().use { output -> input.copyTo(output) }
                }
                if (out.exists()) out.delete()
                tmp.renameTo(out)
            }
            if (updated || !stampValid) {
                runCatching { stampFile.writeText(apkStamp) }
            }
        }

        private fun defaultThreads(): Int =
            Runtime.getRuntime().availableProcessors().coerceIn(2, 4)

        private var sharedEngine: TinyVocaEngine? = null
        private var sharedRefs = 0
        private val sharedLock = Any()

        fun acquireShared(context: Context): TinyVocaEngine = synchronized(sharedLock) {
            val app = context.applicationContext
            val engine = sharedEngine ?: TinyVocaEngine(app, defaultThreads()).also { sharedEngine = it }
            sharedRefs++
            engine
        }

        fun releaseShared() {
            val toClose = synchronized(sharedLock) {
                if (sharedRefs > 0) sharedRefs--
                if (sharedRefs == 0) {
                    sharedEngine.also { sharedEngine = null }
                } else null
            }
            toClose?.close()
        }

        private inline fun <T> withAudioPriority(block: () -> T): T {
            val tid = Process.myTid()
            val prev = Process.getThreadPriority(tid)
            var raised = false
            for (prio in intArrayOf(
                Process.THREAD_PRIORITY_AUDIO,
                Process.THREAD_PRIORITY_URGENT_DISPLAY,
                Process.THREAD_PRIORITY_DISPLAY
            )) {
                if (runCatching { Process.setThreadPriority(tid, prio) }.isSuccess) {
                    raised = true
                    break
                }
            }
            return try {
                block()
            } finally {
                if (raised) runCatching { Process.setThreadPriority(tid, prev) }
            }
        }
    }
}
