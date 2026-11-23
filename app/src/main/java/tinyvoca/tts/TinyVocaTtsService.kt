package tinyvoca.tts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Android system TTS engine backed by the native ggml tiny-voca pipeline. */
class TinyVocaTtsService : TextToSpeechService() {
    private var engine: TinyVocaEngine? = null
    private val synthExecutor = Executors.newSingleThreadExecutor()
    private val requestGen = AtomicInteger(0)
    @Volatile private var stopped = false

    private val prefs get() = getSharedPreferences("tinyvoca", MODE_PRIVATE)

    private fun acquireEngine(): TinyVocaEngine = synchronized(this) {
        engine ?: TinyVocaEngine.acquireShared(applicationContext).also { engine = it }
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        val l = lang?.lowercase(Locale.ROOT) ?: return TextToSpeech.LANG_NOT_SUPPORTED
        if (l != "en" && l != "eng") return TextToSpeech.LANG_NOT_SUPPORTED
        return if (country.isNullOrEmpty()) TextToSpeech.LANG_AVAILABLE else TextToSpeech.LANG_COUNTRY_AVAILABLE
    }

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int =
        onIsLanguageAvailable(lang, country, variant)

    override fun onGetLanguage(): Array<String> = arrayOf("eng", "USA", "")

    override fun onStop() {
        stopped = true
        requestGen.incrementAndGet()
    }

    override fun onGetVoices(): MutableList<Voice> = mutableListOf(
        Voice(TinyVocaEngine.VOICE_F2, Locale.US, Voice.QUALITY_NORMAL, Voice.LATENCY_NORMAL, false, setOf("female")),
        Voice(TinyVocaEngine.VOICE_M2, Locale.US, Voice.QUALITY_NORMAL, Voice.LATENCY_NORMAL, false, setOf("male"))
    )

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String {
        val l = lang?.lowercase(Locale.ROOT)
        if (l != null && l != "en" && l != "eng") return ""
        return TinyVocaEngine.canonicalVoice(prefs.getString("voice", TinyVocaEngine.DEFAULT_VOICE))
    }

    override fun onIsValidVoiceName(voiceName: String?): Int =
        if (voiceName.equals(TinyVocaEngine.VOICE_F2, true) ||
            voiceName.equals(TinyVocaEngine.VOICE_M2, true)) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String?): Int = onIsValidVoiceName(voiceName)

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val gen = requestGen.incrementAndGet()
        stopped = false
        val text = request.charSequenceText?.toString() ?: request.text ?: ""
        val paramVoice = request.params?.getString("voiceName")
            ?: request.params?.getString("voice")
        val requested = when {
            !paramVoice.isNullOrBlank() -> paramVoice
            else -> request.voiceName
        }
        val prefVoice = prefs.getString("voice", TinyVocaEngine.DEFAULT_VOICE)
        val voice = if (!requested.isNullOrBlank() &&
            (requested.equals("F2", true) || requested.equals("M2", true) ||
             requested.equals("juno", true) || requested.equals("mira", true))
        ) TinyVocaEngine.canonicalVoice(requested)
        else TinyVocaEngine.canonicalVoice(prefVoice)

        Log.i(TAG, "onSynthesizeText voice=$voice len=${text.length}")
        try {
            val eng = acquireEngine()
            callback.start(eng.sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)
            if (text.isBlank()) {
                callback.done()
                return
            }

            val maxSamples = (callback.maxBufferSize / 2).coerceAtLeast(1)
            val queue = ArrayBlockingQueue<FloatArray>(32)
            val synthDone = AtomicBoolean(false)
            val synthErr = AtomicReference<Throwable?>(null)
            val writeFailed = AtomicBoolean(false)
            val synthMs = AtomicLong(0L)
            fun isActive() = !stopped && requestGen.get() == gen && !writeFailed.get()

            // Keep synthesis off the callback thread: AOSP blocks after ~500 ms
            // of unconsumed audio in SynthesisPlaybackQueueItem.
            synthExecutor.execute {
                eng.enqueueSpeech(text, voice, queue, synthDone, synthErr, ::isActive, synthMs)
            }

            while (isActive()) {
                val audio = queue.poll(40, TimeUnit.MILLISECONDS)
                if (audio == null) {
                    if (synthDone.get() && queue.isEmpty()) break
                    continue
                }
                if (audio.isEmpty()) break
                var off = 0
                while (off < audio.size && isActive()) {
                    val n = minOf(maxSamples, audio.size - off)
                    val buf = ByteArray(n * 2)
                    for (i in 0 until n) {
                        val v = (audio[off + i] * 32767f).toInt().coerceIn(-32768, 32767)
                        buf[i * 2] = (v and 0xff).toByte()
                        buf[i * 2 + 1] = ((v shr 8) and 0xff).toByte()
                    }
                    if (callback.audioAvailable(buf, 0, buf.size) != TextToSpeech.SUCCESS) {
                        writeFailed.set(true)
                        break
                    }
                    off += n
                }
            }
            synthErr.get()?.let { throw it }

            if (callback.hasStarted() && !callback.hasFinished()) callback.done()
        } catch (e: Throwable) {
            Log.e(TAG, "synthesis failed", e)
            if (callback.hasStarted() && !callback.hasFinished()) callback.done() else callback.error()
        }
    }

    override fun onDestroy() {
        stopped = true
        requestGen.incrementAndGet()
        synthExecutor.shutdownNow()
        val had = synchronized(this) { (engine != null).also { engine = null } }
        if (had) TinyVocaEngine.releaseShared()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "TinyVocaTTS"
    }
}
