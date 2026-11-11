package woof.tts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import woof.tts.text.Normalizer
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/** Android system TTS engine backed by Woof. English only, fully offline. */
class WoofTtsService : TextToSpeechService() {
    private var sharedResources: Pair<WoofEngine, Normalizer>? = null
    private val synthExecutor = Executors.newSingleThreadExecutor()
    private val requestGen = AtomicInteger(0)
    @Volatile private var stopped = false

    private val prefs get() = getSharedPreferences("woof", MODE_PRIVATE)

    private fun acquireResources(): Pair<WoofEngine, Normalizer> = synchronized(this) {
        sharedResources ?: WoofEngine.acquireShared(applicationContext).also { sharedResources = it }
    }

    private fun langAvailable(lang: String?, country: String?): Int {
        val l = lang?.lowercase(Locale.ROOT) ?: return TextToSpeech.LANG_NOT_SUPPORTED
        if (l != "en" && l != "eng") return TextToSpeech.LANG_NOT_SUPPORTED
        return if (country.isNullOrEmpty()) TextToSpeech.LANG_AVAILABLE else TextToSpeech.LANG_COUNTRY_AVAILABLE
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int = langAvailable(lang, country)

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int = langAvailable(lang, country)

    override fun onGetLanguage(): Array<String> = arrayOf("eng", "USA", "")

    override fun onStop() {
        stopped = true
        requestGen.incrementAndGet()
    }

    override fun onGetVoices(): MutableList<Voice> {
        return mutableListOf(
            Voice("juno", Locale.US, Voice.QUALITY_VERY_HIGH, Voice.LATENCY_NORMAL, false, setOf("male")),
            Voice("mira", Locale.US, Voice.QUALITY_VERY_HIGH, Voice.LATENCY_NORMAL, false, setOf("female"))
        )
    }

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String {
        val l = lang?.lowercase(Locale.ROOT)
        if (l != null && l != "en" && l != "eng") return ""
        return prefs.getString("voice", "juno") ?: "juno"
    }

    override fun onIsValidVoiceName(voiceName: String?): Int {
        return if (voiceName.equals("juno", true) || voiceName.equals("mira", true)) TextToSpeech.SUCCESS else TextToSpeech.ERROR
    }

    override fun onLoadVoice(voiceName: String?): Int {
        return if (voiceName.equals("juno", true) || voiceName.equals("mira", true)) TextToSpeech.SUCCESS else TextToSpeech.ERROR
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val gen = requestGen.incrementAndGet()
        stopped = false
        val text = request.charSequenceText?.toString() ?: request.text ?: ""
        val paramVoice = request.params?.getString("voiceName")
            ?: request.params?.getString("voice")
        val reqVoice = request.voiceName
        val prefVoice = prefs.getString("voice", "juno") ?: "juno"

        val voiceName = when {
            !paramVoice.isNullOrBlank() -> paramVoice
            !reqVoice.isNullOrBlank() && (reqVoice.equals("mira", true) || reqVoice.equals("juno", true)) -> reqVoice
            else -> prefVoice
        }
        Log.i(TAG, "onSynthesizeText voice=$voiceName len=${text.length}")
        try {
            val (engine, normalizer) = acquireResources()
            callback.start(engine.sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)
            if (text.isBlank()) {
                callback.done()
                return
            }
            val maxSamples = (callback.maxBufferSize / 2).coerceAtLeast(1)
            val queue = ArrayBlockingQueue<FloatArray>(32)
            val synthDone = AtomicBoolean(false)
            val synthErr = AtomicReference<Throwable?>(null)
            val writeFailed = AtomicBoolean(false)
            fun isActive() = !stopped && requestGen.get() == gen && !writeFailed.get()

            // Decouple synthesis from callback.audioAvailable() because AOSP SynthesisPlaybackQueueItem
            // blocks after ~500ms of unconsumed audio, which otherwise stalls ONNX chunk generation.
            synthExecutor.execute {
                engine.enqueueSpeech(normalizer, text, voiceName, queue, synthDone, synthErr, ::isActive)
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

            if (callback.hasStarted() && !callback.hasFinished()) {
                callback.done()
            }
        } catch (e: Throwable) {
            Log.e(TAG, "synthesis failed", e)
            if (callback.hasStarted() && !callback.hasFinished()) {
                callback.done()
            } else {
                callback.error()
            }
        }
    }

    override fun onDestroy() {
        stopped = true
        requestGen.incrementAndGet()
        synthExecutor.shutdownNow()
        val had = synchronized(this) {
            (sharedResources != null).also { sharedResources = null }
        }
        if (had) WoofEngine.releaseShared()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "WoofTTS"
    }
}