package woof.tts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import woof.tts.text.Normalizer
import woof.tts.text.Respeller
import woof.tts.text.TextSplit
import java.util.Locale

/** Android system TTS engine backed by Woof. English only, fully offline. */
class WoofTtsService : TextToSpeechService() {
    @Volatile private var engineInstance: WoofEngine? = null
    @Volatile private var normInstance: Normalizer? = null
    @Volatile private var stopped = false

    private val prefs get() = getSharedPreferences("woof", MODE_PRIVATE)

    private fun engine(): WoofEngine {
        engineInstance?.let { return it }
        return synchronized(this) {
            engineInstance ?: WoofEngine(applicationContext).also { engineInstance = it }
        }
    }

    private fun normalizer(): Normalizer {
        normInstance?.let { return it }
        return synchronized(this) {
            normInstance ?: Normalizer(Respeller().apply {
                load(assets.open("respell.tsv").bufferedReader().use { it.readText() })
            }).also { normInstance = it }
        }
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
        return if (voiceName == "juno" || voiceName == "mira") TextToSpeech.SUCCESS else TextToSpeech.ERROR
    }

    override fun onLoadVoice(voiceName: String?): Int {
        return if (voiceName == "juno" || voiceName == "mira") TextToSpeech.SUCCESS else TextToSpeech.ERROR
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        stopped = false
        val text = request.charSequenceText?.toString() ?: request.text ?: ""
        val voiceName = request.voiceName?.takeIf { it == "juno" || it == "mira" }
            ?: prefs.getString("voice", "juno") ?: "juno"
        try {
            val engine = engine()
            callback.start(engine.sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)
            if (text.isBlank()) {
                callback.done()
                return
            }
            val spoken = normalizer()(text)
            val pieces = TextSplit.splitText(spoken)
            val maxSamples = (callback.maxBufferSize / 2).coerceAtLeast(1)
            for (piece in pieces) {
                if (stopped) break
                if (piece.isBlank()) continue
                val audio = engine.synthesize(piece, voiceName)
                var off = 0
                while (off < audio.size && !stopped) {
                    val n = minOf(maxSamples, audio.size - off)
                    val buf = ByteArray(n * 2)
                    for (i in 0 until n) {
                        val v = (audio[off + i] * 32767f).toInt().coerceIn(-32768, 32767)
                        buf[i * 2] = (v and 0xff).toByte()
                        buf[i * 2 + 1] = ((v shr 8) and 0xff).toByte()
                    }
                    if (callback.audioAvailable(buf, 0, buf.size) != TextToSpeech.SUCCESS) break
                    off += n
                }
            }
            if (callback.hasStarted() && !callback.hasFinished()) {
                callback.done()
            }
        } catch (e: Exception) {
            Log.e(TAG, "synthesis failed", e)
            if (callback.hasStarted() && !callback.hasFinished()) {
                callback.done()
            } else {
                callback.error()
            }
        }
    }

    override fun onDestroy() {
        engineInstance?.close()
        engineInstance = null
        super.onDestroy()
    }

    companion object {
        private const val TAG = "WoofTTS"
    }
}