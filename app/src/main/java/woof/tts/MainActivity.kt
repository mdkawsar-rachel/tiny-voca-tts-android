package woof.tts

import android.app.Activity
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.SystemClock
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import woof.tts.text.Normalizer
import woof.tts.text.Respeller
import woof.tts.text.TextSplit
import java.util.concurrent.Executors

/** Settings + on-device preview screen.
 *  Uses WoofEngine directly for preview playback via AudioTrack to guarantee 100% genuine Underdog Woof audio. */
class MainActivity : Activity() {
    private var selectedVoice = "juno"
    private lateinit var input: EditText
    private lateinit var status: TextView
    private lateinit var speakBtn: Button
    private lateinit var stopBtn: Button

    private val prefs get() = getSharedPreferences("woof", MODE_PRIVATE)
    private val executor = Executors.newSingleThreadExecutor()
    private var engine: WoofEngine? = null
    private var normalizer: Normalizer? = null
    private var audioTrack: AudioTrack? = null
    @Volatile private var isPlaying = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        selectedVoice = prefs.getString("voice", "juno") ?: "juno"

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }
        val scroll = ScrollView(this).apply { addView(root) }
        setContentView(scroll)

        root.addView(TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 26f
            setPadding(0, 0, 0, 8)
        })
        root.addView(TextView(this).apply {
            text = getString(R.string.credit)
            textSize = 13f
            setPadding(0, 0, 0, 24)
        })

        root.addView(TextView(this).apply { text = "Voice"; textSize = 16f; setPadding(0, 0, 0, 8) })
        val radio = RadioGroup(this)
        val juno = RadioButton(this).apply { text = "Juno (default - male)"; id = 1 }
        val mira = RadioButton(this).apply { text = "Mira (female)"; id = 2 }
        radio.addView(juno)
        radio.addView(mira)
        radio.check(if (selectedVoice.equals("mira", ignoreCase = true)) 2 else 1)
        radio.setOnCheckedChangeListener { _, id ->
            selectedVoice = if (id == 2) "mira" else "juno"
            prefs.edit().putString("voice", selectedVoice).apply()
            status.text = "Voice set to ${selectedVoice.replaceFirstChar { it.uppercase() }}."
        }
        root.addView(radio)

        root.addView(TextView(this).apply { text = "Test text"; textSize = 16f; setPadding(0, 24, 0, 8) })
        input = EditText(this).apply {
            hint = "Type something to speak"
            setText("Please call me back at 415-555-0199 before 5:30 p.m. on Friday.")
            minLines = 3
        }
        root.addView(input)

        val btnLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 16, 0, 16)
        }

        speakBtn = Button(this).apply {
            text = "Speak with Woof"
            isEnabled = false
            setOnClickListener {
                speakText()
            }
        }
        btnLayout.addView(speakBtn)

        stopBtn = Button(this).apply {
            text = "Stop"
            isEnabled = false
            setOnClickListener {
                stopPlayback()
            }
        }
        btnLayout.addView(stopBtn)
        root.addView(btnLayout)

        status = TextView(this).apply { text = "Initializing Woof TTS engine…"; setPadding(0, 8, 0, 16) }
        root.addView(status)

        root.addView(Button(this).apply {
            text = "Open system TTS settings"
            setOnClickListener {
                try {
                    startActivity(Intent("com.android.settings.TTS_SETTINGS"))
                } catch (_: Exception) {
                    status.text = "Open Settings → System → Languages & input → Text-to-speech output."
                }
            }
        })

        root.addView(TextView(this).apply {
            text = getString(R.string.howto)
            textSize = 13f
            setPadding(0, 24, 0, 0)
        })

        // Initialize Woof engine in background thread
        executor.execute {
            try {
                val eng = WoofEngine(applicationContext)
                val respeller = Respeller().apply {
                    load(assets.open("respell.tsv").bufferedReader().use { it.readText() })
                }
                val norm = Normalizer(respeller)
                synchronized(this@MainActivity) {
                    engine = eng
                    normalizer = norm
                }
                runOnUiThread {
                    speakBtn.isEnabled = true
                    status.text = "Woof engine ready (${selectedVoice.replaceFirstChar { it.uppercase() }} voice)."
                }
            } catch (e: Exception) {
                runOnUiThread {
                    status.text = "Failed to load Woof engine: ${e.message}"
                }
            }
        }
    }

    private fun speakText() {
        val text = input.text.toString().ifBlank { "Hello! This is Underdog Woof text to speech." }
        stopPlayback()
        isPlaying = true
        speakBtn.isEnabled = false
        stopBtn.isEnabled = true
        val voice = selectedVoice

        executor.execute {
            try {
                runOnUiThread { status.text = "Synthesizing with Woof ($voice)…" }
                val eng: WoofEngine
                val norm: Normalizer
                synchronized(this@MainActivity) {
                    eng = engine ?: return@execute
                    norm = normalizer ?: return@execute
                }
                val t0 = SystemClock.elapsedRealtime()
                val spoken = norm(text)
                val pieces = TextSplit.splitText(spoken)
                val allAudio = ArrayList<FloatArray>()
                for (piece in pieces) {
                    if (!isPlaying) break
                    if (piece.isBlank()) continue
                    val pcm = eng.synthesize(piece, voice)
                    if (pcm.isNotEmpty()) {
                        allAudio.add(pcm)
                    }
                }
                val synthTimeMs = SystemClock.elapsedRealtime() - t0
                if (!isPlaying || allAudio.isEmpty()) {
                    runOnUiThread {
                        speakBtn.isEnabled = true
                        stopBtn.isEnabled = false
                        status.text = "Ready."
                    }
                    return@execute
                }

                val totalSamples = allAudio.sumOf { it.size }
                val combined = FloatArray(totalSamples)
                var offset = 0
                for (chunk in allAudio) {
                    System.arraycopy(chunk, 0, combined, offset, chunk.size)
                    offset += chunk.size
                }

                val audioDurationSec = totalSamples.toFloat() / eng.sampleRate
                val rtf = if (synthTimeMs > 0) (audioDurationSec * 1000f) / synthTimeMs else 1f

                runOnUiThread {
                    status.text = String.format(
                        "Playing %s voice (%.1fs audio in %dms, %.1fx RTF)…",
                        voice.replaceFirstChar { it.uppercase() },
                        audioDurationSec,
                        synthTimeMs,
                        rtf
                    )
                }

                playAudio(combined, eng.sampleRate)
            } catch (e: Exception) {
                runOnUiThread {
                    status.text = "Synthesis error: ${e.message}"
                    speakBtn.isEnabled = true
                    stopBtn.isEnabled = false
                }
            }
        }
    }

    private fun playAudio(samples: FloatArray, sampleRate: Int) {
        try {
            val minBuf = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_FLOAT
            )
            val bufferSize = maxOf(minBuf, samples.size * 4)
            val track = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()

            synchronized(this) {
                audioTrack = track
            }

            track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            track.play()

            val durationMs = ((samples.size.toDouble() / sampleRate) * 1000).toLong()
            val start = SystemClock.elapsedRealtime()
            while (isPlaying && (SystemClock.elapsedRealtime() - start) < durationMs + 200) {
                Thread.sleep(50)
            }
        } finally {
            runOnUiThread {
                speakBtn.isEnabled = true
                stopBtn.isEnabled = false
                if (isPlaying) {
                    status.text = "Ready."
                }
            }
            isPlaying = false
        }
    }

    private fun stopPlayback() {
        isPlaying = false
        synchronized(this) {
            try {
                audioTrack?.stop()
                audioTrack?.release()
                audioTrack = null
            } catch (_: Exception) {}
        }
        speakBtn.isEnabled = true
        stopBtn.isEnabled = false
    }

    override fun onDestroy() {
        stopPlayback()
        executor.shutdownNow()
        engine?.close()
        super.onDestroy()
    }
}
