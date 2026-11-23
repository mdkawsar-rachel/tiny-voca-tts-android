package tinyvoca.tts

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
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Settings + on-device preview. Preview uses the exact same native engine and
 *  the exact same samples that the system TTS service produces. */
class MainActivity : Activity() {
    private var selectedVoice = TinyVocaEngine.DEFAULT_VOICE
    private lateinit var input: EditText
    private lateinit var status: TextView
    private lateinit var speakBtn: Button
    private lateinit var stopBtn: Button

    private val prefs get() = getSharedPreferences("tinyvoca", MODE_PRIVATE)
    private val synthExecutor = Executors.newSingleThreadExecutor()
    private val playExecutor = Executors.newSingleThreadExecutor()
    private var engine: TinyVocaEngine? = null
    private var acquiredShared = false
    private var destroyed = false
    private var audioTrack: AudioTrack? = null
    private val playbackGen = AtomicInteger(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        selectedVoice = TinyVocaEngine.canonicalVoice(prefs.getString("voice", TinyVocaEngine.DEFAULT_VOICE))
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }
        setContentView(ScrollView(this).apply { addView(root) })

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
        val f2 = RadioButton(this).apply { text = "F2 (female)"; id = 1 }
        val m2 = RadioButton(this).apply { text = "M2 (male)"; id = 2 }
        radio.addView(f2)
        radio.addView(m2)
        radio.check(if (selectedVoice == TinyVocaEngine.VOICE_M2) 2 else 1)
        radio.setOnCheckedChangeListener { _, id ->
            selectedVoice = if (id == 2) TinyVocaEngine.VOICE_M2 else TinyVocaEngine.VOICE_F2
            prefs.edit().putString("voice", selectedVoice).apply()
            status.text = "Voice set to $selectedVoice."
        }
        root.addView(radio)

        root.addView(TextView(this).apply { text = "Test text"; textSize = 16f; setPadding(0, 24, 0, 8) })
        input = EditText(this).apply {
            hint = "Type something to speak"
            setText("Please call me back at 415-555-0199 before 5:30 p.m. on Friday.")
            minLines = 3
        }
        root.addView(input)

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 16, 0, 16) }
        speakBtn = Button(this).apply {
            text = "Speak with tiny-voca"
            isEnabled = false
            setOnClickListener { speakText() }
        }
        stopBtn = Button(this).apply {
            text = "Stop"
            isEnabled = false
            setOnClickListener { stopPlayback() }
        }
        buttons.addView(speakBtn)
        buttons.addView(stopBtn)
        root.addView(buttons)

        status = TextView(this).apply { text = "Initializing native tiny-voca engine…"; setPadding(0, 8, 0, 16) }
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

        synthExecutor.execute {
            try {
                val eng = TinyVocaEngine.acquireShared(applicationContext)
                val active = synchronized(this@MainActivity) {
                    if (destroyed) false else { engine = eng; acquiredShared = true; true }
                }
                if (!active) {
                    TinyVocaEngine.releaseShared()
                    return@execute
                }
                runOnUiThread {
                    speakBtn.isEnabled = true
                    status.text = "tiny-voca ready ($selectedVoice voice)."
                }
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Failed to load tiny-voca: ${e.message}" }
            }
        }
    }

    private fun speakText() {
        val text = input.text.toString().ifBlank { "Hello! This is tiny voca." }
        stopPlayback(updateStatus = false)
        val gen = playbackGen.incrementAndGet()
        val cancelled = AtomicBoolean(false)
        fun isActive() = !cancelled.get() && playbackGen.get() == gen
        speakBtn.isEnabled = false
        stopBtn.isEnabled = true
        val voice = selectedVoice
        val eng = synchronized(this) { engine }
        if (eng == null) {
            speakBtn.isEnabled = false
            stopBtn.isEnabled = false
            return
        }

        val t0 = SystemClock.elapsedRealtime()
        val queue = ArrayBlockingQueue<FloatArray>(12)
        val synthDone = AtomicBoolean(false)
        val synthMs = AtomicLong(0L)
        val synthErr = AtomicReference<Throwable?>(null)

        synthExecutor.execute {
            if (!isActive()) {
                synthDone.set(true)
                queue.offer(FloatArray(0))
                return@execute
            }
            runOnUiThread { if (isActive()) status.text = "Synthesizing with tiny-voca ($voice)…" }
            eng.enqueueSpeech(text, voice, queue, synthDone, synthErr, ::isActive, synthMs)
        }

        playExecutor.execute {
            var track: AudioTrack? = null
            try {
                if (!isActive()) return@execute
                val minBuf = AudioTrack.getMinBufferSize(
                    eng.sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
                )
                val bufferSize = maxOf(minBuf, eng.sampleRate)
                track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .setSampleRate(eng.sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()
                if (track.state != AudioTrack.STATE_INITIALIZED) {
                    throw IllegalStateException("AudioTrack failed to initialize (state=${track.state})")
                }
                synchronized(this@MainActivity) {
                    if (!isActive()) { track.release(); track = null; return@execute }
                    audioTrack = track
                }

                var totalSamples = 0L
                var started = false
                var ttfaMs = 0L
                while (isActive()) {
                    val chunk = queue.poll(40, TimeUnit.MILLISECONDS)
                    if (chunk == null) {
                        if (synthDone.get() && queue.isEmpty()) break
                        continue
                    }
                    if (chunk.isEmpty()) break
                    if (!started) {
                        started = true
                        ttfaMs = SystemClock.elapsedRealtime() - t0
                        track.play()
                    }
                    totalSamples += chunk.size
                    val audioSec = totalSamples.toFloat() / eng.sampleRate
                    val sMs = synthMs.get().coerceAtLeast(1L)
                    val rtf = (audioSec * 1000f) / sMs
                    runOnUiThread {
                        if (isActive()) {
                            status.text = String.format(
                                "Playing %s (TTFA %dms | %.1fs audio in %dms, %.1fx real-time)…",
                                voice, ttfaMs, audioSec, sMs, rtf
                            )
                        }
                    }
                    var off = 0
                    while (off < chunk.size && isActive()) {
                        val written = track.write(chunk, off, minOf(4096, chunk.size - off), AudioTrack.WRITE_BLOCKING)
                        if (written <= 0) break
                        off += written
                    }
                }
                synthErr.get()?.let { throw it }
                if (started && isActive()) {
                    track.stop()
                    val deadline = SystemClock.elapsedRealtime() + 1000L
                    while (isActive() && track.playbackHeadPosition < totalSamples && SystemClock.elapsedRealtime() < deadline) {
                        Thread.sleep(30)
                    }
                }
                val audioSec = totalSamples.toFloat() / eng.sampleRate
                val sMs = synthMs.get().coerceAtLeast(1L)
                val completed = isActive()
                runOnUiThread {
                    if (completed || playbackGen.get() == gen) {
                        status.text = if (completed) {
                            String.format(
                                "Finished %s (%.1fs audio, TTFA %dms, synth %dms, %.1fx real-time).",
                                voice, audioSec, ttfaMs, sMs, (audioSec * 1000f) / sMs
                            )
                        } else "Stopped."
                    }
                }
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Synthesis error: ${e.message}" }
            } finally {
                cancelled.set(true)
                synchronized(this@MainActivity) { if (audioTrack === track) audioTrack = null }
                try { track?.pause(); track?.flush(); track?.release() } catch (_: Exception) {}
                runOnUiThread {
                    if (playbackGen.get() == gen) {
                        speakBtn.isEnabled = engine != null
                        stopBtn.isEnabled = false
                    }
                }
            }
        }
    }

    private fun stopPlayback(updateStatus: Boolean = true) {
        playbackGen.incrementAndGet()
        synchronized(this) {
            try { audioTrack?.pause(); audioTrack?.flush() } catch (_: Exception) {}
        }
        speakBtn.isEnabled = engine != null
        stopBtn.isEnabled = false
        if (updateStatus && ::status.isInitialized) status.text = "Stopped."
    }

    override fun onDestroy() {
        val shouldRelease: Boolean
        synchronized(this) {
            destroyed = true
            shouldRelease = acquiredShared
            acquiredShared = false
            engine = null
        }
        stopPlayback()
        synthExecutor.shutdownNow()
        playExecutor.shutdownNow()
        runCatching { synthExecutor.awaitTermination(500, TimeUnit.MILLISECONDS) }
        if (shouldRelease) TinyVocaEngine.releaseShared()
        super.onDestroy()
    }
}
