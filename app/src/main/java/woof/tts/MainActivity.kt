package woof.tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import java.util.Locale

/** Settings + preview screen. The preview speaks through the real WoofTtsService via the TextToSpeech client. */
class MainActivity : Activity() {
    private lateinit var tts: TextToSpeech
    private var ttsReady = false
    private var selectedVoice = "juno"
    private lateinit var input: EditText
    private lateinit var status: TextView
    private lateinit var speakBtn: Button

    private val prefs get() = getSharedPreferences("woof", MODE_PRIVATE)

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
        val juno = RadioButton(this).apply { text = "Juno (default)"; id = 1 }
        val mira = RadioButton(this).apply { text = "Mira"; id = 2 }
        radio.addView(juno); radio.addView(mira)
        radio.check(if (selectedVoice == "mira") 2 else 1)
        radio.setOnCheckedChangeListener { _, id ->
            selectedVoice = if (id == 2) "mira" else "juno"
            prefs.edit().putString("voice", selectedVoice).apply()
            applyVoice()
        }
        root.addView(radio)

        root.addView(TextView(this).apply { text = "Test text"; textSize = 16f; setPadding(0, 24, 0, 8) })
        input = EditText(this).apply {
            hint = "Type something to speak"
            setText("Please call me back at 415-555-0199 before 5:30 p.m. on Friday.")
            minLines = 3
        }
        root.addView(input)

        speakBtn = Button(this).apply {
            text = "Speak"
            isEnabled = false
            setOnClickListener {
                if (!ttsReady) return@setOnClickListener
                applyVoice()
                val text = input.text.toString().ifBlank { "Hello! This is Woof text to speech." }
                status.text = "Loading engine / synthesizing…"
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "woof_preview")
            }
        }
        root.addView(speakBtn)

        status = TextView(this).apply { text = "Initializing…"; setPadding(0, 16, 0, 16) }
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

        tts = TextToSpeech(this, { initStatus ->
            if (initStatus == TextToSpeech.SUCCESS) {
                val res = tts.setLanguage(Locale.US)
                ttsReady = res >= TextToSpeech.LANG_AVAILABLE
                applyVoice()
                runOnUiThread {
                    speakBtn.isEnabled = ttsReady
                    status.text = if (ttsReady) "Ready." else "English data unavailable."
                }
            } else {
                runOnUiThread { status.text = "TTS initialization failed." }
            }
        }, packageName)

        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) { runOnUiThread { status.text = "Speaking…" } }
            override fun onDone(utteranceId: String?) { runOnUiThread { status.text = "Ready." } }
            @Deprecated("deprecated in Java")
            override fun onError(utteranceId: String?) { runOnUiThread { status.text = "Synthesis error." } }
        })
    }

    private fun applyVoice() {
        if (!ttsReady) return
        val v = tts.voices?.firstOrNull { it.name == selectedVoice }
        if (v != null) tts.voice = v
    }

    override fun onDestroy() {
        tts.stop()
        tts.shutdown()
        super.onDestroy()
    }
}