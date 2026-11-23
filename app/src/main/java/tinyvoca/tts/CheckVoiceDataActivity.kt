package tinyvoca.tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech

/** Answers CHECK_TTS_DATA / GET_SAMPLE_TEXT queries from Android Settings. */
class CheckVoiceDataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (intent?.action == "android.speech.tts.engine.GET_SAMPLE_TEXT") {
            setResult(TextToSpeech.LANG_AVAILABLE, Intent().apply {
                putExtra("sampleText", "This is an example of speech synthesis using tiny-voca.")
            })
        } else {
            setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, Intent().apply {
                putStringArrayListExtra(
                    TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES,
                    arrayListOf("eng-USA", "en-US", "en")
                )
                putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, arrayListOf())
            })
        }
        finish()
    }
}
