package woof.tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech

/** Handles CHECK_TTS_DATA and GET_SAMPLE_TEXT queries from Android Settings.
 *  Many Android ROMs require these intents to be answered before listing a TTS engine in System Settings. */
class CheckVoiceDataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val action = intent?.action
        if (action == "android.speech.tts.engine.GET_SAMPLE_TEXT") {
            val returnData = Intent().apply {
                putExtra("sampleText", "This is an example of speech synthesis using Underdog Woof.")
            }
            setResult(TextToSpeech.LANG_AVAILABLE, returnData)
        } else {
            // CHECK_TTS_DATA
            val returnData = Intent().apply {
                putStringArrayListExtra(
                    TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES,
                    arrayListOf("eng-USA", "en-US", "en")
                )
                putStringArrayListExtra(
                    TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES,
                    arrayListOf()
                )
            }
            setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, returnData)
        }
        finish()
    }
}
