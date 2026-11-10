package woof.tts

import org.junit.Assert.assertEquals
import org.junit.Test
import woof.tts.text.Normalizer
import woof.tts.text.Respeller
import woof.tts.text.TextSplit
import woof.tts.text.Tokenizer
import java.io.File
import java.util.Base64

class TextFrontendTest {
    @Test
    fun parityWithPython() {
        val fixtures = File(System.getProperty("fixtures") ?: error("fixtures property not set"))
        val respell = System.getProperty("respell") ?: error("respell property not set")
        val norm = Normalizer(Respeller.fromFile(respell))
        val failures = mutableListOf<String>()
        var total = 0
        for ((idx, line) in fixtures.readLines().withIndex()) {
            if (line.isBlank()) continue
            total++
            val p = line.split("\t")
            fun dec(i: Int) = String(Base64.getDecoder().decode(p[i]), Charsets.UTF_8)
            val text = dec(0)
            val expNorm = dec(1)
            val expSplit = if (dec(2).isEmpty()) emptyList() else dec(2).split("\n")
            val expTok = dec(3).split(",").map { it.toInt() }
            val gotNorm = norm(text)
            if (gotNorm != expNorm) failures.add("[$idx] NORM in=$text exp=$expNorm got=$gotNorm")
            val gotSplit = TextSplit.splitText(gotNorm)
            if (gotSplit != expSplit) failures.add("[$idx] SPLIT exp=$expSplit got=$gotSplit")
            val gotTok = Tokenizer.encodeText(gotNorm).toList()
            if (gotTok != expTok) failures.add("[$idx] TOKEN exp=$expTok got=$gotTok")
        }
        assertEquals("Kotlin frontend diverges from Python on ${failures.size}/$total fixtures:\n${failures.take(20).joinToString("\n")}", 0, failures.size)
    }
}