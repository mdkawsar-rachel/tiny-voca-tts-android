import woof.tts.text.Normalizer
import woof.tts.text.Respeller
import woof.tts.text.TextSplit
import woof.tts.text.Tokenizer
import java.io.File
import java.util.Base64

fun main(args: Array<String>) {
    val fixtures = File(args[0])
    val respellPath = args[1]
    val norm = Normalizer(Respeller.fromFile(respellPath))
    var pass = 0
    var fail = 0
    for ((idx, line) in fixtures.readLines().withIndex()) {
        if (line.isBlank()) continue
        val p = line.split("\t")
        fun dec(i: Int) = String(Base64.getDecoder().decode(p[i]), Charsets.UTF_8)
        val text = dec(0)
        val expNorm = dec(1)
        val expSplit = if (dec(2).isEmpty()) emptyList() else dec(2).split("\n")
        val expTok = dec(3).split(",").map { it.toInt() }
        val gotNorm = norm(text)
        val gotSplit = TextSplit.splitText(gotNorm)
        val gotTok = Tokenizer.encodeText(gotNorm).toList()
        var ok = true
        if (gotNorm != expNorm) { println("[$idx] NORM\n  in:  $text\n  exp: $expNorm\n  got: $gotNorm"); ok = false }
        if (gotSplit != expSplit) { println("[$idx] SPLIT\n  exp: $expSplit\n  got: $gotSplit"); ok = false }
        if (gotTok != expTok) { println("[$idx] TOKEN\n  exp: $expTok\n  got: $gotTok"); ok = false }
        if (ok) pass++ else fail++
    }
    println("PASS=$pass FAIL=$fail")
    if (fail > 0) kotlin.system.exitProcess(1)
}