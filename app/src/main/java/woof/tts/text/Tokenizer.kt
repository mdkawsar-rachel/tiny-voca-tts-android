package woof.tts.text

object Tokenizer {
    const val BOS = 1
    const val UNK = 2
    private val CHARS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()&/%\$€£@#+=*_<>[]" + "…—"
    val VOCAB: Map<Char, Int> = CHARS.withIndex().associate { (i, c) -> c to i + 3 }

    fun encodeText(s: String): IntArray = intArrayOf(BOS) + s.map { VOCAB[it] ?: UNK }.toIntArray()
}