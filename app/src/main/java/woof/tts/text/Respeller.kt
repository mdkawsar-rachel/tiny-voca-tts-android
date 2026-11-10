package woof.tts.text

import java.io.File
import java.text.Normalizer as JNormalizer

fun fold(s: String): String {
    var x = JNormalizer.normalize(s.replace("’", "'").replace("‘", "'"), JNormalizer.Form.NFKD)
    x = x.filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }.lowercase()
    x = x.replace("ł", "l").replace("ø", "o").replace("æ", "ae").replace("œ", "oe").replace("ß", "ss").replace("đ", "d")
    return rx("[-‐–]").replace(x, " ").words().joinToString(" ")
}

class Respeller {
    val lex = HashMap<String, Pair<String, Boolean>>()
    var maxn = 1

    fun load(content: String) {
        for (line in content.lineSequence()) {
            if (line.isBlank() || line.startsWith("#")) continue
            val p = line.trimEnd('\n').split("\t")
            val key = fold(p[0])
            lex[key] = p[1] to ("c" in (if (p.size > 2) p[2] else ""))
            maxn = maxOf(maxn, key.count { it == ' ' } + 1)
        }
    }

    fun lookup(src: String): Pair<Pair<String, Boolean>, String>? {
        val key = fold(src)
        val hit = lex[key]
        if (hit != null) return hit to ""
        if (key.endsWith("'s") && lex.containsKey(key.dropLast(2))) return lex[key.dropLast(2)]!! to src.takeLast(2)
        return null
    }

    private fun hyphenOk(t: String, a: Int, b: Int, src: String): Boolean {
        if (a >= 1 && t[a - 1] == '-' && a >= 2 && t[a - 2].isLetter()) {
            val other = t.substring(0, a - 1).words().lastOrNull() ?: ""
            if (!(src.isNotEmpty() && src[0].isUpperCase() && other.isNotEmpty() && other[0].isLowerCase())) return false
        }
        if (b < t.length && t[b] == '-' && b + 1 < t.length && t[b + 1].isLetter()) {
            val other = t.substring(b + 1).words().firstOrNull() ?: ""
            if (!(src.isNotEmpty() && src[0].isUpperCase() && other.isNotEmpty() && other[0].isLowerCase())) return false
        }
        return true
    }

    operator fun invoke(input: String): String {
        if (lex.isEmpty()) return input
        val t = input.replace("’", "'")
        val ws = WORD.findAll(t).toList()
        val out = StringBuilder()
        var pos = 0
        var i = 0
        while (i < ws.size) {
            var matched = false
            for (n in minOf(maxn, ws.size - i) downTo 1) {
                val span = ws.subList(i, i + n)
                if (span.zipWithNext().any { (a, b) -> t.substring(a.range.last + 1, b.range.first) !in listOf(" ", "-") }) continue
                val a = span.first().range.first
                val b = span.last().range.last + 1
                val src = t.substring(a, b)
                if (!hyphenOk(t, a, b, src)) continue
                val hit = lookup(src) ?: continue
                val (rep0, caps) = hit.first
                val tail = hit.second
                if (caps && !span.all { it.value.isNotEmpty() && it.value[0].isUpperCase() }) continue
                var rep = rep0
                if (src.isNotEmpty() && src[0].isUpperCase() && rep.isNotEmpty() && rep[0].isLowerCase())
                    rep = rep.replaceFirstChar { it.uppercaseChar() }
                out.append(t.substring(pos, a)).append(rep).append(tail)
                pos = b
                i += n
                matched = true
                break
            }
            if (!matched) i++
        }
        out.append(t.substring(pos))
        return out.toString()
    }

    companion object {
        private val WORD = rx("""[^\W\d_]+(?:'[^\W\d_]+)*""")
        fun fromFile(path: String): Respeller = Respeller().apply { val f = File(path); if (f.exists()) load(f.readText()) }
    }
}