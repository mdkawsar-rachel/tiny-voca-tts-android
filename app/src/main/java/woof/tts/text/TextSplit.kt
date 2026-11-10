package woof.tts.text

import kotlin.math.abs

object TextSplit {
    const val CLAUSE_MAX = 120
    const val FIRST_MIN = 20
    const val MIN_PIECE = 30
    private const val CONJ = """\b(?:and|but|or|nor|that|which|who|whom|whose|when|where|while|because|although|though|so|if|as|than|until)\b"""

    fun splitText(text: String, clauseMax: Int = CLAUSE_MAX, firstMax: Int = 0): List<String> {
        val out = splitInner(text, clauseMax).toMutableList()
        if (out.isNotEmpty() && firstMax > 0) {
            val f = splitFirst(out[0], firstMax)
            out[0] = f[0]
            out.addAll(1, f.drop(1))
        }
        return out
    }

    private fun splitInner(text: String, clauseMax: Int): List<String> {
        val out = ArrayList<String>()
        for (s in rx("""(?<=[.!?…])\s+""").split(text.trim())) {
            if (s.isEmpty()) continue
            if (s.length <= clauseMax) { out.add(s); continue }
            val clauses = rx("""(?<=[,;:])\s+|(?<=—)\s*|(?<=--)\s*""").split(s).filter { it.isNotEmpty() }
            val merged = ArrayList<String>()
            var cur = ""
            for (c in clauses) {
                if (cur.isNotEmpty() && cur.length + 1 + c.length > clauseMax) { merged.add(cur); cur = c }
                else cur = if (cur.isEmpty()) c else (if (cur.endsWith("—") || cur.endsWith("--")) cur + c else cur + " " + c)
            }
            if (cur.isNotEmpty()) merged.add(cur)
            for (m in merged) out.addAll(splitConj(m, clauseMax))
        }
        return out
    }

    private fun splitConj(s: String, clauseMax: Int): List<String> {
        if (s.length <= clauseMax) return listOf(s)
        val cands = rx("""\s(?=$CONJ)""").findAll(s).map { it.range.first }
            .filter { it >= MIN_PIECE && s.length - it >= MIN_PIECE }.toList()
        if (cands.isEmpty()) return listOf(s)
        val c = cands.minByOrNull { abs(it - s.length / 2.0) }!!
        return splitConj(s.substring(0, c).trim(), clauseMax) + splitConj(s.substring(c).trim(), clauseMax)
    }

    private fun splitFirst(s: String, firstMax: Int): List<String> {
        if (s.length <= firstMax) return listOf(s)
        var cuts = rx("""[,;:](?=\s)|—|--""").findAll(s).map { it.range.last + 1 }.toList()
        val q = s.withIndex().filter { it.value in "\"“”" }.map { it.index }
        cuts = cuts.filter { c -> c >= FIRST_MIN && s.length - c >= FIRST_MIN && q.none { abs(it - c) < 12 } }
        if (cuts.isEmpty()) return listOf(s)
        val inside = cuts.filter { it <= firstMax }
        val c = if (inside.isNotEmpty()) inside.max() else cuts.min()
        return listOf(s.substring(0, c).trim(), s.substring(c).trim())
    }
}