package woof.tts.text

fun cardinal(n: Long): String = when {
    n < 20 -> ONES[n.toInt()]
    n < 100 -> TENS[(n / 10).toInt()] + (if (n % 10 == 0L) "" else " " + ONES[(n % 10).toInt()])
    n < 1000 -> ONES[(n / 100).toInt()] + " hundred" + (if (n % 100 == 0L) "" else " and " + cardinal(n % 100))
    else -> {
        val sc = SCALES.firstOrNull { n >= it.first }
        if (sc == null) n.toString() else {
            val rest = n % sc.first
            val tail = if (rest == 0L) "" else (if (rest < 100) " and " else " ") + cardinal(rest)
            cardinal(n / sc.first) + " " + sc.second + tail
        }
    }
}

fun ordinal(n: Long): String {
    val w = cardinal(n).split(" ").toMutableList()
    val last = w.last()
    w[w.lastIndex] = ORD[last] ?: if (last.endsWith("y")) last.dropLast(1) + "ieth" else last + "th"
    return w.joinToString(" ")
}

fun year(n: Int): String {
    if (n in 2000..2009) return cardinal(n.toLong())
    val hi = n / 100
    val lo = n % 100
    return cardinal(hi.toLong()) + when {
        lo == 0 -> " hundred"
        lo < 10 -> " oh " + ONES[lo]
        else -> " " + cardinal(lo.toLong())
    }
}

fun number(s: String): String {
    val x = s.replace(",", "")
    if ("." in x) {
        val a = x.substringBefore(".")
        val b = x.substringAfter(".")
        return (if (a.isNotEmpty()) cardinal(a.toLong()) else "zero") + " point " + b.map { ONES[it.digitToInt()] }.joinToString(" ")
    }
    return cardinal(x.toLong())
}

fun digits(s: String): String = s.filter { it.isDigit() }.map { DIG[it.digitToInt()] }.joinToString(" ")

fun spell(s: String): String {
    val out = ArrayList<String>()
    for (c in s) when {
        c.isDigit() -> out.add(DIG[c.digitToInt()])
        c.isLetter() -> out.add(c.uppercaseChar().toString())
    }
    return out.joinToString(" ")
}

fun pronounceable(w: String): Boolean {
    val s = w.lowercase()
    if (s.length < 4 || !rx("[aeiouy]").containsMatchIn(s)) return false
    if (rx("[^aeiouy]{3,}").containsMatchIn(s) && !rx("(str|sch|thr|chr|spr|scr|ght|nds|nts)").containsMatchIn(s)) return false
    return rx("""(?:[^aeiouy]{0,3}[aeiouy]{1,2})+[^aeiouy]{0,3}""").matchEntire(s) != null
}

fun codeRead(s: String): String {
    val out = ArrayList<String>()
    for (p in s.split('-', '/')) {
        if (p.isEmpty()) continue
        out.add(when {
            p.pyIsDigit() -> if (p.length <= 2 && p[0] != '0') p else digits(p)
            p.pyIsAlpha() && p.pyIsLower() && p.length > 3 && pronounceable(p) -> p
            p.pyIsAlpha() && pronounceable(p) -> p.pyCap()
            else -> spell(p)
        })
    }
    return out.joinToString(" ")
}

fun nread(s: String): String {
    val n = s.toLong()
    if ((s[0] == '0' && s.length > 1) || s.length > 4) return digits(s)
    if (n < 100 || (n % 100 == 0L && s.length == 3) || n % 1000 == 0L) return cardinal(n)
    val hi = n / 100
    val lo = (n % 100).toInt()
    return cardinal(hi) + when {
        lo == 0 -> " hundred"
        lo < 10 -> " oh " + ONES[lo]
        else -> " " + cardinal(lo.toLong())
    }
}

fun splitIdent(s: String): String {
    val x = rx("""(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])""").replace(s.replace("_", " "), " ")
    return x.words().joinToString(" ") { if (it.pyIsUpper() && it.length <= 4) spell(it) else it }
}

fun roman(s: String): Int {
    val v = mapOf('I' to 1, 'V' to 5, 'X' to 10, 'L' to 50, 'C' to 100, 'D' to 500, 'M' to 1000)
    var sum = 0
    for (i in s.indices) {
        val a = v.getValue(s[i])
        val b = if (i + 1 < s.length) v.getValue(s[i + 1]) else 0
        sum += if (b > a) -a else a
    }
    return sum
}

fun romanSub(src: String, m: MatchResult): String {
    val prev = m.g(1)!!
    val num = m.g(3)
    if (num.isNullOrEmpty()) return m.value
    val before = src.substring(0, m.range.first).words()
    val prev2 = if (before.isNotEmpty() && !rx("[.,;:!?)\"]$").containsMatchIn(before.last())) before.last() else ""
    val key = prev.lowercase().trimEnd('.')
    val n = roman(num).toLong()
    val title = { w: String -> rx("[A-Z][a-z]+").matchEntire(w) != null }
    if (key in setOf("through", "thru", "to", "and", "or") &&
        !rx("""\b(?:""" + ROMAN_CARD.joinToString("|") + """)\s+\S+\s+$""", true).containsMatchIn(src.substring(0, m.range.first)))
        return m.value
    if (key in ROMAN_CARD && (num.length > 1 || (num == "I" && prev.isNotEmpty() && prev[0].isUpperCase()) ||
            ((num == "V" || num == "X") && key !in ROMAN_LABEL)))
        return prev + m.g(2) + cardinal(n)
    if (rx("[IVXL]+").matchEntire(num) == null) return m.value
    if (prev in REGNAL || (prev2 in TITLES && title(prev) && num != "I"))
        return prev + m.g(2) + "the " + ordinal(n)
    if (num.length > 1 && title(prev) && rx("[IVXL]*L[IVXL]*").matchEntire(num) == null) {
        val full = title(prev2) && prev2.lowercase() !in ROMAN_STOP
        return prev + m.g(2) + (if (full) "the " + ordinal(n) else cardinal(n))
    }
    return m.value
}

fun fracWords(a: Int, b: Int, lead: Boolean = false): String {
    val name = if (b == 2) (if (a == 1) "half" else "halves")
    else (if (b == 4) "quarter" else ordinal(b.toLong())) + (if (a == 1) "" else "s")
    return (if (lead && a == 1) "a" else cardinal(a.toLong())) + " " + name
}

fun sing(w: String): String {
    val head = w.substringBefore(" per ")
    val sep = if (" per " in w) " per " else ""
    val rest = w.substringAfter(" per ", "")
    if (" of " in head) return w
    val words = head.split(" ").toMutableList()
    val last = words.last()
    words[words.lastIndex] = SING[last] ?: if (last.endsWith("s")) last.dropLast(1) else last
    return words.joinToString(" ") + sep + rest
}

fun unit(k: String, n: String): String = if (n == "1") sing(UNITS.getValue(k)) else UNITS.getValue(k)

fun handle(h: String): String {
    val out = ArrayList<String>()
    for (p in h.split("_")) {
        val m = rx("([a-z]{3,}[nlyex])(ai|hq)").matchEntire(p)
        out.add(if (m != null) m.g(1) + " " + spell(m.g(2)!!) else splitIdent(p))
    }
    return out.filter { it.isNotEmpty() }.joinToString(" underscore ")
}