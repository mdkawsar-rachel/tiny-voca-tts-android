package woof.tts.text

import java.util.regex.Pattern

private fun lit(k: String) = Pattern.quote(k)

fun normTokens(input: String): String {
    var t = input.replace("−", "-").replace("’", "'").replace("‘", "'").replace("“", "\"").replace("”", "\"")
    t = rx("""(?<=\d)([½¼¾⅓⅔⅛])""").replace(t) { " " + it.g(1) }
    val fracMap = mapOf("½" to "1/2", "¼" to "1/4", "¾" to "3/4", "⅓" to "1/3", "⅔" to "2/3", "⅛" to "1/8")
    t = rx("[½¼¾⅓⅔⅛]").replace(t) { fracMap.getValue(it.value) }
    val sup = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
    val norm = "0123456789-"
    t = rx("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+").replace(t) { "^" + it.value.map { c -> norm[sup.indexOf(c)] }.joinToString("") }
    t = rx("""(?<![\w.])([A-Za-z]*)\.NET\b""").replace(t) { (if (it.g(1)!!.isNotEmpty()) spell(it.g(1)!!) + " " else "") + "dot net" }
    t = rx("""\b([A-Za-z])#(?!\w)""").replace(t) { it.g(1) + " sharp" }
    t = rx("""\b([A-Za-z])\+\+(?!\w)""").replace(t) { it.g(1) + " plus plus" }
    t = rx("""\b([A-Za-z_][A-Za-z0-9_]*)\(\)""").replace(t) { splitIdent(it.g(1)!!) }
    t = rx("""\b[a-z]{2,}(?:[A-Z][a-z]+){2,}\b""").replace(t) { splitIdent(it.value) }
    t = rx("""\bv(\d+(?:\.\d+)+)\b""").replace(t) { "version " + it.g(1)!!.split(".").joinToString(" point ") }
    t = rx("""(?<![\w&#])#([A-Za-z][A-Za-z_]*)\b(?![\d-])""").replace(t) { "hashtag " + splitIdent(it.g(1)!!) }
    t = rx("""(?<![\w.@])@([A-Za-z_][A-Za-z0-9_]{1,30})\b""").replace(t) { "at " + handle(it.g(1)!!) }
    t = rx("""\b([A-Z]{1,2}\d[A-Z\d]?|[A-Z]\d[A-Z])\s(\d[A-Z]{2}|\d[A-Z]\d)\b""").replace(t) { spell(it.g(1)!!) + ", " + spell(it.g(2)!!) }
    t = rx("""\b((?:ending|ends) in|last (?:four|4)(?: digits)?(?: of)?|PIN|[Pp]in|passcode|OTP|[Cc]ode)(\s+)(\d{3,8}(?:\s\d{2,4})*)\b(?![.,]\d|-\w)""")
        .replace(t) { it.g(1)!! + it.g(2)!! + it.g(3)!!.words().joinToString(", ") { g -> digits(g) } }
    t = rx("""\b([A-Z]):\\""").replace(t) { it.g(1) + " colon backslash " }
    t = t.replace("\\", " backslash ")
    t = rx("""(?<=\s)\*(?=\s)(?!\s*\d)""").replace(t, "star")
    t = rx("""(?<![\w.])\.(env|gitignore|bashrc|zshrc|npmrc|htaccess|dockerignore)\b""").replace(t) { "dot " + it.g(1) }
    t = rx("""\b(?=\w*_)(?![\w]*\.\w)[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b""").replace(t) {
        it.value.split("_").joinToString(" underscore ") { p -> if (rx("""\d""").containsMatchIn(p) && rx("[A-Za-z]").containsMatchIn(p)) spell(p) else p }
    }
    t = rx("""(?<![\w+])1[-. ](\d{3})[-. ](?:(\d{3})[-. ](\d{4})|([A-Z]{3,}(?:-[A-Z]+)*))\b""").replace(t) { m ->
        "one " + (if (m.g(1) in listOf("800", "900")) cardinal(m.g(1)!!.toLong()) else digits(m.g(1)!!)) + ", " +
            (if (m.g(2) != null) digits(m.g(2)!!) + ", " + digits(m.g(3)!!) else m.g(4)!!.split("-").joinToString(" ") { it.pyCap() })
    }
    val NUM = """(\d[\d,]*(?:\.\d+)?)"""
    val SC = """(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)?"""
    val codes = CURCODE.keys.sortedByDescending { it.length }.joinToString("|") { lit(it) }
    fun cur(c: String, num: String, sc: String?): String {
        val name = CURCODE.getValue(c)
        if (name.length == 1) return name + num + (sc ?: "")
        val whole = num.substringBefore(".")
        val cents = if ("." in num) num.substringAfter(".") else ""
        if (sc == null && cents.length == 2 && name.endsWith("dollars"))
            return "$whole $name" + (if (cents.toInt() != 0) " and ${cents.toInt()} cents" else "")
        return num + (if (sc != null) " " + SCALE[sc.trim()] else "") + " " + name
    }
    t = rx("""(?<![\w$])($codes)\s?$NUM$SC""").replace(t) { cur(it.g(1)!!, it.g(2)!!, it.g(3)) }
    t = rx("""$NUM$SC\s?(USD|EUR|GBP|JPY|CNY|INR|CAD|AUD|CHF|NZD|HKD|SGD|KRW|MXN|BRL|RUB|SEK|NOK|DKK|ZAR|BTC|ETH)\b""")
        .replace(t) { cur(it.g(3)!!, it.g(1)!!, it.g(2)) }
    t = rx("""(?<!\w)(\d+)\s?¢""").replace(t) { it.g(1) + " cents" }
    val cw = mapOf("$" to "dollars", "€" to "euros", "£" to "pounds")
    t = rx("""([$€£])$NUM$SC\s?[-–]\s?[$€£]?$NUM$SC""").replace(t) { m ->
        val c = m.g(1)!!; val a = m.g(2)!!; val s1 = m.g(3); val b = m.g(4)!!; val s2 = m.g(5)
        if (s1 == null && s2 == null && rx("""\.\d\d\b""").containsMatchIn("$a $b")) "$c$a to $c$b"
        else {
            val x1 = if (s1 != null) " " + SCALE[s1.trim()] else ""
            val x2 = if (s2 != null) " " + SCALE[s2.trim()] else ""
            "$a${if (x1 != x2) x1 else ""} to $b$x2 ${cw[c]}"
        }
    }
    t = rx("""([$€£])$NUM(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)""")
        .replace(t) { "${it.g(2)} ${SCALE[it.g(3)!!.trim()]} ${cw[it.g(1)]}" }
    fun pl(n: String, w: String) = "$n $w${if (n == "1") "" else "s"}"
    t = rx("""\b(\d+)\s?h(?:rs?)?\s?(\d{1,2})(?:\s?m(?:ins?)?)?\b""").replace(t) { pl(it.g(1)!!, "hour") + " " + pl(it.g(2)!!, "minute") }
    t = rx("""\b(\d+)\s?m(?:ins?)?\s?(\d{1,2})\s?s(?:ecs?)?\b""").replace(t) { pl(it.g(1)!!, "minute") + " " + pl(it.g(2)!!, "second") }
    t = rx("""(\b(?:for|last|lasts|lasted|lasting|takes?|took|taking|every|within|wait|waited|remaining|runtime|ETA|blocked)\s+(?:\w+\s+){0,2}?)(\d{1,3})m\b(?!\s*(?:dash|sprint|race|swim|hurdles|pool|freestyle|relay))""")
        .replace(t) { it.g(1)!! + pl(it.g(2)!!, "minute") }
    t = rx("""(?<![\w$€£.,'])(')?(\d{2}|1\d{3}|20\d{2})'?s\b""").replace(t) { m ->
        val s = m.g(2)!!; val n = s.toInt()
        if (n % 10 != 0) m.value
        else {
            val w = if (s.length == 4) year(n) else if (n == 0 && m.g(1) != null) "two thousand" else if (n != 0) cardinal(n.toLong()) else ""
            if (w.isEmpty()) m.value else (if (w.endsWith("y")) w.dropLast(1) + "ies" else w + "s")
        }
    }
    t = rx("""(?:(?<=\s)|^)'(\d{2})\b(?!['\w])""").replace(t) { if (it.g(1)!![0] == '0') "oh " + ONES[it.g(1)!!.toInt()] else it.g(1)!! }
    t = rx("""\bWW\s?(II|I|2|1)\b""").replace(t) { "World War " + if (it.g(1) in listOf("II", "2")) "Two" else "One" }
    run { val src = t; t = rx("""\b([A-Za-z][A-Za-z']*\.?)(\s+)($ROMAN_RE)\b(?!'\w|-\d)""").replace(src) { romanSub(src, it) } }
    t = rx("""(?<=[a-z])\((I|II|III|IV|V|VI)\)""").replace(t) { " " + cardinal(roman(it.g(1)!!).toLong()) }
    t = rx("""(\d)\s?[x×*]\s?(?=10\s?\^)""").replace(t) { it.g(1) + " times " }
    t = rx("""$UNIT_PRE\s?(km|m|cm|mm|ft|mi|yd|in)\^([23])\b""")
        .replace(t) { "${it.g(1)} ${if (it.g(3) == "2") "square" else "cubic"} ${UNITS[it.g(2)]}" }
    t = rx("""(\b[A-Za-z]{1,3}|\b\d+(?:\.\d+)?|\))\s?\^\s?\(?([+-]?(?:\d+|[a-z]))\)?""").replace(t) { m ->
        var b = m.g(1)!!; var e = m.g(2)!!
        val neg = e.startsWith("-"); e = e.trimStart('+', '-')
        if (b.pyIsAlpha() && b.length > 1) b = b.toList().joinToString(" ")
        if (!e.pyIsDigit()) "$b to the $e"
        else {
            val n = e.toInt()
            when {
                n == 2 && !neg -> "$b squared"
                n == 3 && !neg -> "$b cubed"
                else -> "$b to the ${if (neg) "minus " else ""}${ordinal(n.toLong())}"
            }
        }
    }
    if (rx("""[=^+]|\bsolve\b|\bequation""").containsMatchIn(t)) {
        t = rx("""(?<=[\w)])\s+[-–]\s+(?=[\w(])""").replace(t, " minus ")
        t = rx("""\b(\d+)([a-fi-jn-rt-z])\b""").replace(t) { it.g(1) + " " + it.g(2) }
    }
    t = rx("""\b(\d+(?:\.\d+)?)x(?=\s+(?:faster|slower|more|less|larger|smaller|bigger|higher|lower|cheaper|better|worse|longer|shorter|the|as|over|speed|zoom)\b)""")
        .replace(t) { it.g(1) + " times" }
    t = rx("""(\d)\s+[x×*]\s+(\d+(?:\.\d+)?)(?=\s*=)""").replace(t) { it.g(1) + " times " + it.g(2) }
    t = rx("""(?<![\w$€£.])(\d+\.\d+|\d+(?=\s?\w+\s+[a-z]{3,}s\b))\s?(M|bn|mn|tn)\b(?=\s+[a-z]{3,})""")
        .replace(t) { "${it.g(1)} ${SCALE[it.g(2)]}" }
    return t
}

fun normPre(input: String): String {
    var t = normTokens(input)
    t = rx("""\b(localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}):(\d{2,5})\b""").replace(t) { "${it.g(1)} colon ${nread(it.g(2)!!)}" }
    fun say(s: String): String = SCHEME_SAY[s.lowercase()] ?: if (pronounceable(s)) s.lowercase() else spell(s)
    t = rx("""(?<![\w.+-])([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\s,.;:!?)\"']|$)""").replace(t) { say(it.g(1)!!) + " colon slash slash" }
    t = rx("""(?<![\w.+-])(?!https?://)([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\w/@\[])((?![\w.-]*@)[A-Za-z][\w-]*(?:\.[\w-]+)+\b)?""")
        .replace(t) { say(it.g(1)!!) + " colon slash slash " + (it.g(2) ?: "").replace(".", " dot ").replace("-", " dash ") }
    t = rx("""\bmailto:(?=[\w.+-]+@)""").replace(t, "mail to ")
    t = rx("""\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b""").replace(t) {
        it.g(1)!!.replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ") + " at " + it.g(2)!!.replace(".", " dot ").replace("-", " dash ")
    }
    t = rx("""\bhttps?://(?![\w.-]*[:@])([A-Za-z][\w-]*(?:\.[\w-]+)+\b)?""").replace(t) { m ->
        val g = m.g(1)
        if (g != null && !rx("""\.(?:com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz)$""").containsMatchIn(g))
            rx("^www ").replace(g.replace(".", " dot ").replace("-", " dash "), "w w w ")
        else g ?: ""
    }
    t = rx("""\bwww\.""").replace(t, "w w w dot ")
    t = rx("""\b((?:[\w-]+\.)+)(com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz)\b""")
        .replace(t) { it.g(1)!!.replace(".", " dot ").replace("-", " dash ") + it.g(2) }
    t = rx("""\b([\w-]+)\.(pdf|png|jpe?g|gif|docx?|xlsx?|csv|txt|zip|mp3|mp4|wav|json|py|js|html?|gradle|ipynb|rtf|pptx?|webm|bin|dat|bat|ps1|vue|jsx|kts|cs|dart|lua|r)\b""")
        .replace(t) { it.g(1)!!.replace("_", " underscore ") + " dot " + it.g(2) }
    t = rx("""(?<![\w@])((?:/?[\w-]+/)*[\w-]*[a-z_][\w-]*)\.(conf|log|cfg|ini|yaml|yml|toml|md|sh|ts|tsx|css|xml|sql|db|bak|tmp|exe|dmg|apk|iso|tar|gz|rar|7z|svg|webp|heic|mov|avi|mkv|flac|ogg|m4a|key|pem|crt|env|lock|rs|go|java|kt|swift|rb|php|c|cpp|h|git)\b""")
        .replace(t) { m ->
            var base = m.g(1)!!.replace("_", " underscore ").replace("/", " slash ").replace("-", " dash ")
            base = base.replace(".", " dot ")
            base + " dot " + m.g(2)
        }
    t = rx("""(?<![\w])/((?:[\w.-]+/)*[\w.-]+)""").replace(t) { " slash " + it.g(1)!!.replace("/", " slash ") }
    t = rx("""\b(\d{1,2})-(\d{1,2})-((?:19|20)\d\d)\b""").replace(t) { "${it.g(1)}/${it.g(2)}/${it.g(3)}" }
    t = rx("""\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d\d)\b""").replace(t) { m ->
        val a = m.g(1)!!.toInt(); val b = m.g(2)!!.toInt()
        if (a in 1..31 && b in 1..12) "the ${ordinal(a.toLong())} of ${MONTHS[b - 1]} ${m.g(3)}" else m.value
    }
    val daysAlt = DAYS.keys.joinToString("|")
    t = rx("""\b($daysAlt)\.?\s*[-–]\s*($daysAlt)\.?""").replace(t) { DAYS[it.g(1)] + " to " + DAYS[it.g(2)] }
    run { val src = t; t = rx("""(?<!\bthe )(?<!\bThe )\b($daysAlt)\.(?=[\s,]|$)""").replace(src) { m ->
        DAYS[m.g(1)]!! + (if (matchStart("""\s*(?:$|[A-Z])""", src.substring(m.range.last + 1))) "." else "")
    } }
    t = rx("""\b(on|by|until|till|every|from|next|this|last) ($daysAlt)\b(?!')""").replace(t) { it.g(1) + " " + DAYS[it.g(2)] }
    val monAlt = MON_ABBR.keys.joinToString("|")
    t = rx("""\b($monAlt)\.(?=\s*\d)""").replace(t) { MON_ABBR[it.g(1)]!! }
    t = rx("""\b($monAlt)\b(?=\s+\d{1,4}\b(?!:))""").replace(t) { MON_ABBR[it.g(1)]!! }
    t = rx("""(?<=\d )($monAlt)\b\.?""").replace(t) { MON_ABBR[it.g(1)]!! }
    run { val src = t; t = rx("""\b(\d{1,4})([A-Za-z]?)\s+((?:(?:North|South|East|West|N\.?|S\.?|E\.?|W\.?)\s+)?(?:[A-Z][\w']*|\d+(?:st|nd|rd|th))(?:\s+[A-Z][\w']*){0,2})\s+$STREET_RE""")
        .replace(src) { m ->
            val num = m.g(1)!!; val letter = m.g(2)!!; val name = m.g(3)!!; val kind = m.g(4)!!; val dot = m.g(5)!!
            val tail = if (dot.isNotEmpty() && matchStart("""\s*(?:$|[A-Z\n])""", src.substring(m.range.last + 1))) "." else ""
            nread(num) + (if (letter.isNotEmpty()) " " + letter.uppercase() else "") + " $name ${STREET[kind] ?: kind}$tail"
        } }
    run { val src = t; t = rx("""(?<!Mount )(?<!Port )(?<!Fort )(?<!Lake )(?<!Cape )\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))\s+(St|Rd|Ave|Blvd|Ln|Hwy|Pkwy|Cres|Terr?)\b(\.?)""")
        .replace(src) { m ->
            val stop = ROMAN_STOP + setOf("on", "at", "visit", "see")
            if (m.g(1)!!.lowercase() in stop || (m.g(2) == "St" && matchStart("""\s+[A-Z]""", src.substring(m.range.last + 1)))) m.value
            else m.g(1) + " " + STREET[m.g(2)] + (if (m.g(3)!!.isNotEmpty() && matchStart("""\s*(?:$|\n)""", src.substring(m.range.last + 1))) "." else "")
        } }
    t = rx("""\b((?:I|US|SR|Route|Highway|Hwy|Interstate)[- ]\d+)\s([NSEW])\b""")
        .replace(t) { it.g(1) + " " + mapOf("N" to "North", "S" to "South", "E" to "East", "W" to "West")[it.g(2)] }
    for ((pat, rep) in ABBR) t = rx(pat).replace(t, rep)
    t = rx("""\b(\d{4})-(\d{2})-(\d{2})\b""").replace(t) { m ->
        val mo = m.g(2)!!.toInt()
        if (mo in 1..12) "${MONTHS[mo - 1]} ${m.g(3)!!.toInt()}, ${m.g(1)}" else m.value
    }
    t = rx("""\b(\d{1,2})(?:st|nd|rd|th)?\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?\s+$MON_RE\b""").replace(t) { m ->
        val a = m.g(1)!!.toInt(); val b = m.g(2)!!.toInt()
        if (a < b && b <= 31) "the ${ordinal(a.toLong())} to the ${ordinal(b.toLong())} of ${m.g(3)}" else m.value
    }
    run { val src = t; t = rx("""\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?$MON_RE\b""").replace(src) { m ->
        val d = m.g(1)!!.toInt()
        if (d in 1..31) (if (rx("""\bthe\s+$""", true).containsMatchIn(src.substring(0, m.range.first))) "" else "the ") + "${ordinal(d.toLong())} of ${m.g(2)}" else m.value
    } }
    t = rx("""\b(1[5-9]\d\d|20\d\d)\s*[-–]\s*(1[5-9]\d\d|20\d\d|\d\d)\b(?![-/.]\d)""").replace(t) { m ->
        val a = m.g(1)!!; val b = m.g(2)!!
        if (b.length == 4 || b.toInt() > a.substring(2).toInt()) "$a to $b" else m.value
    }
    t = rx("""(?<![\d/])\b(\d{1,2})/((?:19|20)\d\d)\b""").replace(t) { m ->
        val mo = m.g(1)!!.toInt()
        if (mo in 1..12) "${MONTHS[mo - 1]} ${m.g(2)}" else m.value
    }
    t = rx("""(?<![\d/])\b(24/7|50/50|9/11)\b(?!/)""").replace(t) { mapOf("24/7" to "24 7", "50/50" to "fifty fifty", "9/11" to "nine eleven")[it.g(1)]!! }
    t = rx("""(?<![\d/.,])\b(\d+)\s+(\d{1,2})/(\d{1,2})\b(?!/\d)""").replace(t) { m ->
        val a = m.g(2)!!.toInt(); val b = m.g(3)!!.toInt()
        if (a in 1 until b && b in FRAC_DEN) "${m.g(1)} and ${fracWords(a, b, true)}" else m.value
    }
    run { val src = t; t = rx("""(?<![\d/])\b(\d{1,2})/(\d{1,2})\b(?!/\d)""").replace(src) { m ->
        val a = m.g(1)!!.toInt(); val b = m.g(2)!!.toInt()
        val end = m.range.last + 1
        val after = src.substring(end, minOf(src.length, end + 12)).lowercase()
        val before = src.substring(maxOf(0, m.range.first - 16), m.range.first)
        if (matchStart("""\s*(stars?|points?|marks?|votes?)\b""", after) || rx("""(rated|scored?|rating|got)\s*$""").containsMatchIn(before.lowercase()))
            "$a out of $b"
        else if (a in 1 until b && b in FRAC_DEN && m.g(1)!![0] != '0' && m.g(2)!![0] != '0' &&
            (!rx(DATE_BEFORE, true).containsMatchIn(before) || matchStart(FRAC_AFTER, after)))
            fracWords(a, b)
        else {
            fun ordn(n: Int) = "$n" + (if (n % 100 in 10..20) "th" else when (n % 10) { 1 -> "st"; 2 -> "nd"; 3 -> "rd"; else -> "th" })
            val the = if (rx("""\bthe\s+$""", true).containsMatchIn(before)) "" else "the "
            when {
                b in 1..12 && a in 1..31 -> "$the${ordn(a)} of ${MONTHS[b - 1]}"
                a in 1..12 && b in 1..31 -> "${MONTHS[a - 1]} ${ordn(b)}"
                else -> m.value
            }
        }
    } }
    t = rx("""\b(\d+(?:\.\d+){2,})(?:-(alpha|beta|rc\d*|preview))?\b""")
        .replace(t) { it.g(1)!!.split(".").joinToString(" point ") + (if (it.g(2) != null) " " + it.g(2) else "") }
    fun pl2(n: String, w: String) = "${n.toInt()} $w${if (n.toInt() == 1) "" else "s"}"
    run { val src = t; t = rx("""\b(\d{1,2}):(\d{2}):(\d{2})\b(?!:)""").replace(src) { m ->
        if (rx("""\b(at|by|until|from)\s+$""").containsMatchIn(src.substring(0, m.range.first)))
            "${m.g(1)}:${m.g(2)} and ${pl2(m.g(3)!!, "second")}"
        else listOf(m.g(1)!! to "hour", m.g(2)!! to "minute", m.g(3)!! to "second")
            .filter { it.first.toInt() != 0 || it.second == "second" }
            .joinToString(", ") { pl2(it.first, it.second) }
    } }
    t = rx("""\b(\d{1,2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\s?[-–]\s?(\d{1,2}(?::\d{2})?\s?[AaPp]\.?[Mm]\b\.?)""").replace(t) { it.g(1) + " to " + it.g(2) }
    t = rx("""\b(\d{1,2}:\d{2})\s?[-–]\s?(\d{1,2}:\d{2})\b""").replace(t) { it.g(1) + " to " + it.g(2) }
    t = rx("""\b(\d{1,2}):(\d)\b(?!\d)""").replace(t) { it.g(1) + " to " + it.g(2) }
    t = rx("""(?<=[Rr]atio of |[Rr]atio is )(\d{1,2}):(\d{1,2})\b""").replace(t) { it.g(1) + " to " + it.g(2) }
    t = rx("""\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})*-\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b""").replace(t) { codeRead(it.value) }
    t = rx("""(?<![\w.,])\+?\(?\d{2,4}\)?(?:[\s.-]\(?\d{2,4}\)?){1,4}(?![\w.,]*\d)""").replace(t) { m ->
        val s = m.value
        if (s.count { it.isDigit() } >= 7) {
            val lead = if (s.trim().startsWith("+")) "plus " else ""
            " " + lead + rx("""\d+""").findAll(s).joinToString(", ") { digits(it.value) } + " "
        } else m.value
    }
    t = rx("""\b((?:[Cc]all|[Dd]ial|[Tt]ext|[Rr]ing)\s+)(\d{3,6})\b""").replace(t) { it.g(1) + digits(it.g(2)!!) }
    t = rx("""\b(\d+)\s?ft\.?\s?(\d{1,2})\s?in\b\.?""")
        .replace(t) { "${it.g(1)} ${if (it.g(1) == "1") "foot" else "feet"} ${it.g(2)} ${if (it.g(2) == "1") "inch" else "inches"}" }
    t = rx("""(\d)'\s?(\d{1,2})(?:"|'')""").replace(t) { it.g(1) + " foot " + it.g(2) }
    val ukeys = UNITS.keys.filter { it != "in" && it != "s" }.sortedByDescending { it.length }.joinToString("|") { lit(it) }
    t = rx("""$UNIT_PRE\s?($ukeys)/($ukeys|s|in)(?![\w/])""")
        .replace(t) { m -> if (m.g(3) == "in") m.value else "${m.g(1)} ${unit(m.g(2)!!, m.g(1)!!)} per ${sing(UNITS[m.g(3)] ?: "seconds")}" }
    t = rx("""$UNIT_PRE\s?($ukeys)(?![\w/'])""").replace(t) { "${it.g(1)} ${unit(it.g(2)!!, it.g(1)!!)}" }
    t = rx("""(\b(?:squared|cubed|to the (?:minus )?(?:\w+ )?\w+(?:th|st|nd|rd))\s)($ukeys)(?:/($ukeys|s))?(?![\w/'])""")
        .replace(t) { m -> m.g(1)!! + UNITS[m.g(2)]!! + (if (m.g(3) != null) " per " + sing(UNITS[m.g(3)] ?: "seconds") else "") }
    t = rx("""\b($ID_WORDS)(\s+(?:No\.?\s*|#\s*)?)([A-Z]{1,3}\s?)?(\d{2,4})([A-Z]{0,2})\b(?![.,]\d|\s?[-–]\s?\d)""").replace(t) { m ->
        m.g(1)!! + m.g(2)!! + (if (m.g(3) != null) spell(m.g(3)!!) + " " else "") + nread(m.g(4)!!) + (if (m.g(5)!!.isNotEmpty()) " " + spell(m.g(5)!!) else "")
    }
    t = rx("""\b([A-Z]{2,3})\s(\d{3,4})\b(?![.,:]\d|\s\d)""").replace(t) { it.g(1) + " " + nread(it.g(2)!!) }
    fun pair(src: String, start: Int): Boolean = !rx(CODE_CTX).containsMatchIn(src.substring(maxOf(0, start - 40), start))
    run { val src = t; t = rx("""(?<![-/.\w])([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})(?![-/\w])""").replace(src) { m ->
        if (pair(src, m.range.first) && (m.g(2)!!.length <= 2 || (m.g(1)!!.length == 2 && m.g(3)!!.isEmpty())))
            listOf(spell(m.g(1)!!), nread(m.g(2)!!), spell(m.g(3)!!)).filter { it.isNotEmpty() }.joinToString(" ")
        else m.value
    } }
    run { val src = t; t = rx("""(?<![-/.\w])(\d{1,4})([A-Z]{1,2})(?![-/\w])""").replace(src) { m ->
        if (pair(src, m.range.first)) nread(m.g(1)!!) + " " + spell(m.g(2)!!) else m.value
    } }
    run { val src = t; t = rx("""(?<![-/.\w])(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d{0,3}){2,}(?![-/\w])""").replace(src) { m ->
        val elems = rx("[A-Z][a-z]?").findAll(m.value).map { it.value }.toList()
        if (elems.all { it in ELEMENTS } && pair(src, m.range.first))
            rx("([A-Z][a-z]?)(\\d*)").findAll(m.value).joinToString(" ") {
                val e = it.g(1)!!; val n = it.g(2)!!
                spell(e) + (if (n.isNotEmpty()) " " + cardinal(n.toLong()) else "")
            }
        else m.value
    } }
    t = rx("""#(\d+(?:-\d+)*)\b""").replace(t) { "number " + it.g(1)!!.split("-").joinToString(", ") { g -> if (g.length <= 3) g else digits(g) } }
    t = rx("""\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b""").replace(t) { codeRead(it.value) }
    t = rx("""\b(?=\w*\d)(?=\w*[A-Z])[A-Z0-9]{3,}\b""").replace(t) { spell(it.value) }
    t = rx("""\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b""").replace(t) { m ->
        if (rx("""\d+(?:st|nd|rd|th|am|pm|s|kg|km|mm|cm|ml|mg|gb|mb|tb|kb|ghz|mhz|kw|kwh|mah|hrs?|mins?|h|m|k|x|g|l|w|v)""", true).matchEntire(m.value) != null) m.value else spell(m.value)
    }
    t = rx("""(?<![\d$€£.,])\b(\d{5,})\b(?![.,]\d)""").replace(t) { digits(it.g(1)!!) }
    t = rx("""(?<![\d/:.-])\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b(?![/:.-]\d)""").replace(t) { it.g(1) + " to " + it.g(2) }
    t = rx("""\b(\d+)\s*h\s*(\d{1,2})\b""").replace(t) { "${it.g(1)} hour${if (it.g(1) == "1") "" else "s"} ${it.g(2)}" }
    t = rx("""\b(\d+)\s*hrs?\b""").replace(t) { "${it.g(1)} hour${if (it.g(1) == "1") "" else "s"}" }
    t = rx("""\b(\d+)\s*mins?\b""").replace(t) { "${it.g(1)} minute${if (it.g(1) == "1") "" else "s"}" }
    t = rx("""$UNIT_PRE\s?($ukeys)(?![\w/'])""").replace(t) { "${it.g(1)} ${unit(it.g(2)!!, it.g(1)!!)}" }
    t = rx("""\s*×\s*""").replace(t, " by ")
    t = rx("""(?<=\d)\s+x\s+(?=\d)""").replace(t, " by ")
    for ((pat, rep) in SYMBOLS) t = rx(pat).replace(t, rep)
    t = rx("""£(\d[\d,]*)\.(\d{2})\b""").replace(t) { if (it.g(2) != "00") "${it.g(1)} pounds ${it.g(2)!!.toInt()}" else "${it.g(1)} pounds" }
    t = rx("""\s+–\s+""").replace(t, " — ").replace("–", "-")
    return rx("""\s{2,}""").replace(t, " ").trim()
}

fun fallback(input: String): String {
    var t = input
    val cur = mapOf("$" to listOf("dollar", "dollars", "cent", "cents"), "€" to listOf("euro", "euros", "cent", "cents"), "£" to listOf("pound", "pounds", "penny", "pence"))
    t = rx("""([$€£])(\d[\d,]*)(?:\.(\d+))?\b""").replace(t) { m ->
        val (one, many, c1, cn) = cur[m.g(1)!!]!!
        val whole = m.g(2)!!.replace(",", "").toLong(); val frac = m.g(3)
        if (frac != null && frac.length != 2) "${number(m.g(2) + "." + frac)} $many"
        else {
            val c = if (frac != null) frac.toInt() else 0
            if (whole == 0L && c != 0) "${cardinal(c.toLong())} ${if (c == 1) c1 else cn}"
            else {
                val out = "${cardinal(whole)} ${if (whole == 1L) one else many}"
                out + (if (c != 0) " and ${cardinal(c.toLong())} ${if (c == 1) c1 else cn}" else "")
            }
        }
    }
    run { val src = t; t = rx("""\b(\d{1,2}):(\d{2})\s*([aApP]\.?[mM])?(?:\.(?=\s+[a-z]))?(?=\W|$)""").replace(src) { m ->
        val h = m.g(1)!!.toInt(); val mi = m.g(2)!!.toInt()
        val rest = src.substring(m.range.last + 1)
        if (mi == 0 && m.g(3) == null && matchStart("""\s*(noon|midnight)""", rest)) cardinal(h.toLong())
        else if (mi == 0 && m.g(3) == null && (h > 12 || h == 0)) (if (h == 0) "zero" else cardinal(h.toLong())) + " hundred"
        else {
            var out = cardinal(h.toLong()) + (if (mi == 0) (if (m.g(3) == null) " o'clock" else "") else if (mi < 10) " oh " + ONES[mi] else " " + cardinal(mi.toLong()))
            if (m.g(3) != null) out += " " + m.g(3)!!.replace(".", "").uppercase()
            out
        }
    } }
    t = rx("""\b(\d{1,2})\s*([aApP])\.?[mM](?:\.(?=\s+[a-z]))?(?=\W|$)""").replace(t) { "${cardinal(it.g(1)!!.toLong())} ${it.g(2)!!.uppercase()}M" }
    t = rx("""\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b""").replace(t) { m ->
        val mo = m.g(1)!!.toInt(); val d = m.g(2)!!.toInt(); var y = m.g(3)!!.toInt()
        if (m.g(3)!!.length == 2) y += 2000
        if (mo in 13..31 && d in 1..12) "the ${ordinal(mo.toLong())} of ${MONTHS[d - 1]} ${year(y)}"
        else if (!(mo in 1..12 && d in 1..31)) m.value
        else "${MONTHS[mo - 1]} ${ordinal(d.toLong())} ${year(y)}"
    }
    t = rx("""\b(${MONTHS.joinToString("|")})\.? (\d{1,2})\b(?!:)""").replace(t) { "${it.g(1)} ${ordinal(it.g(2)!!.toLong())}" }
    t = rx("""(\d)\s?°\s?C\b""").replace(t) { it.g(1) + " degrees Celsius" }
    t = rx("""(\d)\s?°\s?F\b""").replace(t) { it.g(1) + " degrees Fahrenheit" }
    t = rx("""(\d)\s?°""").replace(t) { it.g(1) + " degrees" }
    t = rx("""(?<=\w)/(?=\w)""").replace(t, " slash ")
    t = rx("""\b(\d+)(st|nd|rd|th)\b""").replace(t) { ordinal(it.g(1)!!.toLong()) }
    t = rx("""\b(1[1-9]\d\d|20\d\d)\b(?![.,]\d)""").replace(t) { year(it.g(1)!!.toInt()) }
    t = rx("""(?<![\w-])-(?=\d)""").replace(t, "minus ")
    t = rx("""(?<=[A-Za-z])(?=\d)""").replace(t, " ")
    t = rx("""(\d+(?:\.\d+)?)\s?%""").replace(t) { number(it.g(1)!!) + " percent" }
    t = rx("""\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b""").replace(t) { number(it.value) }
    return t.replace("&", " and ").replace("+", " plus ").replace("=", " equals ").replace("@", " at ").replace("#", " number ")
}

class Normalizer(private val respeller: Respeller? = Respeller()) {
    operator fun invoke(text: String): String {
        var t = normPre(text)
        if (rx("[0-9#%$€£@&/_+=<>±≥≤×°]").containsMatchIn(t)) t = fallback(t)
        t = rx("""\s+([.,!?;:])""").replace(t) { it.g(1)!! }
        t = rx("""\s{2,}""").replace(t, " ").trim()
        if (respeller != null) t = respeller(t)
        return if (text.isNotEmpty() && text[0].isUpperCase()) t.replaceFirstChar { it.uppercaseChar() } else t
    }
}