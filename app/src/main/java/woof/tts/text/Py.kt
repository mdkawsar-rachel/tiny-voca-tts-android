package woof.tts.text

import java.util.regex.Pattern

// Python's re is Unicode-aware by default; Java's predefined classes are ASCII unless this flag is set.
private const val UNI = Pattern.UNICODE_CHARACTER_CLASS

fun rx(p: String, ignoreCase: Boolean = false): Regex =
    Pattern.compile(p, UNI or if (ignoreCase) Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE else 0).toRegex()

fun MatchResult.g(n: Int): String? = groups[n]?.value

fun matchStart(p: String, s: String, ignoreCase: Boolean = false): Boolean = rx(p, ignoreCase).matchAt(s, 0) != null

fun String.pyCap(): String = lowercase().replaceFirstChar { it.uppercaseChar() }
fun String.pyIsUpper(): Boolean = any { it.isUpperCase() } && none { it.isLowerCase() }
fun String.pyIsLower(): Boolean = any { it.isLowerCase() } && none { it.isUpperCase() }
fun String.pyIsAlpha(): Boolean = isNotEmpty() && all { it.isLetter() }
fun String.pyIsDigit(): Boolean = isNotEmpty() && all { it.isDigit() }
fun String.words(): List<String> = trim().split(rx("""\s+""")).filter { it.isNotEmpty() }