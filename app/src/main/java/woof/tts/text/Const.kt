package woof.tts.text

val DIG = "zero one two three four five six seven eight nine".split(" ")

val ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(" ")
val TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split(" ")
val SCALES = listOf(1_000_000_000_000L to "trillion", 1_000_000_000L to "billion", 1_000_000L to "million", 1000L to "thousand")
val MONTHS = "January February March April May June July August September October November December".split(" ")
val ORD = mapOf("one" to "first", "two" to "second", "three" to "third", "five" to "fifth", "eight" to "eighth", "nine" to "ninth", "twelve" to "twelfth")

const val ROMAN_RE = """(?=[MDCLXVI]+\b)M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"""
val ROMAN_CARD: Set<String> = ("chapter chap ch part pt book volume vol act scene section sec phase stage level episode season article " +
    "title class type grade war bowl apollo mark mk vatican rocky fantasy wars trek civilization gemini saturn soyuz annex " +
    "appendix figure fig table plate tier round schedule unit division category series olympiad psalm canto lesson module form")
    .split(" ").toMutableSet().apply {
        addAll(toList().map { it + "s" })
        add("through"); add("thru")
    }
val ROMAN_LABEL = "appendix annex table figure fig plate schedule unit category class type grade tier section form division series".split(" ").toSet()
val ROMAN_STOP = "the a an in on of and for to at by from with my our his her their this that its".split(" ").toSet()
val REGNAL = ("Henry Louis Elizabeth George Edward Richard Charles James William John Paul Benedict Pius Leo Gregory " +
    "Innocent Clement Boniface Urban Alexander Peter Ivan Nicholas Frederick Wilhelm Ludwig Philip Felipe Carlos Gustav " +
    "Gustaf Christian Olaf Haakon Harald Ramesses Ramses Thutmose Amenhotep Ptolemy Darius Xerxes Cyrus Constantine " +
    "Justinian Mehmed Suleiman Selim Murad Napoleon Umberto Mary Anne Victoria Catherine Isabella Margaret Rama Abdullah " +
    "Robert David Kenneth Duncan Alfonso Ferdinand Sancho Manuel Sebastian Stephen Matthias Sigismund Casimir Otto Conrad " +
    "Rudolf Maximilian Leopold Joseph Francis Franz Karl Hugh Baldwin Basil Michael Theodore Edmund Harold Magnus Eric " +
    "Erik Valdemar Frederik Albert Baudouin Philippe Rainier Hassan Faisal Fuad Pedro Afonso Jaime Sixtus Julius Adrian " +
    "Hadrian Celestine Honorius Lothair Pepin Antiochus Seleucus Herod Cleopatra Wenceslaus Kamehameha Tewodros Menelik " +
    "Nebuchadnezzar Shalmaneser Sargon Tiglath Artaxerxes Andronikos Romanos Leon Mohammed Muhammad Charlemagne").split(" ").toSet()
val TITLES = ("King Queen Pope Emperor Empress Tsar Czar Tsarina Kaiser Pharaoh Sultan Shah Prince Princess Duke Duchess " +
    "Count Countess Lord Lady Saint St Archduke Elector Emir Caliph Khan King's Queen's").split(" ").toSet()

val CURCODE = mapOf(
    "USD" to "$", "US$" to "$", "EUR" to "€", "GBP" to "£", "JPY" to "yen", "¥" to "yen", "CNY" to "yuan", "RMB" to "yuan",
    "INR" to "rupees", "₹" to "rupees", "Rs." to "rupees", "Rs" to "rupees", "CAD" to "Canadian dollars", "C$" to "Canadian dollars",
    "AUD" to "Australian dollars", "A$" to "Australian dollars", "NZD" to "New Zealand dollars", "CHF" to "Swiss francs",
    "HKD" to "Hong Kong dollars", "HK$" to "Hong Kong dollars", "SGD" to "Singapore dollars", "KRW" to "won", "₩" to "won",
    "MXN" to "pesos", "BRL" to "reais", "R$" to "reais", "RUB" to "rubles", "₽" to "rubles", "SEK" to "Swedish kronor",
    "NOK" to "Norwegian kroner", "DKK" to "Danish kroner", "ZAR" to "rand", "₦" to "naira", "NGN" to "naira", "₺" to "lira",
    "₱" to "pesos", "₫" to "dong", "BTC" to "bitcoin", "ETH" to "ether", "₿" to "bitcoin")
val SCALE = mapOf("k" to "thousand", "K" to "thousand", "m" to "million", "M" to "million", "mn" to "million", "mm" to "million",
    "bn" to "billion", "b" to "billion", "B" to "billion", "tn" to "trillion", "T" to "trillion", "trn" to "trillion",
    "thousand" to "thousand", "million" to "million", "billion" to "billion", "trillion" to "trillion")
val FRAC_DEN = setOf(2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100)
const val FRAC_AFTER = """\s*(?:of\b|an?\b|cups?|c\.|teaspoons?|tsp|tablespoons?|tbsp|inch|inches|in\b|in\.|"|miles?|mi\b|""" +
    """pounds?|lbs?|ounces?|oz|hours?|hrs?|minutes?|mins?|seconds?|pints?|quarts?|gallons?|liters?|litres?|""" +
    """kilos?|kg|feet|foot|ft|yards?|acres?|sticks?|pieces?|slices?|loaf|tons?|percent|points?|shares?|""" +
    """the\b|times\b|marathon|price|full|empty|done|finished|complete)"""
const val DATE_BEFORE = """(?:\b(?:on|by|until|till|from|since|before|after|due|dated?|deadline|through|thru|starting|effective|""" +
    """expires?|expiry|born|died|week of|shipped|delivered|arrives?|arriving|ordered|paid|placed|scheduled|""" +
    """filed|signed|posted|submitted|opens?|closes?|ends?|starts?|begins?|[Mm]onday|[Tt]uesday|[Ww]ednesday|[Tt]hursday|[Ff]riday|[Ss]aturday|""" +
    """[Ss]unday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?[,:]?\s*)$"""

val ABBR = listOf(
    """\bDr\.(?=\s+[A-Z])""" to "Doctor", """\bMr\.""" to "Mister", """\bMrs\.""" to "Missus", """\bMs\.""" to "Miz",
    """\bProf\.(?=\s+[A-Z])""" to "Professor", """\bSt\.(?=\s+[A-Z])""" to "Saint", """(?<=[a-z] )St\.""" to "Street",
    """\bRd\.""" to "Road", """\bAve\.""" to "Avenue", """\bBlvd\.""" to "Boulevard", """\bApt\.""" to "Apartment",
    """\b[Aa]pprox\.""" to "approximately", """\betc\.""" to "et cetera", """\be\.g\.""" to "for example", """\bi\.e\.""" to "that is",
    """\bvs\.?(?=\s)""" to "versus", """\b[Nn]o\.(?=\s*\d)""" to "number", """\bdept\.""" to "department", """\bext\.""" to "extension",
    """(?<=\d)\s?min\.""" to " minutes", """\bmin\.(?=\s+[a-z])""" to "minimum", """\bmin\.""" to "minutes",
    """\bhr?s?\.(?=\s)""" to "hours", """(?<![\d\s])\s*\b[Ss]ec\.(?=\s*\d)""" to " section", """\b([Ee])st\.(?=\s*\d{4})""" to "$1stablished",
    """\bSgt\.""" to "Sergeant", """\bCapt\.""" to "Captain", """\bLt\.""" to "Lieutenant", """\bCol\.""" to "Colonel",
    """\bGen\.(?=\s+[A-Z])""" to "General", """\bGov\.""" to "Governor", """\bSen\.""" to "Senator", """\bRep\.(?=\s+[A-Z])""" to "Representative",
    """\bRev\.""" to "Reverend", """\bHon\.""" to "Honorable", """\bJr\.""" to "Junior", """\bSr\.""" to "Senior",
    """\bLn\.""" to "Lane", """\bDr\.(?=[\s,.]|$)""" to "Drive", """\bCt\.""" to "Court", """\bPl\.""" to "Place", """\bSq\.""" to "Square",
    """\bHwy\.""" to "Highway", """\bPkwy\.""" to "Parkway", """\bMt\.""" to "Mount", """\bFt\.(?=\s+[A-Z])""" to "Fort",
    """\bSte\.""" to "Suite", """\bBldg\.""" to "Building", """\bfl\.""" to "floor", """\bFl\.""" to "Floor", """\bRm\.""" to "Room",
    """\bEst\.""" to "Estimated", """\best\.""" to "estimated", """\bInc\.""" to "Incorporated", """\bLtd\.""" to "Limited",
    """\bCo\.(?=\s)""" to "Company", """\bCorp\.""" to "Corporation", """\bDept\.""" to "Department", """\bAssn\.""" to "Association",
    """\bUniv\.""" to "University", """\bqty\.?""" to "quantity", """\bref\.""" to "reference", """\btel\.""" to "telephone",
    """\bTel\.""" to "Telephone", """\bfig\.""" to "figure", """\bvol\.""" to "volume", """\bpp\.""" to "pages", """\bp\.(?=\s*\d)""" to "page",
    """\bwk\.""" to "week", """\bmo\.""" to "month", """\byr\.""" to "year", """\bavg\.""" to "average", """\bmax\.""" to "maximum")
val DAYS = mapOf("Mon" to "Monday", "Tue" to "Tuesday", "Tues" to "Tuesday", "Wed" to "Wednesday", "Thu" to "Thursday", "Thur" to "Thursday",
    "Thurs" to "Thursday", "Fri" to "Friday", "Sat" to "Saturday", "Sun" to "Sunday")
val MON_ABBR = mapOf("Jan" to "January", "Feb" to "February", "Mar" to "March", "Apr" to "April", "Jun" to "June", "Jul" to "July",
    "Aug" to "August", "Sep" to "September", "Sept" to "September", "Oct" to "October", "Nov" to "November", "Dec" to "December")
val SYMBOLS = listOf(
    """\s*±\s*""" to " plus or minus ", """\s*≥\s*""" to " greater than or equal to ", """\s*≤\s*""" to " less than or equal to ",
    """\s*≠\s*""" to " not equal to ", """\s*≈\s*""" to " approximately ", """\s*÷\s*""" to " divided by ",
    """\s*→\s*""" to " to ", """\s*√\s*""" to " square root of ", """(?<=\d)\s*%""" to " percent", """\s*‰""" to " per mille",
    """\s+=\s+""" to " equals ", """(?<=\d)\s*<\s*(?=\d)""" to " less than ", """(?<=\d)\s*>\s*(?=\d)""" to " greater than ",
    """\s*&\s*""" to " and ")
val UNITS = mapOf("km" to "kilometers", "m" to "meters", "cm" to "centimeters", "mm" to "millimeters", "kg" to "kilograms", "g" to "grams",
    "mg" to "milligrams", "l" to "liters", "L" to "liters", "ml" to "milliliters", "mL" to "milliliters", "km/h" to "kilometers per hour",
    "mph" to "miles per hour", "GB" to "gigabytes", "MB" to "megabytes", "TB" to "terabytes", "kB" to "kilobytes", "GHz" to "gigahertz",
    "MHz" to "megahertz", "kW" to "kilowatts", "kWh" to "kilowatt hours", "W" to "watts", "V" to "volts", "mAh" to "milliamp hours",
    "ft" to "feet", "in" to "inches", "lb" to "pounds", "lbs" to "pounds", "oz" to "ounces", "h" to "hours", "min" to "minutes",
    "s" to "seconds", "ms" to "milliseconds", "mi" to "miles", "yd" to "yards", "yds" to "yards", "Hz" to "hertz", "kHz" to "kilohertz",
    "dB" to "decibels", "mpg" to "miles per gallon", "kph" to "kilometers per hour", "tsp" to "teaspoons", "tbsp" to "tablespoons",
    "gal" to "gallons", "qt" to "quarts", "mcg" to "micrograms", "µg" to "micrograms", "μg" to "micrograms", "nm" to "nanometers",
    "µm" to "micrometers", "μm" to "micrometers", "ns" to "nanoseconds", "µs" to "microseconds", "μs" to "microseconds",
    "sec" to "seconds", "secs" to "seconds", "Mbps" to "megabits per second", "Gbps" to "gigabits per second",
    "kbps" to "kilobits per second", "kcal" to "kilocalories", "mmHg" to "millimeters of mercury", "dL" to "deciliters",
    "dl" to "deciliters", "PB" to "petabytes", "MW" to "megawatts", "GW" to "gigawatts", "GWh" to "gigawatt hours",
    "MWh" to "megawatt hours", "kPa" to "kilopascals", "hPa" to "hectopascals", "mmol" to "millimoles", "fl oz" to "fluid ounces",
    "sq ft" to "square feet", "sq m" to "square meters", "km2" to "square kilometers", "m2" to "square meters", "cc" to "C C",
    "mA" to "milliamps", "ng" to "nanograms", "pg" to "picograms", "IU" to "I U", "mEq" to "milliequivalents", "kN" to "kilonewtons",
    "Nm" to "newton meters", "TFLOPS" to "teraflops", "rpm" to "R P M", "psi" to "P S I")
val SING = mapOf("feet" to "foot", "inches" to "inch", "hertz" to "hertz", "kilohertz" to "kilohertz", "teaspoons" to "teaspoon",
    "C C" to "C C", "R P M" to "R P M", "P S I" to "P S I", "teraflops" to "teraflop", "I U" to "I U")

const val MON_RE = """(January|February|March|April|May|June|July|August|September|October|November|December)"""
val STREET = mapOf("St" to "Street", "Rd" to "Road", "Ave" to "Avenue", "Av" to "Avenue", "Blvd" to "Boulevard", "Ln" to "Lane", "Dr" to "Drive",
    "Ct" to "Court", "Pl" to "Place", "Sq" to "Square", "Hwy" to "Highway", "Pkwy" to "Parkway", "Cres" to "Crescent",
    "Ter" to "Terrace", "Terr" to "Terrace", "Cir" to "Circle", "Trl" to "Trail", "Pk" to "Park")
val STREET_RE = """(Street|Road|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Highway|Parkway|Crescent|Terrace|Circle|Way|""" +
    """Close|Row|Walk|Alley|Plaza|Trail|Pike|Loop|""" + STREET.keys.joinToString("|") + """)\b(\.?)"""
const val ID_WORDS = """[Rr]oom|[Rr]m\.?|[Oo]ffice|[Ss]uite|[Ss]te\.?|[Aa]partment|[Aa]pt\.?|[Uu]nit|[Gg]ate|[Pp]latform|[Ff]light|""" +
    """[Bb]us|[Rr]oute|[Hh]ighway|[Hh]wy\.?|[Ii]nterstate|[Tt]erminal|[Bb]ay|[Dd]ock|[Pp]ier|[Ss]eat|[Cc]abin|""" +
    """[Ww]ard|[Ll]ocker|[Bb]ooth|[Ss]tudio|[Bb]uilding|[Bb]ldg\.?|[Ee]xit|[Pp]ort|[Ee]xt\.?|[Ee]xtension|""" +
    """[Mm]odel|[Bb]oeing|[Aa]irbus|[Cc]heckpoint|[Hh]angar|[Tt]rain|[Cc]oach|[Ff]erry|[Cc]ar"""
val ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y " +
    "Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg " +
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split(" ").toSet()
const val CODE_CTX = """(?i:\b(?:codes?|ref(?:erence)?|serial|PIN|password|passcode|quote|plate|registration|booking|confirmation|""" +
    """tracking|order|voucher|coupon|promo|ticket|ID|key|token|account|policy|claim|case|invoice|SKU|part)\b)[^.;!?]*$"""
val SCHEME_SAY = mapOf("http" to "H T T P", "https" to "H T T P S", "file" to "file", "git" to "git", "data" to "data", "blob" to "blob",
    "chrome" to "chrome", "spotify" to "spotify", "slack" to "slack", "zoom" to "zoom", "steam" to "steam")
const val UNIT_PRE = """(?<![\w.])(?<!\w-)(\d(?:[\d,]*\d)?(?:\.\d+)?)"""