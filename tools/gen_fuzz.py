"""Emit randomized parity fixtures: random joins of tricky fragments plus light mutations of the
curated cases, to stress Java-vs-Python regex edge paths. Same base64 TSV format as gen_fixtures."""
import base64
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underdog_woof.normalize import Normalizer
from underdog_woof.textsplit import split_text
from underdog_woof.runtime import encode_text
from gen_fixtures import CASES

FRAGS = [
    "Call", "Dial", "Text", "415-555-0199", "+1 (212) 555-0147", "1-800-555-1234", "1-800-GOT-JUNK",
    "ext. 99", "PIN 0042", "card ending in 7788", "the last four digits are 9021",
    "at 5:30 p.m.", "by 14:00", "9am-5pm", "2:03:45", "at 12:00 noon", "10:00-16:30", "3:2", "ratio of 16:9",
    "on 2026-10-21", "14.03.2027", "12-25-1999", "March 5", "5 March 2027", "3/2027", "14/03", "03/14",
    "01/10/2026", "1/10/26", "Mon-Fri", "Tue.", "next Monday", "Jan. 5", "Dec. 31", "3-17 November",
    "1968-2020", "1990-95", "the 14th of July",
    "$4.2 million", "$50-75", "EUR 12.99", "¥300", "£7.50", "$1.5", "$0.99", "25¢", "USD 100", "100 USD",
    "$1,234.56", "€50", "£5-10", "$2.5B", "$300k", "$10 million", "₹500", "CAD 40", "5000 KRW", "R$ 25",
    "1,234", "3.14", "1000000", "42nd", "1st", "3rd", "21st", "100th", "1999", "2005", "2026", "'80s",
    "1990s", "'98", "the 2000s", "95%", "12345678", "00421933", "42", "v2", "x5", "A1", "B52",
    "1/2 cup", "3/4 cup", "1 1/2", "2/3 of the", "½", "1½", "1/4", "7/8 inch", "4/5 stars", "3/5",
    "5km", "10 mg/kg", "3 ft 5 in", "6'2\"", "100°C", "70°F", "90°", "5GB", "3.5GHz", "100 kWh", "2h 15m",
    "45min", "3hrs", "1h30", "10x20", "4 × 8", "60 mph", "2TB", "500 mg", "2.5 kg", "100 km/h", "5K",
    "4.5M visitors", "30 gal", "45m", "100m dash", "50m",
    "Henry VIII", "Chapter XIV", "World War II", "Super Bowl LIX", "John Smith III", "Rocky IV", "Frozen II",
    "Type II", "Appendix III", "Benedict XVI", "Elizabeth II", "Act I", "WWII", "Louis XIV", "King George V",
    "CS-2026-0813", "AB-4471-Z", "X7K9", "ORD-9Q2-77L", "gate B12", "AA229", "A380", "Room 402", "47B",
    "UA 482", "Seat 14A", "Model X100", "Boeing 737", "884213", "Q7X9M2", "AB12CD34",
    "C6H12O6", "H2O", "CO2", "NaCl", "Fe2O3",
    "https://example.com", "www.example.com", "http://localhost:8080", "user.name+tag@example.co.uk",
    "mailto:a@b.com", "ftp://files.example.com", "https://www.example.co.uk/path", "@openai", "@Jane_Doe",
    "#MachineLearning", "#OpenAI", "example.com", "sub.example.org", "https://github.com/user/repo",
    r"C:\Users\me\file.txt", ".env", "API_KEY", "getUserName()", "report_final.pdf", "src/main.py",
    "/usr/local/bin", ".NET", "C#", "C++", "v2.4.1", "4.0.1-beta", "app.yaml", "script.sh", "logs/error.log",
    "SELECT * FROM users", "snake_case_name", "https://a.io/x",
    "742 Evergreen Terrace", "4 Elm Rd", "1600 Pennsylvania Avenue NW", "I-95 N", "Route 66", "Elm St.",
    "5th Ave", "100 Main Street, Suite 200", "Apt. 5B",
    "x^2 = 4", "6.02 x 10^23", "10^23", "3x - 2 = 11", "5 > 3", "2 ± 1", "50%", "24/7", "50/50", "9/11",
    "Bread & butter", "→", "÷", "√", "≈", "≠", "≥ 90", "≤ 100", "3×4", "x squared", "2^10", "e^x",
    "5 m^2", "10 km^2",
    "Siobhan", "Saoirse Ronan", "Eoin's", "Worcestershire", "gnocchi", "Siobhan-led", "Worcestershire-glazed",
    "dove-cote", "Will", "Grace", "Nice", "Siobhán", "Łukasz", "Kraków", "Renée", "Zoë", "Beyoncé", "café",
    "naïve", "résumé", "façade", "São Paulo", "Zürich", "Malmö", "the sake", "Will Smith",
    "SW1A 1AA", "M14 5TG", "K1A 0B6",
]
PUNCT = [". ", ", ", "; ", "! ", "? ", " -- ", " and ", " or ", " ", " "]


def b64(s):
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    out = sys.argv[2] if len(sys.argv) > 2 else "fuzz.tsv"
    rng = random.Random(1234)
    norm = Normalizer(use_nemo=False)
    cases = []
    for _ in range(n):
        k = rng.randint(1, 6)
        s = ""
        for j in range(k):
            s += rng.choice(FRAGS)
            if j < k - 1:
                s += rng.choice(PUNCT)
        if rng.random() < 0.5:
            s = s[:1].upper() + s[1:]
        cases.append(s)
    # light mutations of curated cases: swap case of first char, append punctuation, wrap in quotes
    for t in CASES:
        cases.append(t.lower())
        cases.append('"' + t + '"')
        cases.append(t + " Thanks!")
    seen = set()
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for t in cases:
            if t in seen or not t.strip():
                continue
            seen.add(t)
            spoken = norm(t)
            pieces = split_text(spoken)
            ids = encode_text(spoken)
            f.write("\t".join([b64(t), b64(spoken), b64("\n".join(pieces)), b64(",".join(map(str, ids)))]) + "\n")
            written += 1
    print(f"wrote {written} fuzz fixtures -> {out}")


if __name__ == "__main__":
    main()