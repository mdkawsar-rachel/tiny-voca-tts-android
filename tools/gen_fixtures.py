"""Generate input -> expected fixtures from the Python text front end (normalizer without NeMo,
respeller, sentence splitter, char tokenizer) for testing the Kotlin port. Output: base64 TSV,
one record per line: text, normalized, split pieces (\\n-joined), token ids (comma-joined)."""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underdog_woof.normalize import Normalizer
from underdog_woof.textsplit import split_text
from underdog_woof.runtime import encode_text

CASES = [
    # plain prose
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "Please call me back at 415-555-0199 before 5:30 p.m. on Friday.",
    "She said, \"I'll be there soon,\" and left.", "What a wonderful day! Isn't it?",
    "Wait... what? No way!", "It was a long, dark, and stormy night; nobody stirred.",
    "The results were surprising -- indeed, shocking.", "He earned $50,000 a year.",
    # phones
    "Call 415-555-0199 now.", "My number is +1 (415) 555-0199.", "Dial 1-800-829-1040 for help.",
    "Try 1-800-FLOWERS today.", "Call 911 in an emergency.", "Reach me at ext. 42 tomorrow.",
    "Text 555123 for info.", "The number is 415.555.0199.", "+44 20 7946 0958 is London.",
    # times
    "Meet at 5:30 p.m.", "The store opens at 9am and closes at 9pm.", "Work is 9am-5pm.",
    "The movie runs 2:03:45.", "It happened at 12:00 noon.", "Departure is 14:00.",
    "Arrive by 08:05.", "The race took 1:02:33.", "Open 10:00-16:30 daily.",
    "At 7:00 the sun rises.", "It's 3:2 in favor.", "A ratio of 16:9 is widescreen.",
    # dates
    "The deadline is 2026-10-21.", "Born on 14.03.2027 in Paris.", "Due 12-25-1999.",
    "See you March 5.", "It happened on 5 March 2027.", "The report covers 3/2027.",
    "Submit by 14/03.", "The ides of 03/14.", "Filed on 01/10/2026.", "Signed 1/10/26.",
    "Open Mon-Fri.", "It ships Tue.", "See you next Monday.", "The week of Jan. 5 is busy.",
    "It ended Dec. 31.", "On 3-17 November we travel.", "The 14th of July is a holiday.",
    "From 1968-2020 the town grew.", "Back in 1990-95 things were different.",
    # money
    "It cost $4.2 million.", "Prices range $50-75.", "The fee is EUR 12.99.", "It was ¥300.",
    "Tea costs £7.50.", "About $1.5 left.", "Only $0.99!", "It was 25¢.", "Pay USD 100.",
    "It cost 100 USD.", "The total is $1,234.56.", "He paid €50.", "A £5-10 tip is fine.",
    "Revenue hit $2.5B last year.", "They raised $300k.", "The budget is $10 million.",
    "It's ₹500 only.", "Costs CAD 40.", "That's 5000 KRW.", "Pay R$ 25.",
    # numbers / ordinals / years / decades
    "There were 1,234 people.", "Pi is 3.14.", "A million is 1000000.", "She finished 42nd.",
    "It's my 1st, 2nd, and 3rd time.", "The 21st century began in 2001.", "The 100th episode aired.",
    "In 1999 we partied.", "It was 2005.", "Welcome to 2026!", "I love the '80s.",
    "The 1990s were great.", "Back in '98.", "The 2000s called.", "She scored 95%.",
    "Order 12345678 shipped.", "Account 00421933 is closed.", "The answer is 42.",
    # fractions
    "Add 1/2 cup of sugar.", "Use 3/4 cup flour.", "I ate 1 1/2 pizzas.", "2/3 of the class passed.",
    "Mix in ½ teaspoon.", "She ran 1½ miles.", "About 1/4 of them agreed.", "A 7/8 inch pipe.",
    "He got 4/5 stars.", "Rated 3/5 by critics.",
    # units / durations / dimensions
    "It is 5km away.", "Dose: 10 mg/kg daily.", "He is 3 ft 5 in tall.", "She is 6'2\" tall.",
    "Water boils at 100°C.", "It's 70°F outside.", "Turn it 90°.", "The file is 5GB.",
    "A 3.5GHz processor.", "It uses 100 kWh.", "The trip takes 2h 15m.", "Wait 45min.",
    "It lasted 3hrs.", "The film is 1h30.", "A 10x20 room.", "The board is 4 × 8 feet.",
    "Drive at 60 mph.", "The drive is 2TB.", "Take 500 mg twice daily.", "It weighs 2.5 kg.",
    "The speed was 100 km/h.", "A 5K run.", "4.5M visitors came.", "The tank holds 30 gal.",
    "It ran for 45m after the start.", "A 100m dash.", "The pool is 50m long.",
    # Roman numerals
    "Henry VIII had six wives.", "Read Chapter XIV.", "World War II ended in 1945.",
    "Super Bowl LIX was close.", "John Smith III arrived.", "Rocky IV is a classic.",
    "Frozen II was popular.", "Type II diabetes.", "See Appendix III.", "Pope Benedict XVI resigned.",
    "Elizabeth II reigned long.", "The mark is IV.", "Act I, Scene II.", "WWII was devastating.",
    # codes / identifiers
    "Order CS-2026-0813 shipped.", "Flight AB-4471-Z is late.", "Code X7K9 unlocks it.",
    "Ref ORD-9Q2-77L please.", "Go to gate B12.", "Flight AA229 boards now.", "An A380 landed.",
    "Room 402 is ready.", "Gate 47B closes soon.", "Flight UA 482 is delayed.", "PIN 0420 works.",
    "Card ending in 4459.", "The last four digits are 1234.", "Serial AB12CD34.", "Seat 14A please.",
    "Model X100 is new.", "Boeing 737 landed.", "The code is 884213.", "Booking ref Q7X9M2.",
    # chemistry
    "Glucose is C6H12O6.", "Water is H2O.", "CO2 levels rose.", "Salt is NaCl.",
    # URLs / email / handles / hashtags
    "Visit https://example.com today.", "Go to www.example.com.", "Open http://localhost:8080.",
    "Mail user.name+tag@example.co.uk.", "Use mailto:a@b.com.", "Fetch ftp://files.example.com.",
    "See https://www.example.co.uk/path.", "Follow @openai.", "Thanks @Jane_Doe!",
    "Trending #MachineLearning.", "#OpenAI announced it.", "Visit example.com for more.",
    "The site sub.example.org works.", "Open https://github.com/user/repo.",
    # paths / files / tech
    r"Open C:\Users\me\file.txt.", "Edit the .env file.", "Set API_KEY first.", "Call getUserName() now.",
    "Open report_final.pdf.", "Run src/main.py.", "The path /usr/local/bin is set.",
    "Use .NET for this.", "Learn C# and C++.", "Update to v2.4.1.", "Version 4.0.1-beta is out.",
    "The config app.yaml is here.", "Run bash script.sh.", "Check logs/error.log.",
    "SELECT * FROM users;", "A snake_case_name here.", "The URL https://a.io/x.",
    # addresses
    "Live at 742 Evergreen Terrace.", "It's at 4 Elm Rd.", "1600 Pennsylvania Avenue NW.",
    "Take I-95 N.", "Drive Route 66 west.", "On Elm St. we met.", "Visit 5th Ave.",
    "The office is 100 Main Street, Suite 200.", "Apt. 5B is upstairs.",
    # math / symbols
    "Solve x^2 = 4.", "Avogadro is 6.02 x 10^23.", "Compute 10^23 particles.",
    "If 3x - 2 = 11 then x is five.", "Clearly 5 > 3.", "It is 2 ± 1.", "About 50% done.",
    "A 3:2 majority.", "Open 24/7.", "It's a 50/50 chance.", "Remember 9/11.",
    "Bread & butter.", "Go → there.", "Ten ÷ two.", "The √ of nine.", "It ≈ five.",
    "x ≠ y.", "Score ≥ 90.", "Limit ≤ 100.", "A 3×4 grid.", "x squared plus y cubed.",
    "2^10 is 1024.", "e^x grows fast.", "The area is 5 m^2.", "A 10 km^2 zone.",
    # names / respellings
    "Siobhan called earlier.", "Saoirse Ronan starred in it.", "Eoin's book is great.",
    "Add Worcestershire sauce.", "I ate gnocchi for lunch.", "A Siobhan-led team won.",
    "It was Worcestershire-glazed.", "The dove-cote stood empty.", "Will you come?",
    "Grace said hello.", "Nice to meet you.", "Siobhán smiled.", "Łukasz visited Kraków.",
    "Renée and Zoë came.", "Beyoncé performed.", "The café opened.", " naïve résumé façade.",
    # postcodes
    "Send to SW1A 1AA please.", "The office is M14 5TG.", "Mail K1A 0B6.",
    # long text for splitting
    "The committee met on Tuesday to discuss the proposal, which had been revised several times "
    "over the past month, and after a long debate they agreed to submit it to the board for final "
    "approval before the end of the quarter.",
    "Although the weather was terrible and the roads were closed, we decided to continue the journey "
    "because the meeting was important and could not be rescheduled at such short notice.",
    "This is a short one.",
    "First sentence here. Second sentence follows! A third one? Yes, indeed.",
    # mixed / tricky
    "The meeting is at 5:30 p.m. on 2026-10-21 in Room 402; bring $25.50 for lunch (about £20).",
    "Call +1-800-555-0199 or email help@example.com before Fri., 12/05.",
    "He ranked 3rd with 98.6% on 03/14, earning $1,000 and a 1/2 share.",
    "Version v3.2.1 ships 2026-11-01 with C# support and a 2x faster engine.",
    "From 9am-5pm, Mon-Fri, the office at 742 Evergreen Terrace (Room 402) is open.",
]


def b64(s):
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fixtures.tsv"
    norm = Normalizer(use_nemo=False)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for t in CASES:
            spoken = norm(t)
            pieces = split_text(spoken)
            ids = encode_text(spoken)
            f.write("\t".join([b64(t), b64(spoken), b64("\n".join(pieces)),
                               b64(",".join(map(str, ids)))]) + "\n")
            n += 1
    print(f"wrote {n} fixtures -> {out}")


if __name__ == "__main__":
    main()