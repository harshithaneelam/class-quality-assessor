"""Local 3-layer name de-identification for class transcripts.

Layer 1  context patterns: self-introductions, vocatives, teacher address forms.
Layer 2  Indian + international name gazetteer, spaCy PERSON entities and a
         capitalised-proper-noun heuristic, all filtered by a whitelist of
         English / chess / place / calendar / platform words.
Layer 2b rare-word rule: tokens with English Zipf frequency < 1.5 (wordfreq)
         that no other layer explains are masked as low-confidence [NAME]
         (Whisper often writes unfamiliar names in lowercase or only at sentence
         start). They join a student cluster if they are a spelling variant.
Layer 3  fuzzy clustering of spelling variants (rapidfuzz ratio >= 82, or same
         4-char prefix and length within 2) -> [TEACHER], [STUDENT_1..N] in order
         of first appearance, then a global case-insensitive whole-word pass.

PRIVACY: discovered names live only inside this function call; they are never
returned, logged or written anywhere.
"""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

# --------------------------------------------------------------- word lists
FIRST_NAMES = """
aadhi aadhya aadi aadil aadit aadithya aadvik aahana aakash aalia aamir aanya aaradhya aarav aariz
aarna aarohi aarush aarya aashi aayush abhay abhijith abhinav abhishek abin abu adarsh adeeb adhil
adhira adhvik adil aditi aditya adithya adithyan advaith advait advika afra afsal agastya ahaan ahana
ahmed aishwarya ajay ajith akash akhil akshara akshay alan albin alen alia alisha alvin amaan amal
amala amay ameya amina amir amit amitha amrita amruth anagha anaika anamika ananya anand ananth
anay anika anil anirudh anish anjali anju ankit ankita ann anna annie anoop anshika anu anupama
anvi anvitha anya aparna archana arhaan arijit arjun arnav aron arpit arun aruna arvind arya aryan
asha ashik ashish ashwin ashwini athira athul atharv avani avantika avni ayaan ayan ayesha ayush
bhavana bhavya chaitanya chandana chetan chinmay daksh darsh darshan deepa deepak deepika dev devansh
devika devna dhanush dhruv dhruva diksha dilip dinesh divya diya drishti ekta eshaan esha eshan
farhan fathima fatima gauri gautam gautham gayathri gayatri geetha gokul gopal gopika govind gowri
hamsa hamza hanna hannah hari harini harish harsha harshit hemant hiba hridya ibrahim irfan
isha ishaan ishan ishika ishita jahnavi janvi jayesh jeevan jithin jiya jyothi jyoti kabir kailash
kalyani kamal karan karthik kartik karthika kashvi kavya keerthana keerthi kian kiara kiran krish
krishna krithika kriti kunal lakshmi lavanya madhav madhu mahesh maithili malavika manav manasa
manish manoj maya medha meera megha meghna mehul midhun mihir mira mithun mohammed mohammad mohan
mohit mridul muhammad muskan naina nakul namitha nanda nandana nandini naveen navya navaneeth neel
neha nehal nidhi nihal nikhil nila nirmal niranjan niranjana nisha nishant nithin nithya nivedita
niya ojas pallavi pankaj parth parvathi pavan pooja pradeep pranav pranavi prarthana prateek pratik
praveen preeti prisha priya priyanka rachna radha raghav rahul rajat rajesh rakesh ramesh rani
rashmi ravi reeva rehan reyansh rhea riaan riddhi rishabh rishi ritika ritvik riya rohan rohit
ronak roshan ruchi rudra rupa saanvi sachin sahil sai saisha sakshi sameer samarth samhita sana
sandeep sanjana sanjay sanvi sara sarah saransh sarthak satish saurabh shaurya shiva shivam shivani
shlok shreya shreyas shriya shruti siddharth siddhi simran sita siya smriti sneha sonam sooraj sourav
sreya sreehari sreya sruthi srinivas sudhir suhana sujith sunil suraj surya swathi tanmay tanvi tanya
tara tejas tharun trisha tushar uday ujjwal urvi vaibhav vaishnav vaishnavi varun varsha vasudha
ved vedant veda vihaan vikram vimal vinay vineeth vinod vishal vishnu vivaan vivek yash yashika
yuvan yuvraj zara zayn zoya
aaron adam aiden alex alexander alice amelia andrew anthony ava benjamin caleb charlotte chloe
christopher daniel david dylan elena elijah ella emily emma ethan evan gabriel hanna harper henry
isaac isabella jacob james jason jessica john jonathan joseph joshua julia kevin liam lucas matthew
mia michael natalie nathan noah olivia ryan samuel sebastian sophia steven thomas tyler william zoe
"""
SURNAMES = """
agarwal ahmed ali bhat bhatt chakraborty chandra chatterjee chaudhary das desai dutta fernandes gandhi
george ghosh gupta iyer iyengar jain jose joshi kapoor khan krishnan kulkarni kumar kurian kutty
mathew mehta menon mishra mohan mukherjee nair namboodiri pandey patel patil philip pillai prasad
rajan rao reddy saxena sebastian shah sharma shetty singh sinha srivastava subramanian thomas
tiwari varghese varma verma warrier yadav
"""
GAZETTEER = frozenset(FIRST_NAMES.split()) | frozenset(SURNAMES.split())

CONTEXT_STOP = set("""
fine good here seven years ready going playing sharing knight bishop rook queen king pawn check
checkmate castle stalemate fork pin lichess white black
""".split())

WHITELIST = set("""
i ok okay yes yeah yep no nope so and but now then very great sorry thank thanks please hello hi hey
bye welcome everyone everybody all right correct wrong wait listen look see let come go what why how
when where who which is are am was can could will would should do did does have has had it this that
there these those they we you he she my your our their his her its one two three four five six seven
eight nine ten eleven twelve twenty hundred first second third last next today tomorrow yesterday
morning evening afternoon night class puzzle puzzles homework game games move moves play player players
opponent point points level team group sir maam ma'am mam madam miss mister coach teacher tutor
trainer mentor mr mrs ms dr student students kids children friend friends guys
chess knight knights bishop bishops rook rooks queen queens king kings pawn pawns check checkmate mate
castle castling castles stalemate fork forks pin pins skewer skewers lichess board boards square
squares file files rank ranks diagonal diagonals opening openings endgame middlegame gambit sicilian
french italian spanish english ruy lopez caro kann scholar scholars fried liver passant promotion
tactic tactics discovered double attack defence defense draw resign blunder trap traps center centre
magnus carlsen kasparov fischer gukesh praggnanandhaa nakamura hikaru caruana capablanca morphy karpov
kramnik polgar viswanathan
fianchetto zugzwang zwischenzug intermezzo desperado outpost isolani opposition triangulation luft
prophylaxis prise smothered philidor lucena petrov petroff scandinavian alekhine pirc nimzo nimzowitsch
grunfeld catalan benoni dutch berlin najdorf dragon chessbrainz chesskid chessable chessbase stockfish
checkmated stalemated underpromotion
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december
india indian kerala kochi cochin trivandrum calicut bangalore bengaluru chennai mumbai delhi
hyderabad pune kolkata goa dubai abu dhabi sharjah uae emirates qatar doha saudi oman muscat kuwait
bahrain us usa america american uk britain england london canada australia singapore germany
ireland europe asia africa hindi malayalam tamil telugu kannada arabic urdu marathi gujarati bengali
zoom google meet teams whatsapp youtube mac windows ipad iphone android samsung apple microsoft gmail
powerpoint chrome internet wifi screen mic camera video audio chat link
god christmas diwali onam eid easter
""".split())

ADDRESS_TITLES = r"(?:coach|teacher|tutor|miss|mister|mr|mrs|ms|madam)"
POST_TITLES = r"(?:sir|ma'?am|madam|miss)"
NAME = r"([A-Z][a-z]{2,15})"
NAME2 = r"([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)"
STAFF = r"(?:chess\s+)?(?:coach|teacher|tutor|trainer|mentor|instructor)"

_TEACHER_PATTERNS = [  # (pattern, only_in_first_5_min)
    (re.compile(rf"(?i:\b(?:i am|i'm|this is|my name is)\s+(?:your\s+)?{STAFF})\s*,?\s*{NAME2}"), False),
    (re.compile(rf"(?i:\b(?:i am|i'm|this is|my name is))\s+{NAME2}\s*,?\s*(?i:(?:and\s+)?(?:i am|i'm|i'll be|i will be)?\s*your\s+{STAFF})"), False),
    (re.compile(rf"(?i:\byour\s+{STAFF}\s+(?:is|will be)?)\s*{NAME}"), False),
    (re.compile(rf"(?i:\b{ADDRESS_TITLES})\.?\s+{NAME}"), False),
    (re.compile(rf"{NAME}\s+(?i:{POST_TITLES})\b"), False),
]
_SELF_INTRO = [
    re.compile(rf"(?i:\bmy name is)\s+{NAME2}"),
    re.compile(rf"(?i:\bi am|\bi'm)\s+{NAME}"),
    re.compile(rf"(?i:\bthis is)\s+{NAME}"),
]
_VOCATIVE = [
    re.compile(r"(?i:\b(?:yes|good job|well done|very good|correct|try again|unmute|go ahead|tell me|"
               r"what about|how about|hi|hello|welcome|bye|thank you|okay))\s*,?\s*" + NAME),
    re.compile(NAME + r"\s*,\s*(?i:can you|what is|your turn|unmute|tell me|are you there)"),
]
_CLASSY = re.compile(r"(?i)\b(today|class|welcome|we will|we'll|i will|i'll|let's|session)\b")
_PLACEHOLDER = re.compile(r"\[(?:TEACHER|STUDENT_\d+)\]")
_STUDENT_PH = re.compile(r"\[STUDENT_\d+\]")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _ok_token(tok: str) -> bool:
    t = tok.lower().strip("'")
    return (3 <= len(t) <= 15 and t.isalpha() and t not in WHITELIST and t not in CONTEXT_STOP)


@lru_cache(maxsize=1)
def _nlp():
    """spaCy en_core_web_sm NER, or None if unavailable (gazetteer/heuristics still run)."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except OSError:
            try:
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except Exception:
                return None
    except ImportError:
        return None


RARE_ZIPF = 1.5


def _zipf():
    """wordfreq English Zipf lookup, or None if wordfreq is not installed."""
    try:
        from wordfreq import zipf_frequency
        return lambda w: zipf_frequency(w, "en")
    except ImportError:
        return None


def _stop_words() -> set[str]:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS
        return set(STOP_WORDS)
    except ImportError:
        return set()


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _similar(a: str, b: str) -> bool:
    if fuzz.ratio(a, b) >= 82:
        return True
    return len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4] and abs(len(a) - len(b)) <= 2


def deidentify_transcript(segments: list[dict], extra_known_names: list[str] | None = None) -> dict:
    """Mask person names in Whisper segments with consistent role placeholders."""
    texts = [s.get("text", "") for s in segments]
    starts = [float(s.get("start", 0.0)) for s in segments]
    stop = _stop_words()
    extra = {t.lower() for n in (extra_known_names or []) for t in re.findall(r"[A-Za-z]+", n) if len(t) >= 3}

    strength: dict[str, int] = {}   # token -> 3 strong, 2 medium, 1 weak
    weak_count: dict[str, int] = {}
    teacher_tokens: set[str] = set()
    uf = _UF()

    def add(group: list[str], level: int, teacher: bool = False):
        toks = [t.lower() for t in group if _ok_token(t)]
        for t in toks:
            strength[t] = max(strength.get(t, 0), level)
            uf.find(t)
            if teacher:
                teacher_tokens.add(t)
        for a, b in zip(toks, toks[1:]):
            uf.union(a, b)

    # Layer 1 — context patterns
    for i, text in enumerate(texts):
        early = starts[i] <= 300
        for rx, _ in _TEACHER_PATTERNS:
            for m in rx.finditer(text):
                add(m.group(1).split(), 3, teacher=True)
        for k, rx in enumerate(_SELF_INTRO):
            for m in rx.finditer(text):
                toks = m.group(1).split()
                if k == 0 and early and not teacher_tokens and len(text.split()) >= 10 and _CLASSY.search(text):
                    add(toks, 3, teacher=True)  # tutor's own intro at the start of class
                else:
                    add(toks, 3)
        for rx in _VOCATIVE:
            for m in rx.finditer(text):
                add([m.group(1)], 3)

    # Layer 2 — gazetteer (case-insensitive for >= 4 letters), extra names, spaCy, capitalisation
    for i, text in enumerate(texts):
        for m in _WORD.finditer(text):
            w = m.group(0)
            lw = w.lower()
            if lw in extra:
                add([w], 3)
            elif lw in GAZETTEER and (len(lw) >= 4 or w[0].isupper()):
                add([w], 2)
            elif (w[0].isupper() and not w.isupper() and lw not in stop and _ok_token(w)):
                before = text[:m.start()].rstrip()
                if before and before[-1] not in ".?!\"'":  # mid-sentence capital
                    weak_count[lw] = weak_count.get(lw, 0) + 1
                    strength.setdefault(lw, 1)
                    uf.find(lw)
    nlp = _nlp()
    if nlp is not None:
        for doc in nlp.pipe(texts, batch_size=64):
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    add([t.text for t in ent if t.text[:1].isupper()], 2)

    # Layer 2b — rare-word rule (any casing, including sentence starts)
    rare: set[str] = set()
    zipf = _zipf()
    if zipf is not None:
        seen = set()
        for text in texts:
            for w in _WORD.findall(text):
                lw = w.lower()
                if lw in seen or lw in strength or len(lw) < 4 or lw in stop or not _ok_token(w):
                    continue
                seen.add(lw)
                if zipf(lw) < RARE_ZIPF:
                    rare.add(lw)
                    strength[lw] = 0
                    uf.find(lw)

    # Adjacent candidate tokens ("Arjun Nair") belong to one person.
    cand = set(strength)
    for text in texts:
        ws = [w.lower() for w in _WORD.findall(text)]
        for a, b in zip(ws, ws[1:]):
            if a in cand and b in cand and a != b:
                uf.union(a, b)

    # Layer 3 — fuzzy clustering of spelling variants
    toks = sorted(cand)
    for i, a in enumerate(toks):
        for b in toks[i + 1:]:
            if _similar(a, b):
                uf.union(a, b)
    comps: dict[str, set[str]] = {}
    for t in toks:
        comps.setdefault(uf.find(t), set()).add(t)
    kept = [c for c in comps.values()
            if any(strength[t] >= 2 for t in c) or sum(weak_count.get(t, 0) for t in c) >= 2]
    low_conf = [c for c in comps.values() if c not in kept and c & rare]

    # Order of first appearance, then placeholders
    variant_rx = {}
    first_pos: dict[int, tuple[int, int]] = {}
    for ci, comp in enumerate(kept):
        alts = "|".join(re.escape(v) for v in sorted(comp, key=len, reverse=True))
        rx = re.compile(rf"(?<![\w\[])(?:{alts})(?![\w\]])", re.I)
        variant_rx[ci] = rx
        for si, text in enumerate(texts):
            m = rx.search(text)
            if m:
                first_pos[ci] = (si, m.start())
                break
    placeholder: dict[int, str] = {}
    n_student = 0
    for ci in sorted(first_pos, key=first_pos.get):
        if kept[ci] & teacher_tokens:
            placeholder[ci] = "[TEACHER]"
        else:
            n_student += 1
            placeholder[ci] = f"[STUDENT_{n_student}]"

    lookup = {v: placeholder[ci] for ci in placeholder for v in kept[ci]}
    for comp in low_conf:
        for v in comp:
            lookup.setdefault(v, "[NAME]")
    if lookup:
        alts = "|".join(re.escape(v) for v in sorted(lookup, key=len, reverse=True))
        global_rx = re.compile(rf"(?<![\w\[])(?:{alts})(?:'s)?(?![\w\]])", re.I)
    else:
        global_rx = None

    total = low_conf_hits = 0
    counts: dict[str, int] = {}
    teacher_masked = False
    masked = []
    for seg, text in zip(segments, texts):
        new = text
        if global_rx is not None:
            def _sub(m):
                nonlocal total, teacher_masked, low_conf_hits
                total += 1
                s = m.group(0)
                base = s[:-2] if s.lower().endswith("'s") else s
                ph = lookup[base.lower()]
                if ph == "[TEACHER]":
                    teacher_masked = True
                elif ph == "[NAME]":
                    low_conf_hits += 1
                return ph + ("'s" if base is not s else "")
            new = global_rx.sub(_sub, text)
            new = re.sub(r"(\[(?:TEACHER|STUDENT_\d+|NAME)\])(?:\s+\1)+", r"\1", new)
        mentioned = list(dict.fromkeys(_STUDENT_PH.findall(new)))
        for ph in mentioned:
            counts[ph] = counts.get(ph, 0) + len(re.findall(re.escape(ph), new))
        masked.append({**seg, "text": new, "students_mentioned": mentioned})

    return {
        "masked_segments": masked,
        "stats": {
            "teacher_name_masked": teacher_masked,
            "distinct_student_placeholders": n_student,
            "total_mask_replacements": total,
            "low_confidence_name_masks": low_conf_hits,
            "per_student_mention_counts": dict(sorted(counts.items(), key=lambda kv: int(kv[0][9:-1]))),
        },
    }
