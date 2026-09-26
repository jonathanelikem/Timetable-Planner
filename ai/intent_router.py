"""
Semantic intent interpretation for the Ask tab.

This is the one component in the system where AI is appropriate, and it
is deliberately confined to a single job: deciding *which* timetable
function should answer a question. It never computes an answer. Clash
detection, congestion scoring and combination ranking all remain exact
arithmetic in core/ and analytics/, because a model that is usually
right about whether 11:30-13:20 overlaps 12:00-13:50 is not good enough
for a decision that affects whether a student graduates on time.

Two interpreters are available, tried in order:

  1. SENTENCE EMBEDDINGS (sentence-transformers). Encodes the question
     and a set of example phrasings per intent, and takes the closest
     match by cosine similarity. Genuine semantic matching: it handles
     wording never seen before, so "will these two courses collide?"
     resolves to clash_check without the word "clash" appearing.

  2. LEXICAL SIMILARITY (no dependencies, no download). A TF-IDF style
     cosine over the same example phrasings. Weaker than embeddings on
     unseen wording, but it needs no model file, so the Ask tab still
     works on a machine with no network — which matters when the system
     is being demonstrated on someone else's laptop.

If neither is confident, the function returns None and app.py falls back
to its keyword rules. Three layers, each degrading into the next, and
the user is always told which one answered.

Interface (kept stable so an alternative implementation can be dropped
in without touching app.py):

    interpret_intent_with_score(question) -> (intent | None, score)
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Intent labels. These must match the branches in app.py's route().
CLASH_CHECK = "clash_check"
FREE_TIME = "free_time"
WORKLOAD_HEAVIEST = "workload_heaviest"
WORKLOAD_LIGHTEST = "workload_lightest"
CAMPUS_DAYS = "campus_days"
COURSE_LOOKUP = "course_lookup"

# Example phrasings per intent. The interpreter matches against these
# rather than against keywords, so adding a way of asking something is a
# matter of adding a sentence here. Written the way students actually
# type, including the blunt short forms.
EXAMPLES: dict[str, list[str]] = {
    CLASH_CHECK: [
        "can I take these two courses together",
        "do these courses clash",
        "is there a conflict between these classes",
        "do these two run at the same time",
        "will these courses collide",
        "can I register for both of these",
        "are these compatible",
        "do these overlap on my timetable",
        "is it possible to do both of these this semester",
        "would these two courses work together",
    ],
    FREE_TIME: [
        "when am I free",
        "what gaps do I have between classes",
        "when is my longest break",
        "do I have any free periods",
        "when could I schedule something",
        "what time am I available during the week",
        "when can I meet someone",
        "how much spare time do I have",
        "which day do I have nothing on",
        "when is my free time",
    ],
    WORKLOAD_HEAVIEST: [
        "which day is busiest",
        "what is my heaviest day",
        "when am I most loaded",
        "which day has the most classes",
        "what is my worst day",
        "which day is most congested",
        "when is my schedule packed",
        "which is my hardest day of the week",
    ],
    WORKLOAD_LIGHTEST: [
        "which day is lightest",
        "what is my easiest day",
        "which day has the fewest classes",
        "when is my quietest day",
        "which day am I least busy",
        "what is my most relaxed day",
        "which teaching day has the smallest load",
    ],
    CAMPUS_DAYS: [
        "which days do I need to be on campus",
        "when do I have to come in",
        "do I need to travel to school that day",
        "which classes are online",
        "can I stay home on any day",
        "how many days do I commute",
        "do I have to be there in person",
        "which days are in person",
    ],
    COURSE_LOOKUP: [
        "when does this course run",
        "what time is this class",
        "where is this course held",
        "what venue is this class in",
        "tell me about this course",
        "when is this lecture",
        "what are the details for this course",
    ],
}

# Below this, the interpreter declines and the caller falls back to
# keyword rules. Set by inspection rather than measurement: high enough
# that an unrelated sentence is refused, low enough that ordinary
# rephrasing is accepted. A tunable, not a fact.
EMBEDDING_THRESHOLD = 0.42
LEXICAL_THRESHOLD = 0.30

_STOPWORDS = {
    "a", "am", "an", "and", "any", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "have", "how", "i", "if", "in", "is", "it", "me",
    "my", "of", "on", "or", "that", "the", "these", "this", "those", "to",
    "two", "up", "was", "what", "when", "where", "which", "will", "with",
    "would", "you", "your", "there", "here", "hello", "hi", "hey",
    "please", "thanks", "thank", "ok", "okay", "just", "also", "about",
}

# Populated on first use. None means "not tried yet", False means "tried
# and unavailable", so the import cost is paid at most once.
_model = None
_example_vectors = None


def _tokenise(text: str) -> list[str]:
    """Words only, lowercased, stopwords and course codes removed.

    Course codes are stripped because they identify *which* courses the
    question is about, not *what* is being asked. Leaving them in would
    make every question about OMIS 404 look similar regardless of intent.
    """
    text = re.sub(r"\b[a-zA-Z]{2,5}\s?\d{3}[a-zA-Z]?\b", " ", text.lower())
    words = re.findall(r"[a-z]+", text)
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _load_embedding_model():
    """Load sentence-transformers if it is installed and a model is cached.

    Deliberately silent on failure. A missing package or an absent model
    file is an expected state, not an error: the lexical interpreter
    below covers it, and the user is told which path answered.
    """
    global _model, _example_vectors

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        _model = False
        return False

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        flat, labels = [], []
        for intent, phrases in EXAMPLES.items():
            for phrase in phrases:
                flat.append(phrase)
                labels.append(intent)
        vectors = model.encode(flat, convert_to_tensor=True,
                               show_progress_bar=False)
        _model = (model, util)
        _example_vectors = (vectors, labels)
        return _model
    except Exception:
        # Most often: no cached model and no network to fetch one.
        _model = False
        return False


def _embedding_intent(question: str):
    """Closest intent by cosine similarity over sentence embeddings."""
    loaded = _load_embedding_model()
    if not loaded:
        return None, 0.0

    model, util = loaded
    vectors, labels = _example_vectors

    try:
        q = model.encode(question, convert_to_tensor=True,
                         show_progress_bar=False)
        scores = util.cos_sim(q, vectors)[0]
        best = int(scores.argmax())
        return labels[best], float(scores[best])
    except Exception:
        return None, 0.0


def _lexical_intent(question: str):
    """Closest intent by TF-IDF cosine. No model, no download, no network.

    Scores the question against each intent's examples and takes the best
    single example rather than the intent average, so one closely matching
    phrasing is enough — which is how people actually ask.
    """
    q_tokens = _tokenise(question)
    if not q_tokens:
        return None, 0.0

    documents = [(intent, _tokenise(p))
                 for intent, phrases in EXAMPLES.items()
                 for p in phrases]

    # Inverse document frequency over the example corpus, so common words
    # ("day", "course") count for less than distinguishing ones
    # ("clash", "online", "lightest").
    n_docs = len(documents)
    doc_freq = Counter()
    for _, tokens in documents:
        for token in set(tokens):
            doc_freq[token] += 1

    def weight(token: str) -> float:
        return math.log((1 + n_docs) / (1 + doc_freq.get(token, 0))) + 1.0

    def vector(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {t: counts[t] * weight(t) for t in counts}

    def cosine(a: dict, b: dict) -> float:
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[t] * b[t] for t in shared)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    q_vec = vector(q_tokens)
    best_intent, best_score, best_shared = None, 0.0, ()
    for intent, tokens in documents:
        score = cosine(q_vec, vector(tokens))
        if score > best_score:
            best_intent = intent
            best_score = score
            best_shared = tuple(set(q_tokens) & set(tokens))

    # A match resting on one shared word is only trustworthy when that
    # word is distinctive. "overlap" appears in one intent and settles
    # the question on its own; a word spread across many examples does
    # not, and matching on it would be coincidence dressed up as
    # understanding.
    if len(best_shared) == 1:
        rare = weight(best_shared[0]) >= math.log((1 + n_docs) / 3) + 1.0
        if not rare and best_score < 0.72:
            return None, 0.0

    return best_intent, best_score


def interpret_intent_with_score(question: str):
    """
    Work out what the student is asking.

    Returns (intent, score). Intent is None when neither interpreter is
    confident enough, which tells the caller to fall back to its keyword
    rules rather than guessing.
    """
    if not question or not question.strip():
        return None, 0.0

    intent, score = _embedding_intent(question)
    if intent and score >= EMBEDDING_THRESHOLD:
        return intent, score

    intent, score = _lexical_intent(question)
    if intent and score >= LEXICAL_THRESHOLD:
        return intent, score

    return None, 0.0


def interpreter_in_use() -> str:
    """Which interpreter is active, for display in the interface."""
    loaded = _load_embedding_model()
    return "sentence embeddings" if loaded else "lexical similarity"
