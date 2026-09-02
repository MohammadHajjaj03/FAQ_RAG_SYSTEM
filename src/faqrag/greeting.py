"""Instant replies for greetings, so a "السلام عليكم" never pays for retrieval.

A greeting carries no information need: there is nothing in the FAQ corpus to
retrieve and nothing for the model to ground an answer in. Running the full
pipeline on one costs roughly ten seconds and usually ends in a refusal, which
reads as a broken bot. Matching them up front turns that into an immediate,
friendly reply that invites the actual question.

Detection is deliberately conservative. It fires only when the *whole* message
is greeting material -- "كيفك؟" matches, "السلام عليكم، كيف أسدد الفاتورة؟"
does not, and goes down the normal path.
"""

from __future__ import annotations

import random
import re

from .lang import Language, detect_language, normalise

# Written against normalised text (see :func:`~faqrag.lang.normalise`): no
# tashkeel, أ/إ/آ folded to ا, ة to ه, ى to ي. So "تحية" is matched as "تحيه"
# and "أهلاً" as "اهلا".
_AR_GREETINGS = (
    r"السلام\s*عليكم(?:\s*ورحمه\s*الله)?(?:\s*وبركاته)?",
    r"وعليكم\s*السلام(?:\s*ورحمه\s*الله)?(?:\s*وبركاته)?",
    r"سلام(?:ات)?",
    r"مرحبا?(?:\s*بك|\s*بكم)?",
    r"مراحب",
    r"هلا(?:\s*و?الله|\s*بك|\s*فيك|\s*هلا)?",
    r"اهلا?(?:\s*و?سهلا?)?(?:\s*بك|\s*بكم|\s*فيك)?",
    r"يا\s*هلا",
    r"صباح\s*(?:الخير|النور|الفل|الورد)",
    r"مساء\s*(?:الخير|النور|الفل|الورد)",
    r"تحيه|تحياتي",
    r"كيف\s*(?:ك|كم|حالك|حالكم|الحال|الاحوال|امورك)",
    r"شلونك|شخبارك|شخبارك?م|وش\s*اخبارك",
    r"(?:ايش|وش|شو)\s*(?:اخبارك|الاخبار|علومك|مسوي)",
    r"اخبارك|اخباركم|عساك\s*طيب|عسا?كم\s*بخير",
    r"قواك\s*الله",
    r"الله\s*يقويك",
    r"حياك\s*الله|حياك",
    r"يعطيك\s*العافيه|الله\s*يعافيك",
    r"طال\s*عمرك",
)

_EN_GREETINGS = (
    r"h(?:i+|ey+|ello+|owdy)",
    r"yo+",
    r"good\s*(?:morning|afternoon|evening|day)",
    r"greetings",
    # Apostrophes are stripped before matching, so "what's up" arrives as
    # "what s up".
    r"what\s*'?s?\s*up|wass?up|sup",
    r"how\s*(?:are\s*(?:you|u)|r\s*u|'?s?\s*it\s*going|are\s*things|have\s*you\s*been)",
    r"peace\s*be\s*upon\s*you",
    r"as-?salam[ou]?\s*alaykum|salam",
)

# One greeting, optionally chained to more ("هلا والله، كيفك يا ابو سهل") and
# optionally trailed by an address or filler. Anchored at both ends: a message
# that contains anything else is a real question wearing a polite hat.
_FILLER = r"(?:\s*(?:يا|و|ال)?\s*)"
_SEPARATOR = r"(?:\s*[,،؛;\-]?\s*(?:و\s*)?)"
_ANY_GREETING = "(?:" + "|".join(_AR_GREETINGS + _EN_GREETINGS) + ")"
# A vocative may follow any greeting in the chain -- "السلام عليكم سهل، كيفك؟"
# addresses the bot mid-message, not only at the end. It is always optional trim
# around a real greeting and never a match on its own, or "يا ريت أعرف كيف أسدد"
# would be mistaken for small talk.
#
# The bare-name form is restricted to the bot's own name. Allowing any bare word
# after a greeting would swallow real questions -- "مرحبا كم رسوم التحويل" would
# read its "كم" as a name and never reach retrieval.
_VOCATIVE = (
    r"(?:\s*(?:يا\s*)?(?:ابو|ام|استاذ|دكتور|شيخ|مهندس|كابتن|اخ|اخوي)\s*\w+"
    r"|\s*يا\s*\w+"
    r"|\s*(?:ابو\s*)?سهل"
    r"|\s*(?:abu\s*)?sahl"
    r"|\s*(?:there|guys|folks|everyone|all|mate|friend))?"
)
_GREETING_ONLY_RE = re.compile(
    rf"^{_FILLER}{_ANY_GREETING}{_VOCATIVE}"
    rf"(?:{_SEPARATOR}{_ANY_GREETING}{_VOCATIVE})*[\s!?.،؟…ـ]*$",
    re.IGNORECASE,
)

# Punctuation and emoji are noise here; strip them before matching so that
# "السلام عليكم!! 😄" is still a bare greeting.
_STRIP_RE = re.compile(r"[^\w\s]|_", re.UNICODE)

# The salam is a formula with a prescribed answer, so it is the one greeting
# that does not get a random reply: returning the full ورحمة الله وبركاته is
# what a Saudi user expects, and rotating it away would read as rudeness.
_SALAM_RE = re.compile(r"^(?:و\s*عليكم\s*)?السلام(?:\s*عليكم)?\b|^سلام(?:ات)?\b")

_SALAM_REPLY_AR = (
    "وعليكم السلام ورحمة الله وبركاته! 🌟 يا أهلاً وسهلاً، نورت والله! "
    "اطرح سؤالك وأنا جاهز أجاوبك على طول ⚡"
)
_SALAM_REPLY_EN = (
    "Wa alaykum as-salam wa rahmatullahi wa barakatuh! 🌟 Welcome — "
    "go ahead and ask your question, I'm right here ⚡"
)

_AR_REPLIES = (
    "هلا والله وغلا! 😄 حياك الله وبياك، وجعل أيامك كلها خير وسعد. "
    "قول لي وش تبي تعرف وأنا أدور لك الجواب في لمح البصر!",
    "يا هلا والله بالغالي! 🌹 الله يقويك ويعطيك العافية. "
    "تفضل اسأل عن أي شي، وأنا في الخدمة على مدار الساعة ☕",
    "أهلاً وسهلاً ومرحبتين! 👋 كلك ذوق والله. "
    "اطرح سؤالك وخلنا نشوف وش عندنا لك من إجابات 🔎",
    "حياك الله! 😊 نورت المكان وزادت البهجة. "
    "أنا مساعدك للأسئلة الشائعة — اسأل وأنا أرد عليك فورًا بدون تأخير!",
)

_EN_REPLIES = (
    "Hey there! 👋 Great to see you. I'm your FAQ assistant — ask me anything "
    "and I'll dig up the answer in a flash ⚡",
    "Hello and welcome! 🌟 Hope your day is going brilliantly. "
    "Fire away with your question whenever you're ready 😊",
    "Hi! 😄 All good on my end and ready to help. "
    "What would you like to know?",
    "Greetings! 🌹 Lovely of you to drop by. Ask away — I'm right here.",
)


def is_greeting(text: str) -> bool:
    """Return whether ``text`` is *only* a greeting, with no question attached.

    Args:
        text: The raw user message.

    Returns:
        ``True`` when the entire message is greeting material and can be
        answered instantly; ``False`` for anything carrying a real request,
        including a greeting followed by a question.
    """
    if not text or not text.strip():
        return False
    cleaned = _STRIP_RE.sub(" ", normalise(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or len(cleaned.split()) > 8:
        return False
    return bool(_GREETING_ONLY_RE.match(cleaned))


def is_salam(text: str) -> bool:
    """Return whether ``text`` opens with the salam formula."""
    cleaned = _STRIP_RE.sub(" ", normalise(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return bool(_SALAM_RE.match(cleaned))


def greeting_reply(lang: Language | None = None, text: str = "") -> str:
    """Return a warm canned reply in the greeter's language.

    A salam gets its prescribed answer -- "وعليكم السلام ورحمة الله وبركاته" --
    every time. Any other greeting gets one of several replies at random, so
    repeat greetings in one session do not read like a stuck record.

    Args:
        lang: Language to reply in; detected from ``text`` when omitted.
        text: The original greeting, used to detect the language and the salam.

    Returns:
        A ready-to-send greeting.
    """
    lang = lang or detect_language(text, default="ar")
    if is_salam(text):
        return _SALAM_REPLY_AR if lang == "ar" else _SALAM_REPLY_EN
    return random.choice(_AR_REPLIES if lang == "ar" else _EN_REPLIES)
