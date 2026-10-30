"""Shared by training and evaluation: model, sampling, prompt, image encoding, answer parsing, and the rewards."""
import base64, collections, io, os, random, re
from PIL import Image

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)      # Qwen3-VL recommended sampling
MAX_PIXELS = 512 * 512
# The official MMMU-Pro chain-of-thought suffix (MMMU-Benchmark/MMMU, mmmu-pro/prompts.yaml, cot.standard).
PROMPT_STANDARD = ("Answer the preceding multiple choice question. The last line of your response should be of the following "
                   "format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.")
LETTERS = "ABCDEFGHIJ"
ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)

def cap(im):
    """RGB, downscaled to at most MAX_PIXELS pixels (aspect ratio kept)."""
    im = im.convert("RGB"); w, h = im.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5; im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
    return im

def content(text, im):
    """OpenAI-style message content: the text first, then the image (MMMU-Pro order)."""
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]

def parse_answer(text, n_options):
    """The official MMMU-Pro format ('Answer: $LETTER'), else the last bare option letter; None if neither is valid."""
    valid = set(LETTERS[:n_options]); after = text.split("</think>")[-1]
    for c in reversed(ANS_RE.findall(after) or ANS_RE.findall(text)):
        if c.upper() in valid: return c.upper()
    for ch in reversed(re.findall(r"\b([A-J])\b", after)):
        if ch in valid: return ch
    return None

# ---------- rewards: one list of floats per group of G rollouts; an answer of None is a spoiled ballot ----------
def correctness(answers, gold):
    return [1.0 if a == gold else 0.0 for a in answers]

def collision(answers, gold, lam=1.0):
    """Collision RL: a parsed wrong answer pays lam * the share of the group (itself included) giving that answer."""
    G = len(answers); cnt = collections.Counter(answers)
    return [1.0 if a == gold else (-lam * cnt[a] / G if a is not None else 0.0) for a in answers]

def dfuser(answers, gold, rng):
    """D-FUSEr's diverse-failure reward (Baek et al., ICML 2026), our reimplementation: a parsed wrong answer that differs
    from the group's most common wrong answer gets 1/2 (ties broken at random); estimated online from the group."""
    r = correctness(answers, gold); wrong = collections.Counter(a for a in answers if a is not None and a != gold)
    if wrong:
        m = max(wrong.values()); modal = rng.choice(sorted(a for a, n in wrong.items() if n == m))
        r = [x + (0.5 if (a is not None and a != gold and a != modal) else 0.0) for x, a in zip(r, answers)]
    return r

def csr(answers, gold, lam=0.5, parsed_only=False):
    """CSR-style agreement reward (2605.15588) with an exact-match judge and a constant weight:
    R_j = c_j - lam * mean_{i != j} |1[a_j == a_i] - c_j|. With parsed_only=False (as run) spoiled peers count as votes."""
    G = len(answers); c = correctness(answers, gold); r = list(c)
    for j in range(G):
        if answers[j] is None: continue
        peers = [i for i in range(G) if i != j and (answers[i] is not None or not parsed_only)]
        if peers: r[j] -= lam * sum(abs(float(answers[j] == answers[i]) - c[j]) for i in peers) / len(peers)
    return r

def edas_adjust(adv, answers, gold, alpha=0.4, beta=0.2, kappa=2.0):
    """EDAS (2605.17333, official-code rule) applied to group advantages of the wrong parsed rollouts:
    one distinct wrong answer -> -beta * base; else alpha * base * (I_i - H) / ln N_w; |delta| <= |A| / kappa."""
    import math
    w = [i for i, x in enumerate(answers) if x is not None and x != gold]; N = len(w)
    if N <= 1: return adv
    cnt = collections.Counter(answers[i] for i in w); K = len(cnt); base = sum(abs(adv[i]) for i in w) / N
    if K == 1: d = {i: -beta * base for i in w}
    else:
        H = sum(-(k / N) * math.log(k / N) for k in cnt.values())
        d = {i: alpha * base * ((-math.log(cnt[answers[i]] / N)) - H) / math.log(N) for i in w}
    out = list(adv)
    for i, dd in d.items(): out[i] = adv[i] + math.copysign(min(abs(dd), abs(adv[i]) / kappa), dd)
    return out

def floor_spoiled(r, answers, floor):
    """Give spoiled ballots `floor`. Rewards that can go negative (collision, csr) need floor = -lambda, or an unfinished
    answer outscores a common wrong one and the policy is paid for not answering."""
    return [floor if a is None else x for x, a in zip(r, answers)]
