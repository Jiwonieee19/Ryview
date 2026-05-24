from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="Fake Review Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD MODEL + TOKENIZER
# =========================================================

MODEL_PATH = Path(__file__).resolve().parent.parent / "roberta_fake_review_model"

tokenizer = RobertaTokenizer.from_pretrained(
    str(MODEL_PATH.resolve()),
    local_files_only=True
)
model = RobertaForSequenceClassification.from_pretrained(
    str(MODEL_PATH.resolve()),
    local_files_only=True
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
model.eval()

print(f"Model loaded on: {DEVICE}")

# =========================================================
# REQUEST SCHEMA
# =========================================================

class Review(BaseModel):
    text: str

# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
def home():
    return {"message": "Fake Review Detection API is running", "status": "ok"}

# =========================================================
# TOKEN ATTRIBUTION (Gradient x Input saliency)
# =========================================================

def get_token_attributions(text: str, pred_class: int):
    """
    Computes per-token importance using Gradient x Input saliency.
    Returns list of (word, score) pairs where score in [-1, 1].
    Positive score = pushed toward predicted class.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    embeddings = model.roberta.embeddings(inputs["input_ids"])
    embeddings.retain_grad()
    embeddings_input = embeddings.clone().detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=embeddings_input,
        attention_mask=inputs["attention_mask"]
    )

    score = outputs.logits[0, pred_class]
    model.zero_grad()
    score.backward()

    saliency = (embeddings_input.grad * embeddings_input).detach()
    token_scores = saliency.sum(dim=-1).squeeze(0).cpu().numpy()

    max_abs = np.abs(token_scores).max()
    if max_abs > 0:
        token_scores = token_scores / max_abs

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze(0).cpu().tolist()
    )

    words = []
    skip = {"<s>", "</s>", "<pad>"}

    for token, score in zip(tokens, token_scores):
        if token in skip:
            continue
        if token.startswith("Ġ") or not words:
            words.append({
                "word":  token.replace("Ġ", ""),
                "score": float(score)
            })
        else:
            words[-1]["word"]  += token
            words[-1]["score"] += float(score)

    max_abs = max(abs(w["score"]) for w in words) if words else 1.0
    if max_abs > 0:
        for w in words:
            w["score"] = round(w["score"] / max_abs, 4)

    # Keep the original word (with punctuation) for the highlight view,
    # and add a clean display_word (sentence punctuation stripped, % preserved)
    # for use in pills and bar charts.
    _STRIP_CHARS = ".!?,;:'\"-"
    for w in words:
        w["display_word"] = w["word"].strip(_STRIP_CHARS)

    return words



# =========================================================
# SENTENCE-LEVEL SCORING
# =========================================================

def get_sentence_scores(text: str, pred_class: int, base_conf: float):
    """
    Scores each sentence by running the model with that sentence masked out.
    score = base_conf - conf_without_sentence
    Positive  -> sentence supports the verdict (removing it hurts confidence).
    Negative  -> sentence works against the verdict (removing it helps).
    Normalized to [-1, 1].
    """
    import re as _re

    raw_sentences = _re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if len(sentences) <= 1:
        return [{"sentence": text.strip(), "score": 1.0}]

    raw_scores = []
    for i, sent in enumerate(sentences):
        remaining = " ".join(s for j, s in enumerate(sentences) if j != i)
        if not remaining.strip():
            raw_scores.append(0.0)
            continue
        inputs = tokenizer(
            remaining,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs    = F.softmax(outputs.logits, dim=1)
        new_conf = float(probs[0, pred_class].item())
        raw_scores.append(base_conf - new_conf)

    max_abs = max(abs(s) for s in raw_scores) if raw_scores else 1.0
    if max_abs > 0:
        raw_scores = [round(s / max_abs, 4) for s in raw_scores]

    return [{"sentence": sent, "score": sc} for sent, sc in zip(sentences, raw_scores)]

# =========================================================
# PREDICT ROUTE
# =========================================================

@app.post("/predict")
def predict(review: Review):
    inputs = tokenizer(
        review.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs                = F.softmax(outputs.logits, dim=1)
    confidence, pred_cls = torch.max(probs, dim=1)
    pred_class           = int(pred_cls.item())

    label_map = {0: "Genuine", 1: "Deceptive"}
    label     = label_map[pred_class]

    token_scores    = get_token_attributions(review.text, pred_class)
    base_conf       = float(confidence.item())
    sentence_scores = get_sentence_scores(review.text, pred_class, base_conf)

    return {
        "label":           label,
        "confidence":      base_conf,
        "token_scores":    token_scores,
        "sentence_scores": sentence_scores
    }


# =========================================================
# DASHBOARD ROUTE
# =========================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    label:        str = "",
    confidence:   str = "",
    text:         str = "",
    token_scores: str = ""
):
    import json, urllib.parse, re

    is_deceptive  = label == "Deceptive"
    verdict_color = "#F0527A" if is_deceptive else "#52C9A0"
    verdict_bg    = "rgba(240,82,122,0.10)" if is_deceptive else "rgba(82,201,160,0.10)"
    verdict_icon  = "&#9888;" if is_deceptive else "&#10003;"
    conf_display  = f"{confidence}%" if confidence else "&#8212;"
    against_color = "#52C9A0" if is_deceptive else "#F0527A"

    tokens_data = []
    if token_scores:
        try:
            tokens_data = json.loads(urllib.parse.unquote(token_scores))
        except Exception:
            tokens_data = []

    # Compute sentence scores server-side from text + parsed confidence
    sentences_data = []
    if text:
        try:
            pred_class_dash = 0 if label == "Genuine" else 1
            base_conf_dash  = float(str(confidence).strip().rstrip("%"))
            base_conf_dash  = base_conf_dash / 100.0 if base_conf_dash > 1.0 else base_conf_dash
            sentences_data  = get_sentence_scores(text, pred_class_dash, base_conf_dash)
        except Exception:
            sentences_data = []

    # ── Confidence ring offset ────────────────────────────────────────────────
    try:
        conf_val    = float(confidence) if confidence else 0.0
        conf_offset = 251.2 * (1 - conf_val / 100)
    except Exception:
        conf_offset = 251.2

    # Review text in the first card is now plain text only.
    highlighted_html = ""
    if text:
      highlighted_html = text.replace("<", "&lt;").replace(">", "&gt;")

    # Token sensitivity map removed — sentence-level scoring is the primary explanation

    # ── Visual explain panel ──────────────────────────────────────────────────

    # 1. Tilt meter
    tilt_html = ""
    try:
        pos_scores  = [w["score"] for w in tokens_data if w["score"] > 0]
        neg_scores  = [abs(w["score"]) for w in tokens_data if w["score"] < 0]
        support_sum = sum(pos_scores)
        oppose_sum  = sum(neg_scores)
        net         = support_sum - oppose_sum
        denom       = support_sum + oppose_sum
        tilt_pct    = (net / denom * 100.0) if denom > 0 else 0.0

        # Fix 1: tilt direction must reflect which side actually dominates.
        # net > 0  → supporting (toward verdict) wins  → bar grows rightward
        # net < 0  → opposing (against verdict) wins   → bar grows leftward
        tilt_side  = "right" if net >= 0 else "left"
        tilt_width = min(abs(tilt_pct) / 2, 50)
        tilt_color = verdict_color if net >= 0 else against_color

        if net >= 0:
            tilt_summary = f"+{tilt_pct:.1f}% net — token weight favours <strong>{label}</strong>"
        else:
            tilt_summary = f"{tilt_pct:.1f}% net — token weight leans <strong>against {label}</strong> (model still confident due to global context)"

        tilt_html = f"""
        <div class="viz-section">
          <div class="viz-label">Overall token influence balance</div>
          <div class="viz-sublabel">How much token weight supports vs. opposes the verdict. A split where individual tokens lean against the verdict while overall confidence is high means the model is relying on global context rather than single words.</div>
          <div class="tilt-label-row">
            <span style="color:{against_color};">&#9650; Against {label}</span>
            <span style="color:{verdict_color};">&#9650; Toward {label}</span>
          </div>
          <div class="tilt-track">
            <div class="tilt-center-line"></div>
            <div class="tilt-fill {tilt_side}" style="width:{tilt_width:.1f}%;background:{tilt_color};"></div>
          </div>
          <div class="tilt-footer">
            <span>Opposing: {oppose_sum:.2f}</span>
            <span>Supporting: {support_sum:.2f}</span>
          </div>
          <div class="tilt-summary" style="color:{tilt_color};">{tilt_summary}</div>
        </div>"""
    except Exception:
        tilt_html = ""

    # 2. Sentence bar chart
    bar_html = ""
    strongest_sentence = None
    try:
        sorted_sents = sorted(sentences_data, key=lambda x: abs(x["score"]), reverse=True)

        rows = ""
        for item in sorted_sents:
            score     = item["score"]
            sent      = item["sentence"].replace("<", "&lt;").replace(">", "&gt;")
            # Truncate long sentences for the label
            label_txt = sent if len(sent) <= 72 else sent[:69] + "…"
            width_pct = min(abs(score) * 100, 100)
            bar_color = verdict_color if score > 0 else against_color
            dir_color = bar_color
            direction = "&#8593; supports verdict" if score > 0 else "&#8595; weakens verdict"
            sign      = "+" if score > 0 else "&#8722;"
            rows += f"""
            <div class="sent-row">
              <div class="sent-label" title="{sent}">{label_txt}</div>
              <div class="sent-bar-wrap">
                <div class="bar-track">
                  <div class="bar-fill" style="width:{width_pct:.1f}%;background:{bar_color};">
                    <span class="bar-val">{sign}{abs(score):.2f}</span>
                  </div>
                </div>
                <span class="bar-dir" style="color:{dir_color};">{direction}</span>
              </div>
            </div>"""

        bar_html = f"""
        <div class="sent-chart">{rows}</div>"""
        strongest_sentence = sorted_sents[0] if sorted_sents else None
    except Exception:
        bar_html = ""
        strongest_sentence = None

    cf_html = ""

    # Sentence card — standalone card separate from explain card
    if bar_html:
        sent_card = (
            '<div class="sent-card">'
            '<p class="sec-title">Sentence-level influence</p>'
            '<p class="sent-sublabel">'
            'Each sentence scored by how much removing it shifts model confidence. '
            '<span style="color:' + verdict_color + ';">&#9632; supports ' + label + '</span> &nbsp; '
            '<span style="color:' + against_color + ';">&#9632; weakens ' + label + '</span>'
            '</p>'
            + bar_html +
            '</div>'
        )
    else:
        sent_card = ""

    # Same-card explanation for why the model leaned genuine or deceptive.
    why_card = ""
    if label:
      if strongest_sentence:
        strongest_text = strongest_sentence["sentence"].replace("<", "&lt;").replace(">", "&gt;")
        strongest_score = strongest_sentence["score"]
        strongest_dir = "toward" if strongest_score > 0 else "against"
        strongest_label = "supports" if strongest_score > 0 else "pushes against"
        strongest_excerpt = strongest_text if len(strongest_text) <= 96 else strongest_text[:93] + "…"

        why_title = "Why the model labeled it this way"
        why_intro = (
            f"Sentence-level influence: <strong>{strongest_excerpt}</strong> is the strongest shift the model found, scoring {strongest_score:+.2f} and {strongest_dir} the {label.lower()} verdict."
        )
        why_detail = (
            f"Model confidence breakdown: the final confidence is {conf_display}, so the label is not coming from one sentence alone. "
            f"The sentence bars and confidence ring together show how the model combines the strongest evidence before choosing {label.lower()}."
        )
        why_tail = (
            f"Why this label: the model is treating the review as {label.lower()} because its strongest sentence-level signal {strongest_label} that verdict more than the opposite one."
        )
      else:
        why_title = "Why the model labeled it this way"
        why_intro = "Sentence-level influence: the model produced a label, but no sentence breakdown was available to display here."
        why_detail = f"Model confidence breakdown: the final confidence is {conf_display}, and the label still comes from the model’s internal sentence scoring."
        why_tail = f"Why this label: the prediction reflects the model output for {label.lower()}, even when the sentence cards could not be rendered."

      why_card = (
        '<div class="sent-card">'
        f'<p class="sec-title">{why_title}</p>'
        '<p class="sent-sublabel" style="margin-bottom:10px;">'
        + why_intro +
        '</p>'
        '<p class="sent-sublabel" style="margin-bottom:10px;">'
        + why_detail +
        '</p>'
        '<p class="sent-sublabel" style="margin-bottom:0;">'
        + why_tail +
        '</p>'
        '</div>'
      )

    # ── Pre-build conditional blocks ──────────────────────────────────────────
    if highlighted_html:
        rc = (
            '<div class="review-card">'
        '<p class="sec-title">Review Text</p>'
            '<p style="margin-top:-4px;margin-bottom:12px;color:#7A7290;font-size:12px;line-height:1.5;">'
        'Plain review text shown without highlight colours.'
            '</p>'
        '<div class="review-text">' + highlighted_html + '</div>'
        '</div>'
        )
    else:
        rc = ""

    wc = ""  # Token sensitivity map removed

    # ── Full HTML ─────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RyView &#8212; Analysis Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{
      --bg:      #110D1A;
      --s1:      #1C1628;
      --s2:      #241E33;
      --border:  rgba(255,255,255,0.08);
      --text:    #EAE6F0;
      --muted:   #7A7290;
      --accent:  #9B6DFF;
      --verdict: {verdict_color};
      --against: {against_color};
      --vbg:     {verdict_bg};
      --font-h:  'Syne', sans-serif;
      --font-b:  'DM Sans', sans-serif;
    }}
    html{{scroll-behavior:smooth}}
    body{{
      background:var(--bg);color:var(--text);
      font-family:var(--font-b);font-size:15px;
      line-height:1.6;min-height:100vh;
    }}
    .blob{{
      position:fixed;border-radius:50%;
      filter:blur(130px);pointer-events:none;z-index:0;
    }}
    .blob-1{{
      width:550px;height:550px;
      background:{"rgba(240,82,122,0.07)" if is_deceptive else "rgba(82,201,160,0.06)"};
      top:-180px;right:-180px;
      animation:drift 14s ease-in-out infinite alternate;
    }}
    .blob-2{{
      width:450px;height:450px;
      background:rgba(155,109,255,0.06);
      bottom:-120px;left:-120px;
      animation:drift 14s ease-in-out infinite alternate;
      animation-delay:-7s;
    }}
    @keyframes drift{{
      from{{transform:translate(0,0) scale(1)}}
      to{{transform:translate(28px,18px) scale(1.05)}}
    }}
    .page{{
      position:relative;z-index:1;
      max-width:800px;margin:0 auto;
      padding:44px 24px 80px;
    }}
    .header{{
      display:flex;align-items:center;
      justify-content:space-between;
      margin-bottom:40px;
      animation:fadeUp .4s ease both;
    }}
    .brand{{display:flex;align-items:center;gap:10px;}}
    .brand-icon{{
      width:36px;height:36px;border-radius:10px;
      background:linear-gradient(135deg,var(--accent),#7C3AED);
      display:flex;align-items:center;justify-content:center;
      font-size:18px;font-weight:800;color:white;
      font-family:var(--font-h);
    }}
    .brand-name{{font-family:var(--font-h);font-size:18px;font-weight:800;}}
    .status-pill{{
      display:flex;align-items:center;gap:6px;
      padding:6px 14px;border-radius:999px;
      border:1px solid var(--border);
      background:var(--s1);
      font-size:12px;color:var(--muted);
    }}
    .pulse{{
      width:7px;height:7px;border-radius:50%;
      background:#52C9A0;box-shadow:0 0 6px #52C9A0;
      animation:pulse 2s infinite;
    }}
    @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(1.4)}}}}

    /* ── Verdict card ── */
    .verdict-card{{
      background:var(--s1);
      border:1px solid var(--border);
      border-radius:24px;
      padding:32px 36px;
      margin-bottom:16px;
      position:relative;overflow:hidden;
      animation:fadeUp .4s .08s ease both;
    }}
    .verdict-card::before{{
      content:'';
      position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,var(--verdict),transparent);
    }}
    .verdict-top{{
      display:flex;align-items:center;
      justify-content:space-between;
      margin-bottom:22px;
    }}
    .eyebrow{{
      font-size:11px;font-weight:500;
      letter-spacing:.1em;text-transform:uppercase;
      color:var(--muted);
    }}
    .verdict-badge{{
      display:flex;align-items:center;gap:7px;
      background:var(--vbg);
      border:1px solid var(--verdict);
      border-radius:999px;padding:5px 15px;
      font-family:var(--font-h);font-size:13px;font-weight:700;
      color:var(--verdict);
    }}
    .verdict-main{{
      display:flex;align-items:flex-end;
      justify-content:space-between;gap:16px;
    }}
    .verdict-label{{
      font-family:var(--font-h);
      font-size:52px;font-weight:800;
      line-height:1;letter-spacing:-2px;
      color:var(--verdict);
    }}
    .verdict-desc{{
      font-size:13px;color:var(--muted);
      margin-top:8px;max-width:340px;line-height:1.6;
    }}
    .conf-ring{{flex-shrink:0;position:relative;width:96px;height:96px;}}
    .conf-ring svg{{transform:rotate(-90deg);}}
    .ring-track{{fill:none;stroke:var(--s2);stroke-width:8;}}
    .ring-fill{{
      fill:none;stroke:var(--verdict);
      stroke-width:8;stroke-linecap:round;
      stroke-dasharray:251.2;
      stroke-dashoffset:{conf_offset};
    }}
    .conf-center{{
      position:absolute;inset:0;
      display:flex;flex-direction:column;
      align-items:center;justify-content:center;
    }}
    .conf-val{{font-family:var(--font-h);font-size:12px;font-weight:800;line-height:1;}}
    .conf-lbl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}

    /* ── Review card ── */
    .sec-title{{
      font-size:11px;font-weight:600;
      letter-spacing:.1em;text-transform:uppercase;
      color:var(--muted);margin-bottom:12px;
    }}
    .review-card{{
      background:var(--s1);border:1px solid var(--border);
      border-radius:20px;padding:24px 28px;
      margin-bottom:16px;
      animation:fadeUp .4s .16s ease both;
    }}
    .review-text{{
      font-size:14.5px;line-height:1.85;
      color:var(--text);word-break:break-word;
    }}
    .legend-row{{
      display:flex;align-items:center;gap:16px;
      margin-top:14px;padding-top:12px;
      border-top:1px solid var(--border);
      font-size:12px;color:var(--muted);
    }}
    .legend-dot{{
      width:10px;height:10px;border-radius:3px;
      display:inline-block;margin-right:5px;
    }}

    /* ── Sentence card ── */
    .sent-card{{
      background:var(--s1);border:1px solid var(--border);
      border-radius:20px;padding:24px 28px;
      margin-bottom:16px;
      animation:fadeUp .4s .25s ease both;
    }}
    .sent-sublabel{{
      font-size:12px;color:var(--muted);
      margin-bottom:16px;line-height:1.5;
    }}


    /* ── Visual explain card ── */
    .explain-card{{
      background:var(--s1);border:1px solid var(--border);
      border-radius:20px;padding:24px 28px;
      margin-bottom:16px;
      animation:fadeUp .4s .28s ease both;
    }}
    .explain-title{{
      font-family:var(--font-h);
      font-size:16px;font-weight:700;
      color:var(--text);margin-bottom:20px;
    }}
    .section-divider{{
      border:none;border-top:1px solid var(--border);
      margin:22px 0;
    }}

    /* Shared viz section */
    .viz-section{{margin-bottom:4px;}}
    .viz-label{{
      font-size:13px;font-weight:600;
      color:var(--text);margin-bottom:4px;
    }}
    .viz-sublabel{{
      font-size:11px;color:var(--muted);
      margin-bottom:12px;line-height:1.5;
    }}

    /* Tilt meter */
    .tilt-label-row{{
      display:flex;justify-content:space-between;
      font-size:11px;font-weight:600;
      margin-bottom:6px;
    }}
    .tilt-track{{
      height:14px;border-radius:7px;
      background:var(--s2);
      position:relative;overflow:hidden;
    }}
    .tilt-center-line{{
      position:absolute;left:50%;top:0;bottom:0;
      width:1px;background:rgba(255,255,255,0.15);
      z-index:1;
    }}
    .tilt-fill{{
      position:absolute;top:0;bottom:0;border-radius:7px;
    }}
    .tilt-fill.right{{left:50%;}}
    .tilt-fill.left {{right:50%;}}
    .tilt-footer{{
      display:flex;justify-content:space-between;
      font-size:11px;color:var(--muted);
      margin-top:6px;
    }}
    .tilt-summary{{
      font-size:12px;font-weight:600;
      margin-top:8px;line-height:1.5;
    }}

    /* Sentence bar chart */
    .sent-chart{{display:flex;flex-direction:column;gap:12px;}}
    .sent-row{{display:flex;flex-direction:column;gap:5px;}}
    .sent-label{{
      font-size:12.5px;color:var(--text);line-height:1.4;
      word-break:break-word;
    }}
    .sent-bar-wrap{{display:flex;align-items:center;gap:10px;}}
    .bar-track{{
      flex:1;height:22px;
      background:var(--s2);border-radius:5px;
      position:relative;overflow:hidden;
    }}
    .bar-fill{{
      height:100%;border-radius:5px;
      display:flex;align-items:center;padding-left:8px;
      min-width:32px;
    }}
    .bar-val{{font-size:10px;font-weight:700;color:rgba(255,255,255,0.9);white-space:nowrap;}}
    .bar-dir{{font-size:11px;min-width:110px;white-space:nowrap;}}

    .disclaimer{{
      margin-top:16px;font-size:11px;
      color:var(--muted);line-height:1.6;
    }}

    /* ── Footer ── */
    .footer{{
      text-align:center;font-size:12px;color:var(--muted);
      margin-top:44px;animation:fadeUp .4s .34s ease both;
    }}

    @keyframes fadeUp{{
      from{{opacity:0;transform:translateY(14px)}}
      to{{opacity:1;transform:translateY(0)}}
    }}
    @media(max-width:560px){{
      .verdict-label{{font-size:36px;}}
      .verdict-card{{padding:22px 18px;}}
      .bar-label{{min-width:70px;max-width:70px;}}
      .cf-panel{{flex-direction:column;}}
    }}
  </style>
</head>
<body>
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="page">

    <header class="header">
      <div class="brand">
        <div class="brand-icon">&#119825;</div>
        <div class="brand-name">RyView</div>
      </div>
      <div class="status-pill">
        <div class="pulse"></div>
        Analysis Complete
      </div>
    </header>

    <!-- Verdict card -->
    <div class="verdict-card">
      <div class="verdict-top">
        <div class="eyebrow">Analysis Verdict</div>
        <div class="verdict-badge">{verdict_icon}&nbsp;{label if label else "&#8212;"}</div>
      </div>
      <div class="verdict-main">
        <div>
          <div class="verdict-label">{label if label else "&#8212;"}</div>
          <div class="verdict-desc">
            {"This review contains language patterns the model associates with deceptive content. The sections below explain what influenced this decision." if is_deceptive else "This review contains language patterns the model associates with genuine content. The sections below explain what influenced this decision."}
          </div>
        </div>
        <div class="conf-ring">
          <svg width="96" height="96" viewBox="0 0 100 100">
            <circle class="ring-track" cx="50" cy="50" r="40"/>
            <circle class="ring-fill"  cx="50" cy="50" r="40"/>
          </svg>
          <div class="conf-center">
            <div class="conf-val">{conf_display}</div>
            <div class="conf-lbl">Confidence</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Sentence-highlighted review text -->
    {rc}

    <!-- Sentence influence card (primary explanation) -->
    {sent_card}

    <!-- Why this verdict card -->
    {why_card}

    <!-- Tilt meter + methodology note -->
    <div class="explain-card">
      <div class="explain-title">Model confidence breakdown</div>

      {tilt_html}

      <p class="disclaimer">
        <em>How to read this:</em> The sentence highlight and sentence-level influence bars are the primary explanation &#8212;
        they show which parts of the review actually shifted the model&#8217;s confidence.
        The token sensitivity map is a secondary signal showing which individual words the model is most sensitive to,
        not necessarily what drove the final verdict.
        All scores are approximate &#8212; interpret alongside human judgement.
        Model: RoBERTa fine-tuned on deceptive hotel reviews.
      </p>
    </div>

    <div class="footer">
      Powered by <strong>RyView</strong> &nbsp;&middot;&nbsp;
      ReImagined RoBERTa for Deceptive Hotel Review Detection
    </div>

  </div>
</body>
</html>"""