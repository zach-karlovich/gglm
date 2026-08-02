"""Load the generator and answer questions, with or without retrieved sources.
Greedy decoding throughout, so runs repeat exactly."""

GENERATOR = "Qwen/Qwen2.5-7B-Instruct" # picked in check-in 3

SYSTEM = (
    "You are a light gas gun and hypervelocity impact researcher and engineer. "
    "Answer the question concisely, in one sentence or less. "
    "If you do not know the answer, say so."
)

RAG_SYSTEM = (
    "You are a light gas gun and hypervelocity impact assistant. "
    "Answer the question concisely using only the numbered sources. "
    "Cite the sources you use by number. "
    "If they do not contain the answer, say so."
)

# two demos ahead of the real question: answer in one cited sentence, and
# refuse when the sources don't have it. Check-in 4 showed the model is
# right but wordy (hit@5 0.76 vs substring 0.25) and never refuses on its
# own (squadv2 NoAns 0.0). Demo text comes from the check-in 3 dev pairs,
# never the test split.
DEMOS = [
    (
        "[1] Two-stage light gas guns: A two-stage light gas gun has two separate "
        "firing stages. The first stage can be driven by powder or gas, and the "
        "second stage is typically gas.",
        "What can drive the first stage?",
        "Powder or gas [1].",
    ),
    (
        "[1] Sabot design: A sabot carries the projectile through the launch tube "
        "and is designed to separate from it after the projectile exits the muzzle.",
        "What alloy is the rupture diaphragm made of?",
        "The sources do not contain this.",
    ),
]


def load_generator(model_id=GENERATOR):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tok


def build_messages(question, contexts=None):
    if contexts:
        sources = "\n\n".join(
            f"[{n}] {c['title']}: {c['text']}" for n, c in enumerate(contexts, 1)
        )
        msgs = [{"role": "system", "content": RAG_SYSTEM}]
        for demo_src, demo_q, demo_a in DEMOS:
            msgs.append({"role": "user", "content": f"Sources:\n{demo_src}\n\nQuestion: {demo_q}"})
            msgs.append({"role": "assistant", "content": demo_a})
        msgs.append({"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"})
        return msgs
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]


def render(tok, messages):
    """Apply the chat template, folding the system turn into the first user
    turn for templates that reject system roles (Mistral)."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception: # jinja: roles must alternate user/assistant
        merged = dict(
            messages[1], content=messages[0]["content"] + "\n\n" + messages[1]["content"]
        )
        return tok.apply_chat_template(
            [merged] + messages[2:], tokenize=False, add_generation_prompt=True
        )


def chat(model, tok, messages, max_new_tokens=160):
    """Greedy single-turn completion of a chat message list."""
    import torch

    text = render(tok, messages)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def answer(model, tok, question, contexts=None, max_new_tokens=160): # short answers, the golds are one sentence
    return chat(model, tok, build_messages(question, contexts), max_new_tokens)
