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
        return [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]


def chat(model, tok, messages, max_new_tokens=160):
    """Greedy single-turn completion of a chat message list."""
    import torch

    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
