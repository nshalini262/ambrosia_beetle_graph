import pickle
import json
import re
import os
import sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

CHUNKS_FILE   = "chunks.pkl"
OUTPUT_FILE   = "triples.jsonl"
RESUME_FILE   = "resume_index.txt"      #last completed chunk to resume b/w runs

#hugging face model
MODEL_NAME    = "mistralai/Mistral-7B-Instruct-v0.2"

MAX_NEW_TOKENS = 512
PROMPT_CHAR_LIMIT = 2500        

# GPU check
if not torch.cuda.is_available():
    print("WARNING: No GPU detected — running on CPU (will be slow).")
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# Load model
print(f"\nLoading model: {MODEL_NAME} …")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,      # half-precision → halves VRAM usage
    device_map="auto",              # automatically spreads across available GPUs
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,                # greedy decoding → consistent JSON output
    temperature=None,               # must be None when do_sample=False
    top_p=None,
    return_full_text=False,         # return only the generated part
)

print("Model loaded.\n")



SYSTEM_MSG = (
    "You are a scientific knowledge-graph builder specialising in forest entomology. "
    "Your ONLY job is to output a valid JSON array of triples. "
    "Do NOT write any explanation, preamble, or markdown. "
    "Each triple must have exactly three keys: \"subject\", \"predicate\", \"object\"."
)


#optimizing prompt for mistral
def build_prompt(text: str) -> str:
    """Build a chat-style prompt using Mistral's [INST] format."""
    user_msg = (
        "Extract subject–predicate–object triples from the scientific text below.\n"
        "Focus on: beetle species, fungi species, host trees, locations, "
        "ecological interactions (e.g. infects, inhabits, found_in, associated_with).\n\n"
        f"Text:\n{text[:PROMPT_CHAR_LIMIT]}\n\n"
        "Return ONLY a JSON array, e.g.:\n"
        '[{"subject":"Ips typographus","predicate":"infects","object":"Picea abies"}]'
    )
    # mistral template
    return f"[INST] {SYSTEM_MSG}\n\n{user_msg} [/INST]"


#JSON extraction
def extract_json(raw: str) -> list[dict]:
    """Try several strategies to pull a valid JSON array from messy LLM output."""

    # 1) Direct parse (ideal case)
    stripped = raw.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2) Find the first '[' and last ']' and try parsing that slice
    start = stripped.find("[")
    end   = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 3) Strip markdown code fences (```json … ```) then retry
    no_fence = re.sub(r"```(?:json)?|```", "", stripped).strip()
    try:
        result = json.loads(no_fence)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 4) Extract individual {...} objects as a fallback
    objects = re.findall(r"\{[^{}]+\}", stripped, re.DOTALL)
    triples = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if {"subject", "predicate", "object"}.issubset(obj.keys()):
                triples.append(obj)
        except json.JSONDecodeError:
            continue
    return triples

#extraction function
def extract_triples(text: str) -> list[dict]:
    prompt = build_prompt(text)
    try:
        output = pipe(prompt)[0]["generated_text"]
        triples = extract_json(output)
        return triples
    except Exception as e:
        print(f"  [extraction error] {e}")
        return []


# Resume helper
def load_resume_index() -> int:
    if os.path.isfile(RESUME_FILE):
        with open(RESUME_FILE) as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0

def save_resume_index(idx: int):
    with open(RESUME_FILE, "w") as f:
        f.write(str(idx))



if __name__ == "__main__":

    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE} …")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    total = len(chunks)
    print(f"Total chunks: {total}")

    resume_from = load_resume_index()
    if resume_from > 0:
        print(f"Resuming from chunk {resume_from} (delete '{RESUME_FILE}' to restart).\n")

    total_triples = 0
    failed_chunks = []

    # Open output file in append mode so progress is never lost
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for idx in range(resume_from, total):
            chunk = chunks[idx]

            # Support both LangChain Document objects and plain strings
            text = getattr(chunk, "page_content", None) or str(chunk)

            triples = extract_triples(text)

            # Write each triple as a separate JSON line
            # NEW
            for t in triples:
                if not isinstance(t, dict):  # skip anything that isn't a proper triple
                    continue
                t["chunk_index"] = idx
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
           
            out.flush()
            os.fsync(out.fileno())   # flush OS buffer → guarantees disk write

            total_triples += len(triples)
            save_resume_index(idx + 1)

            # Progress report
            pct = (idx + 1) / total * 100
            print(f"[{idx + 1:>6}/{total}] ({pct:5.1f}%)  triples this chunk: {len(triples):3d}  |  total so far: {total_triples}")

            # warn when triples are empty 
            if len(triples) == 0:
                failed_chunks.append(idx)

    print(f"\n✓ Done. {total_triples} triples written to '{OUTPUT_FILE}'.")
    if failed_chunks:
        print(f"  Chunks with 0 triples extracted ({len(failed_chunks)}): {failed_chunks[:20]}{'…' if len(failed_chunks) > 20 else ''}")
       
