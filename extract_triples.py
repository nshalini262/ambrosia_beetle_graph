import pickle
import json
from langchain_community.llms import Ollama

CHUNKS_FILE = "chunks.pkl"
OUTPUT_FILE = "triples.jsonl"

PROMPT_TEMPLATE = """
Extract subject–predicate–object triples from the text below.
Return ONLY a JSON list like:
[
  {{"subject": "...", "predicate": "...", "object": "..."}},
  ...
]

Prioritize:
- species
- fungi
- organisms
- locations
- habitats
- interactions
- attributes (size, color, shape)
- ecological relationships

Text:
{text}
"""

#initialize llama locally
llm = Ollama(model="llama3.2")

def extract_triples(text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    result = llm.invoke(prompt)
    try:
        triples = json.loads(result)
        return triples
    except json.JSONDecodeError:
        # fallback 
        return []

if __name__ == "__main__":
    chunks = pickle.load(open(CHUNKS_FILE, "rb"))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            # support LangChain Document objects
            text = chunk.page_content if hasattr(chunk, "page_content") else chunk
            triples = extract_triples(text)
            for t in triples:
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
            print(f"Processed chunk {i+1}/{len(chunks)}")
    print(f"Saved triples to {OUTPUT_FILE}")
