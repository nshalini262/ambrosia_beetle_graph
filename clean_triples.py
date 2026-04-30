import json
import csv
import re

TRIPLES_FILE   = "triples.jsonl"
SPECIES_FILE   = "valid_species_names.csv"
SPECIES_COLUMN = "scientificName"
OUTPUT_FILE    = "filtered_triples.json"

print(f"Loading species names from {SPECIES_FILE} ...")

valid_species = set()
with open(SPECIES_FILE, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row[SPECIES_COLUMN].strip()
        if name:
            valid_species.add(name.lower())

print(f"Total valid species loaded: {len(valid_species)}")

# Build a single compiled regex — much faster than looping one by one
print("Compiling species regex ...")
species_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(valid_species, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
print("Regex ready.\n")

def contains_species(text: str) -> bool:
    return bool(species_pattern.search(text))

print(f"Filtering triples from {TRIPLES_FILE} ...")

total    = 0
matched  = 0
failed   = []
filtered = []

with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        total += 1

        try:
            triple  = json.loads(line)
            subject = str(triple.get("subject", ""))
            obj     = str(triple.get("object", ""))

            if contains_species(subject) or contains_species(obj):
                filtered.append(triple)
                matched += 1

        except json.JSONDecodeError as e:
            failed.append((line_num, str(e)))

        if total % 50000 == 0:
            print(f"  Processed {total:,} triples | matched so far: {matched:,}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"\n✓ Done!")
print(f"  Total triples scanned:  {total:,}")
print(f"  Triples matched:        {matched:,}")
print(f"  Triples rejected:       {total - matched:,}")
print(f"  Failed to parse:        {len(failed)}")
print(f"  Saved to:               {OUTPUT_FILE}")

if failed:
    print(f"\n  Failed lines: {failed[:10]}")
