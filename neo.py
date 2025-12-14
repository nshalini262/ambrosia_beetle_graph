from neo4j import GraphDatabase
import json

NEO4J_URI = 
NEO4J_USER = 
NEO4J_PASSWORD =   

TRIPLES_FILE = "triples.jsonl"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_constraints(session):
    session.run("""
        CREATE CONSTRAINT IF NOT EXISTS FOR (s:Entity)
        REQUIRE s.name IS UNIQUE
    """)

def insert_triple(session, subj, pred, obj):
    session.run("""
        MERGE (a:Entity {name: $s})
        MERGE (b:Entity {name: $o})
        MERGE (a)-[r:RELATION {type: $p}]->(b)
    """, s=subj, p=pred, o=obj)

def load_triples():
    with driver.session(database="neo4j") as session:
        print("Creating uniqueness constraint...")
        create_constraints(session)

        print("Loading triples from JSONL...")
        with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                subj = t.get("subject")
                pred = t.get("predicate")
                obj = t.get("object")
                if subj and pred and obj:
                    insert_triple(session, subj, pred, obj)

        print("Done inserting triples.")

if __name__ == "__main__":
    load_triples()
    print("Graph import complete.")
