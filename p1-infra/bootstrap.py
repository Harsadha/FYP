import json
import os
import time
from datetime import datetime, timezone
from kafka import KafkaProducer
from change_watcher import sha256_of_file, build_event

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "p2_estimation", "corpus")

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

if __name__ == "__main__":
    files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    print(f"Bootstrapping {len(files)} documents...")
    for fname in files:
        path = os.path.join(CORPUS_DIR, fname)
        evt = build_event(fname, "CREATE", sha256_of_file(path))
        producer.send("kg.change-events", evt)
    producer.flush()
    print("Bootstrap events published. Make sure graph_updater.py is running to consume them.")