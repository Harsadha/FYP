import json
import os
from kafka import KafkaConsumer
from graph_store import GraphStore
from chunker import chunk_text

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "p2_estimation", "corpus")

store = GraphStore()

consumer = KafkaConsumer(
    "kg.change-events",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="graph-updater",
)


def process_event(evt: dict):
    if evt["change_type"] == "DELETE":
        print(f"[graph-updater] DELETE for {evt['source_artifact_id']} — sprint scope: log only, no removal")
        return

    applied_version = store.upsert_source(
        evt["source_artifact_id"], evt["new_content_hash"], evt["detected_at"]
    )

    path = os.path.join(CORPUS_DIR, evt["source_artifact_id"])
    if not os.path.exists(path):
        print(f"[graph-updater] WARNING: {path} not found on disk, skipping chunking")
        return

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    store.upsert_chunks(evt["source_artifact_id"], chunks)
    print(f"[graph-updater] {evt['change_type']} {evt['source_artifact_id']} "
          f"-> version {applied_version}, {len(chunks)} chunks")


if __name__ == "__main__":
    print("graph-updater listening on kg.change-events...")
    for message in consumer:
        process_event(message.value)