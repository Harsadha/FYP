import hashlib
import json
import os
import time
from datetime import datetime, timezone

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from kafka import KafkaProducer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "p2_estimation", "corpus")
TOPIC = "kg.change-events"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_event(artifact_id: str, change_type: str, content_hash: str | None) -> dict:
    return {
        "event_id": f"{artifact_id}-{int(time.time() * 1000)}",
        "source_artifact_id": artifact_id,
        "change_type": change_type,
        "new_content_hash": content_hash,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


class CorpusHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        self._emit(event.src_path, "CREATE")

    def on_modified(self, event):
        if event.is_directory:
            return
        self._emit(event.src_path, "UPDATE")

    def on_deleted(self, event):
        if event.is_directory:
            return
        self._emit(event.src_path, "DELETE", skip_hash=True)

    def _emit(self, path: str, change_type: str, skip_hash: bool = False):
        artifact_id = os.path.basename(path)
        content_hash = None if skip_hash else sha256_of_file(path)
        evt = build_event(artifact_id, change_type, content_hash)
        producer.send(TOPIC, evt)
        producer.flush()
        print(f"[watcher] published {change_type} for {artifact_id}")


if __name__ == "__main__":
    observer = Observer()
    observer.schedule(CorpusHandler(), CORPUS_DIR, recursive=False)
    observer.start()
    print(f"Watching {CORPUS_DIR} ... press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()