from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "kg.change-events",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
)
print("Listening on kg.change-events...")
for message in consumer:
    print(json.dumps(message.value, indent=2))