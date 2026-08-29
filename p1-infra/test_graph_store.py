from graph_store import GraphStore

store = GraphStore()

v1 = store.upsert_source("doc1.txt", "hash_aaa", "2026-08-24T10:00:00Z")
print("first call, expect 1:", v1)

v2 = store.upsert_source("doc1.txt", "hash_aaa", "2026-08-24T10:05:00Z")
print("same hash again, expect still 1:", v2)

v3 = store.upsert_source("doc1.txt", "hash_bbb", "2026-08-24T10:10:00Z")
print("changed hash, expect 2:", v3)

store.close()