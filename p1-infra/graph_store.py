from neo4j import GraphDatabase

class GraphStore:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="Tpassword123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upsert_source(self, external_ref: str, content_hash: str, ts: str) -> int:
        """Idempotent upsert. Returns the version number after this call.
        Version only increments if the content actually changed — this is
        what makes duplicate events a no-op instead of a bug."""
        query = """
        MERGE (s:SOURCE {external_ref: $external_ref})
        ON CREATE SET s.content_hash = $content_hash, s.version = 1, s.last_modified = $ts
        ON MATCH SET
            s.version = CASE WHEN s.content_hash <> $content_hash
                              THEN s.version + 1 ELSE s.version END,
            s.content_hash = $content_hash,
            s.last_modified = $ts
        RETURN s.version AS applied_version
        """
        with self.driver.session() as session:
            result = session.run(query, external_ref=external_ref,
                                  content_hash=content_hash, ts=ts)
            return result.single()["applied_version"]

    def upsert_chunks(self, source_external_ref: str, chunks: list[str]):
        query = """
        MERGE (s:SOURCE {external_ref: $source_ref})
        MERGE (d:DOCUMENT {external_ref: $source_ref})
        MERGE (s)-[:PARENT_OF]->(d)
        WITH d
        UNWIND range(0, size($chunks) - 1) AS idx
        MERGE (c:CHUNK {id: $source_ref + '-chunk-' + toString(idx)})
        SET c.text = $chunks[idx], c.stale = false
        MERGE (c)-[:DERIVED_FROM]->(d)
        """
        with self.driver.session() as session:
            session.run(query, source_ref=source_external_ref, chunks=chunks)

    def get_candidates(self, source_external_ref: str, hops: int = 2) -> list[str]:
        """Returns CHUNK ids within `hops` graph steps of the given SOURCE.
        This is what P3's dependency-tracing estimator calls directly."""
        query = f"""
        MATCH (s:SOURCE {{external_ref: $ref}})-[:PARENT_OF|DERIVED_FROM*1..{hops}]-(n:CHUNK)
        RETURN DISTINCT n.id AS artifact_id
        """
        with self.driver.session() as session:
            result = session.run(query, ref=source_external_ref)
            return [r["artifact_id"] for r in result]

    def apply_invalidate(self, artifact_id: str):
        query = "MATCH (c:CHUNK {id: $id}) SET c.stale = true RETURN c.id AS id"
        with self.driver.session() as session:
            result = session.run(query, id=artifact_id)
            record = result.single()
            return record["id"] if record else None

    def apply_retain(self, artifact_id: str):
        # No-op by design — logged, not silent, per the architecture spec's
        # requirement that "decided not to act" be distinguishable from
        # "forgot to act."
        return artifact_id