from qdrant_client import QdrantClient

client = QdrantClient(
    host="127.0.0.1",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,
    check_compatibility=False,
    timeout=30,
)

info = client.get_collection("pdf_knowledge_base")
print("collection ok")

points, offset = client.scroll(
    collection_name="pdf_knowledge_base",
    limit=1,
    with_payload=True,
    with_vectors=False,
)

print("offset:", offset)
for p in points:
    print(p.payload)