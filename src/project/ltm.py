import psycopg
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URI")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def _as_pgvector(vector: list[float]) -> str:
    """Format an embedding so psycopg can pass it to pgvector reliably."""
    return "[" + ",".join(str(value) for value in vector) + "]"


def create_memory_table():

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id SERIAL PRIMARY KEY,
                    memory TEXT NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        conn.commit()



def save_memory(memory: str):

    vector = _as_pgvector(embeddings.embed_query(memory))

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO long_term_memory
                (memory, embedding)
                VALUES (%s, %s::vector)
                """,
                (memory, vector)
            )

        conn.commit()



def retrieve_memories(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.55
):

    query_vector = _as_pgvector(embeddings.embed_query(query))

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    memory,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM long_term_memory
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_vector,
                    query_vector,
                    min_similarity,
                    query_vector,
                    limit
                )
            )

            rows = cur.fetchall()

    return [memory for memory, _similarity in rows]
