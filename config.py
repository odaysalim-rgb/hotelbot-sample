import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    # OpenAI configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Paths
    data_dir: str = os.getenv("DATA_DIR", "data")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")

    # RAG parameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    k: int = int(os.getenv("K", "4"))  # top-k documents to retrieve

    # MySQL / SQL settings
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_db: str = os.getenv("MYSQL_DB", "dubai_analytics")
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "basit456")
    mysql_table: str = os.getenv("MYSQL_TABLE", "dubai_hotels_daily")
    # Optional full SQLAlchemy URL; if empty, it will be built from the above pieces.
    mysql_url: str = os.getenv("MYSQL_URL", "mysql+pymysql://root:vRwmNEgcCupORQptMUSWEtpMzvGKAKnu@gondola.proxy.rlwy.net:12931/railway")


settings = Settings()


