from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

import certifi
from neo4j._sync.driver import Driver, GraphDatabase
from neo4j._conf import TrustCustomCAs

from config.settings import get_settings


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        if uri.startswith("neo4j+s://"):
            uri = uri.replace("neo4j+s://", "neo4j://", 1)
            self.driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                encrypted=True,
                trusted_certificates=TrustCustomCAs(certifi.where()),
            )
        else:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    @contextmanager
    def session(self) -> Iterator[Any]:
        with self.driver.session(database=self.database) as session:
            yield session

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.session() as session:
            return [record.data() for record in session.run(query, parameters or {})]

    def close(self) -> None:
        self.driver.close()


@lru_cache(maxsize=1)
def get_neo4j_client() -> Neo4jClient:
    settings = get_settings()
    if not settings.neo4j_password:
        raise RuntimeError("NEO4J_PASSWORD is required for Neo4j access")
    return Neo4jClient(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password, settings.neo4j_database)
