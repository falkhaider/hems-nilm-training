from __future__ import annotations

# Zweck: PostgreSQL-Adapter für DEDDIAG -> pandas.DataFrame

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import psycopg
from psycopg import sql


@dataclass
class DeddiagPgConfig:
    # Konfiguration DB-Zugangsdaten
    db: "DbConfigLike"


@dataclass
class DbConfigLike:
    # DB-Credentials (Host/Port/DB/User/Pass)
    host: str
    port: int
    dbname: str
    user: str
    password: str


class DeddiagPgDataset:
    # Dataset-Client: Liest die DEDDIAG-Serien über die DB-Funktion get_measurements()
    def __init__(self, cfg: DeddiagPgConfig, schema: str = "public") -> None:
        self.cfg = cfg
        self.schema = schema
        self._conn: Optional[psycopg.Connection] = None

    def _ensure_conn(self) -> psycopg.Connection:
        # Stellt eine die Verbindung
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                host=self.cfg.db.host,
                port=self.cfg.db.port,
                dbname=self.cfg.db.dbname,
                user=self.cfg.db.user,
                password=self.cfg.db.password,
            )
        return self._conn

    def close(self) -> None:
        # Verbindung schließen
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def to_df(self, item_id: int, start: datetime, end: datetime) -> pd.DataFrame:
        # Messreihe als DataFrame ['time','value'] laden
        conn = self._ensure_conn()
        
        query = sql.SQL(
            "SELECT g.time, g.value "
            "FROM {}.get_measurements(%s::int, %s::timestamp, %s::timestamp) AS g "
            "ORDER BY g.time"
        ).format(sql.Identifier(self.schema))

        # Pandas: SQL ausführen und DataFrame erzeugen
        df = pd.read_sql_query(
            query.as_string(conn),
            conn,
            params=(int(item_id), start, end),
        )
        if df.empty:
            return df

        # Typbereinigung: time -> datetime, value -> float
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
        return df
