from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DB_DEFAULT_PATH = "risk_profile_sessions.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str = DB_DEFAULT_PATH) -> sqlite3.Connection:
    db_file = Path(db_path)
    if db_file.parent != Path("."):
        db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_DEFAULT_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                mode_key TEXT,
                mode_title TEXT,
                episode_id TEXT,
                score_model_version TEXT,
                onboarding_json TEXT NOT NULL,
                episode_meta_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decision_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_idx INTEGER NOT NULL,
                event_ts_utc TEXT NOT NULL,
                month_index INTEGER,
                regime TEXT,
                phase_label TEXT,
                action_label TEXT,
                rebalance INTEGER,
                go_to_cash INTEGER,
                reduce_risk INTEGER,
                end_early INTEGER,
                discomfort_1_5 INTEGER,
                portfolio_return REAL,
                portfolio_value REAL,
                drawdown REAL,
                raw_event_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_decision_events_session_idx
            ON decision_events(session_id, event_idx);

            CREATE TABLE IF NOT EXISTS final_scores (
                session_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                score_model_version TEXT NOT NULL,
                risk_tolerance_score REAL NOT NULL,
                behavioral_reactivity_score REAL NOT NULL,
                raw_final_risk_index REAL,
                final_risk_index REAL NOT NULL,
                risk_group_3 TEXT NOT NULL,
                risk_group_5 TEXT NOT NULL,
                profile_reliability_score REAL,
                profile_reliability_label TEXT,
                features_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """
        )


def start_session(
    session_id: str,
    onboarding: dict[str, Any],
    mode_key: str,
    mode_title: str,
    episode_id: str,
    episode_meta: dict[str, Any],
    db_path: str = DB_DEFAULT_PATH,
) -> None:
    started_at = _utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, started_at_utc, mode_key, mode_title, episode_id,
                onboarding_json, episode_meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                mode_key=excluded.mode_key,
                mode_title=excluded.mode_title,
                episode_id=excluded.episode_id,
                onboarding_json=excluded.onboarding_json,
                episode_meta_json=excluded.episode_meta_json
            """,
            (
                session_id,
                started_at,
                mode_key,
                mode_title,
                episode_id,
                json.dumps(onboarding, ensure_ascii=False),
                json.dumps(episode_meta, ensure_ascii=False),
            ),
        )


def append_decision_event(
    session_id: str,
    event_idx: int,
    event_data: dict[str, Any],
    db_path: str = DB_DEFAULT_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO decision_events (
                session_id, event_idx, event_ts_utc, month_index, regime, phase_label, action_label,
                rebalance, go_to_cash, reduce_risk, end_early, discomfort_1_5,
                portfolio_return, portfolio_value, drawdown, raw_event_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                int(event_idx),
                str(event_data.get("timestamp_utc", _utc_now_iso())),
                int(event_data.get("month_index", 0)),
                str(event_data.get("regime", "")),
                str(event_data.get("phase_label", "")),
                str(event_data.get("action_label", "")),
                int(event_data.get("rebalance", 0)),
                int(event_data.get("go_to_cash", 0)),
                int(event_data.get("reduce_risk", 0)),
                int(event_data.get("end_early", 0)),
                int(event_data.get("discomfort_1_5", 0)),
                float(event_data.get("portfolio_return", 0.0)),
                float(event_data.get("portfolio_value", 0.0)),
                float(event_data.get("drawdown", 0.0)),
                json.dumps(event_data, ensure_ascii=False),
            ),
        )


def save_final_scores(
    session_id: str,
    scores: dict[str, Any],
    db_path: str = DB_DEFAULT_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO final_scores (
                session_id, created_at_utc, score_model_version, risk_tolerance_score,
                behavioral_reactivity_score, raw_final_risk_index, final_risk_index,
                risk_group_3, risk_group_5, profile_reliability_score, profile_reliability_label,
                features_json, scores_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                created_at_utc=excluded.created_at_utc,
                score_model_version=excluded.score_model_version,
                risk_tolerance_score=excluded.risk_tolerance_score,
                behavioral_reactivity_score=excluded.behavioral_reactivity_score,
                raw_final_risk_index=excluded.raw_final_risk_index,
                final_risk_index=excluded.final_risk_index,
                risk_group_3=excluded.risk_group_3,
                risk_group_5=excluded.risk_group_5,
                profile_reliability_score=excluded.profile_reliability_score,
                profile_reliability_label=excluded.profile_reliability_label,
                features_json=excluded.features_json,
                scores_json=excluded.scores_json
            """,
            (
                session_id,
                _utc_now_iso(),
                str(scores.get("score_model_version", "heuristic_v2")),
                float(scores.get("risk_tolerance_score", 0.0)),
                float(scores.get("behavioral_reactivity_score", 0.0)),
                float(scores.get("raw_final_risk_index", scores.get("final_risk_index", 0.0))),
                float(scores.get("final_risk_index", 0.0)),
                str(scores.get("risk_group_3", "")),
                str(scores.get("risk_group_5", "")),
                float(scores.get("profile_reliability_score", 0.0)),
                str(scores.get("profile_reliability_label", "")),
                json.dumps(scores.get("features", {}), ensure_ascii=False),
                json.dumps(scores, ensure_ascii=False),
            ),
        )
        conn.execute(
            """
            UPDATE sessions
            SET completed_at_utc=?, score_model_version=?
            WHERE session_id=?
            """,
            (
                _utc_now_iso(),
                str(scores.get("score_model_version", "heuristic_v2")),
                session_id,
            ),
        )


def load_calibration_dataset(db_path: str = DB_DEFAULT_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                s.session_id,
                s.mode_key,
                s.started_at_utc,
                s.completed_at_utc,
                s.onboarding_json,
                f.features_json,
                f.final_risk_index,
                f.risk_group_3,
                f.risk_group_5,
                f.profile_reliability_score,
                f.profile_reliability_label,
                f.score_model_version
            FROM sessions s
            JOIN final_scores f ON f.session_id = s.session_id
            ORDER BY s.started_at_utc ASC
            """,
            conn,
        )

    if frame.empty:
        return frame

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        features = json.loads(row["features_json"])
        onboarding = json.loads(row["onboarding_json"])
        out = {
            "session_id": row["session_id"],
            "mode_key": row["mode_key"],
            "started_at_utc": row["started_at_utc"],
            "completed_at_utc": row["completed_at_utc"],
            "investment_horizon_years": int(onboarding.get("investment_horizon_years", 0) or 0),
            "final_risk_index": float(row["final_risk_index"]),
            "risk_group_3": row["risk_group_3"],
            "risk_group_5": row["risk_group_5"],
            "profile_reliability_score": float(row["profile_reliability_score"] or 0.0),
            "profile_reliability_label": row["profile_reliability_label"],
            "score_model_version": row["score_model_version"],
        }
        for key, value in features.items():
            out[key] = value
        rows.append(out)

    return pd.DataFrame(rows)
