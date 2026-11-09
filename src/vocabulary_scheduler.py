
"""
Day 19 — Daily vocabulary scheduling.

Selects five new words and up to five review words per learner.
Assignments are persisted in SQLite to survive Streamlit reruns.
"""

from datetime import date

from src.vocabulary_db import connect_db, initialize_database


def get_daily_words(learner_id, study_date=None):
    """Return persistent daily new and review vocabulary assignments."""
    if not learner_id or not learner_id.strip():
        raise ValueError("A learner ID is required.")

    initialize_database()
    today = study_date or date.today().isoformat()

    with connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")

        existing = conn.execute(
            """
            SELECT COUNT(*)
            FROM daily_assignments
            WHERE learner_id = ? AND study_date = ?
            """,
            (learner_id, today),
        ).fetchone()[0]

        if existing == 0:
            # Select words never assigned to this learner.
            new_words = conn.execute(
                """
                SELECT v.word_id
                FROM vocabulary v
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM daily_assignments a
                    WHERE a.learner_id = ?
                      AND a.word_id = v.word_id
                      AND a.assignment_type = 'new'
                )
                ORDER BY RANDOM()
                LIMIT 5
                """,
                (learner_id,),
            ).fetchall()

            for (word_id,) in new_words:
                conn.execute(
                    """
                    INSERT INTO daily_assignments
                        (learner_id, study_date, word_id, assignment_type)
                    VALUES (?, ?, ?, 'new')
                    """,
                    (learner_id, today, word_id),
                )

            # Review previously introduced vocabulary.
            # Incorrect answers receive higher priority.
            review_words = conn.execute(
                """
                SELECT v.word_id
                FROM vocabulary v
                JOIN daily_assignments previous
                  ON previous.word_id = v.word_id
                 AND previous.learner_id = ?
                 AND previous.assignment_type = 'new'
                 AND previous.study_date < ?
                LEFT JOIN learner_progress p
                  ON p.word_id = v.word_id
                 AND p.learner_id = ?
                GROUP BY v.word_id
                ORDER BY
                    COALESCE(p.incorrect_count, 0) DESC,
                    COALESCE(p.last_reviewed, '') ASC,
                    v.word_id ASC
                LIMIT 5
                """,
                (learner_id, today, learner_id),
            ).fetchall()

            for (word_id,) in review_words:
                conn.execute(
                    """
                    INSERT INTO daily_assignments
                        (learner_id, study_date, word_id, assignment_type)
                    VALUES (?, ?, ?, 'review')
                    """,
                    (learner_id, today, word_id),
                )

        conn.commit()

        # Read the persisted assignments.
        rows = conn.execute(
            """
            SELECT
                v.word_id,
                v.german,
                v.english,
                v.level,
                v.category,
                a.assignment_type
            FROM daily_assignments a
            JOIN vocabulary v ON v.word_id = a.word_id
            WHERE a.learner_id = ?
              AND a.study_date = ?
            ORDER BY
                CASE a.assignment_type
                    WHEN 'new' THEN 0
                    ELSE 1
                END,
                v.word_id
            """,
            (learner_id, today),
        ).fetchall()

    columns = [
        "word_id", "german", "english",
        "level", "category", "assignment_type"
    ]

    return [dict(zip(columns, row)) for row in rows]
