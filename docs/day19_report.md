# TransistorAI — Day 19 Development Report

**Date:** November 9, 2026  
**Project:** TransistorAI — German-to-English Machine Translation & Language Learning  
**Development focus:** Multi-user support, personalized learning progress, and persistent evaluation data  
**Status:** Implementation completed and committed to Git

---

## 1. Executive Summary

Day 20 focused on transforming TransistorAI from a primarily single-user application into a multi-user Streamlit MVP.

The application now supports individual user profiles, allowing multiple learners to use the same application while maintaining independent learning progress, translation attempts, and human preference evaluation votes.

The implementation reuses existing SQLite databases and model evaluation components, minimizing changes to the machine translation pipeline.

**Key achievements:**
- Integrated user registration and profile selection.
- Connected Daily Vocabulary to individual user profiles.
- Implemented user-specific Test Yourself attempts and scores.
- Connected Human Preference Evaluation to individual users.
- Verified database persistence and access isolation.
- Resolved a Streamlit/PyTorch development-environment issue.
- Committed the multi-user implementation to Git.

## 2. Multi-User Architecture

### User profiles

The application uses a dedicated SQLite database:

`data/users/users.sqlite3`

Each user has a unique UUID and username. The active profile is maintained in Streamlit session state and passed to personalized application components.

### Application structure

| Streamlit tab | User-specific? | Functionality |
|---|---|---|
| Translate | No | Shared German-to-English translation |
| Test Yourself | Yes | Translation exercises and evaluation scores |
| Daily Vocabulary | Yes | Daily words, quizzes, and learning progress |
| Human Preference Evaluation | Yes | Blind translation comparisons and voting |
| Human Evaluation Analytics | No | Aggregated human evaluation results |

This architecture separates individual learning activities from shared research functionality.

## 3. Daily Vocabulary Integration

Updated `vocabulary_page.py` to accept the selected user's UUID rather than relying on a hardcoded learner identifier.

The selected UUID is passed from `MVP_streamlit_app.py` into the vocabulary component.

**Validated functionality:**
- Independent daily vocabulary assignments.
- Separate quiz answers and progress.
- Correct progress restoration when switching profiles.
- Existing legacy vocabulary records preserved.

The underlying vocabulary dataset and scheduling logic remain unchanged.

## 4. Test Yourself — Persistent User-Specific Attempts

Updated `evaluation/streamlit_attempts.py` and the corresponding Streamlit interface.

### Database migration

Added a nullable `user_id` column to the existing `attempts` table in:

`data/exercises/translation_attempts.sqlite3`

This supports new user-owned attempts while retaining historical records.

### Updated database functions

| Function | Change |
|---|---|
| `initialize_database()` | Supports schema migration and user index |
| `save_attempt()` | Stores the selected user UUID |
| `get_attempt()` | Restricts retrieval to the attempt owner |
| `save_scores()` | Restricts score updates to the attempt owner |
| `get_user_attempts()` | Retrieves a user's attempt history |

Streamlit answer and active-attempt state keys were also updated to avoid mixing user sessions.

### Validation results

The database contained two historical attempts before migration.

After testing:

| Record type | Count |
|---|---:|
| Historical attempts | 2 |
| New user-owned attempts | 1 |
| Total | 3 |

An explicit ownership test confirmed:

- The correct user could retrieve their attempt.
- An unrelated user ID could not retrieve that attempt.
- Both historical attempts remained intact.

**Result:** User-specific attempt persistence and retrieval isolation verified.

## 5. Human Preference Evaluation Integration

Updated `human_preference_page.py` to use the selected application user UUID directly as the evaluation participant ID.

This removes the need for a separate anonymous participant selection and manual resume process.

The evaluation continues to use:

`data/human_evaluation/preference_votes.sqlite3`

### Research functionality retained

- Dataset of 200 German sentences.
- Four translation systems: E0, E1, E2, and E3.
- Blind translation comparisons.
- Deterministic option shuffling.
- Shared 20-sentence subset for inter-annotator agreement.
- Persistent votes and evaluation progress.
- Shared analytics across participants.

Historical research votes were no longer required, allowing the previous voting database to be reset.

### Database verification

Two distinct participant UUIDs were recorded:

| Participant | Saved votes |
|---|---:|
| Participant A | 3 |
| Participant B | 1 |
| **Total** | **4** |

**Result:** Independent participant vote storage confirmed.

## 6. Troubleshooting and Environment Stability

During Streamlit startup, a PyTorch file-watcher warning appeared:

`Tried to instantiate class '__path__._path'`

The application also returned to VS Code after selecting a user.

The application was subsequently launched through a regular macOS Terminal using the Conda environment and disabled Streamlit file watching:

```bash
conda activate transistorai-fresh

python -m streamlit run MVP_streamlit_app.py \
  --server.fileWatcherType none
```

This resolved the immediate application-launch problem.

The issue was consistent with a development-environment interaction involving Streamlit, PyTorch, or VS Code, although the precise root cause was not independently established.

## 7. Git Version Control

The multi-user implementation was committed successfully.

Key implementation files included:

- `src/user_db.py`
- `user_selection_page.py`
- `MVP_streamlit_app.py`
- `evaluation/streamlit_attempts.py`
- `human_preference_page.py`
- `vocabulary_page.py`

Local SQLite databases and generated research artifacts were intentionally excluded from the feature commit.

## 8. Validation Summary

| Test | Result |
|---|---|
| User selection screen | Passed |
| Daily Vocabulary profile isolation | Passed |
| Test Yourself attempt persistence | Passed |
| Test Yourself retrieval isolation | Passed |
| Historical attempt preservation | Passed |
| Human preference vote persistence | Passed |
| Streamlit application startup | Passed |
| Full five-tab regression test | Pending final confirmation |
| Persistence after complete application restart | Pending final confirmation |

## 9. Remaining Work

Before the final MVP presentation:

1. Run an end-to-end regression test covering all five tabs.
2. Confirm user progress persists after a complete application restart.
3. Verify that Human Evaluation Analytics correctly incorporates the new voting data.
4. Update the README with the new multi-user architecture and application workflow.
5. Prepare the final presentation and live demonstration.

Optional enhancement: add a Test Yourself history interface so users can revisit previous translations and evaluation scores.

## 10. Conclusion

Day 20 delivered a significant improvement to the TransistorAI MVP.

The application now supports multiple learners with independent vocabulary progress, translation exercise attempts, and human preference evaluation votes, while retaining shared machine translation and research analytics functionality.

The implementation demonstrates practical experience with Streamlit session management, SQLite schema migration, persistent state, user-specific data isolation, and application integration.

**Overall outcome:** Multi-user implementation completed, targeted persistence and isolation tests passed, and changes committed to Git. Final end-to-end regression testing remains outstanding.