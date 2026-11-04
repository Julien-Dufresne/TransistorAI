from evaluation.streamlit_comet import calculate_comet_scores


def main():
    scores = calculate_comet_scores(
        source="Der Server konnte keine Verbindung zur Datenbank herstellen.",
        user_translation="The server could not connect to the database.",
        model_translation="The server established a connection to the database.",
        reference="The server could not establish a connection to the database.",
    )

    print("\nCOMET RESULTS")
    print(f"Correct translation:   {scores['user_comet']:.4f}")
    print(f"Incorrect translation: {scores['model_comet']:.4f}")


if __name__ == "__main__":
    main()