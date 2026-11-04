
import csv
from pathlib import Path

general = [
    ("Ich habe heute Morgen meinen Schlüssel vergessen.", "I forgot my key this morning.", "Past tense"),
    ("Kannst du mir bitte bei dieser Aufgabe helfen?", "Can you please help me with this task?", "Polite requests"),
    ("Wir fahren nächstes Wochenende nach Berlin.", "We are going to Berlin next weekend.", "Future plans"),
    ("Das Wetter ist heute besser als gestern.", "The weather is better today than yesterday.", "Comparatives"),
    ("Sie arbeitet seit fünf Jahren in diesem Unternehmen.", "She has been working at this company for five years.", "Present perfect"),
    ("Ich würde gerne einen Tisch für zwei Personen reservieren.", "I would like to reserve a table for two people.", "Conditional"),
    ("Obwohl es regnete, gingen wir spazieren.", "Although it was raining, we went for a walk.", "Concessive clauses"),
    ("Der Zug hat wegen des schlechten Wetters Verspätung.", "The train is delayed because of the bad weather.", "Causation"),
    ("Hast du schon einmal in Irland gelebt?", "Have you ever lived in Ireland?", "Life experience"),
    ("Wenn ich mehr Zeit hätte, würde ich ein Buch schreiben.", "If I had more time, I would write a book.", "Hypothetical condition"),
    ("Bitte schließen Sie das Fenster, bevor Sie gehen.", "Please close the window before you leave.", "Formal imperative"),
    ("Meine Schwester spricht drei Sprachen fließend.", "My sister speaks three languages fluently.", "Adverbs"),
    ("Wir müssen die Entscheidung bis Freitag treffen.", "We have to make the decision by Friday.", "Obligation"),
    ("Er hat vergessen, seine Hausaufgaben abzugeben.", "He forgot to hand in his homework.", "Infinitives"),
    ("Dieses Restaurant ist bekannt für seine regionalen Gerichte.", "This restaurant is known for its regional dishes.", "Passive constructions"),
    ("Ich freue mich darauf, dich wiederzusehen.", "I am looking forward to seeing you again.", "Idiomatic expressions"),
    ("Das Museum ist montags geschlossen.", "The museum is closed on Mondays.", "Schedules"),
    ("Sie fragte mich, ob ich morgen Zeit hätte.", "She asked me if I would have time tomorrow.", "Indirect questions"),
    ("Je früher wir anfangen, desto schneller sind wir fertig.", "The earlier we start, the sooner we will finish.", "Correlative comparatives"),
    ("Das ist das interessanteste Buch, das ich je gelesen habe.", "This is the most interesting book I have ever read.", "Superlatives"),
    ("Niemand wusste, warum das Treffen abgesagt wurde.", "Nobody knew why the meeting had been cancelled.", "Past perfect passive"),
    ("Ich bin daran gewöhnt, früh aufzustehen.", "I am used to getting up early.", "Habitual expressions"),
    ("Sie hat mir versprochen, pünktlich zu kommen.", "She promised me she would arrive on time.", "Reported intentions"),
    ("Wir sollten die Vor- und Nachteile sorgfältig abwägen.", "We should carefully weigh the advantages and disadvantages.", "Modal verbs"),
    ("Der Film war nicht so spannend, wie ich erwartet hatte.", "The film was not as exciting as I had expected.", "Comparison"),
    ("Er tut so, als ob er alles wüsste.", "He acts as if he knew everything.", "Subjunctive"),
    ("Ich habe weder Hunger noch Durst.", "I am neither hungry nor thirsty.", "Negation"),
    ("Sobald ich zu Hause bin, rufe ich dich an.", "As soon as I get home, I will call you.", "Time clauses"),
    ("Sie hat sich entschieden, im Ausland zu studieren.", "She has decided to study abroad.", "Reflexive verbs"),
    ("Wir haben uns seit letztem Sommer nicht gesehen.", "We have not seen each other since last summer.", "Present perfect"),
    ("Der Lehrer erklärte die Aufgabe noch einmal.", "The teacher explained the task once again.", "Past tense"),
    ("Es ist wichtig, genügend Wasser zu trinken.", "It is important to drink enough water.", "Impersonal constructions"),
    ("Ich weiß nicht, wem dieses Fahrrad gehört.", "I do not know who this bicycle belongs to.", "Indirect questions"),
    ("Sie konnte wegen des Lärms nicht einschlafen.", "She could not fall asleep because of the noise.", "Modal verbs"),
    ("Das Paket sollte spätestens morgen ankommen.", "The package should arrive by tomorrow at the latest.", "Expectations"),
    ("Wir haben beschlossen, das Projekt gemeinsam durchzuführen.", "We decided to carry out the project together.", "Verb phrases"),
    ("Er ist genauso groß wie sein Bruder.", "He is as tall as his brother.", "Equality comparisons"),
    ("Ich hätte dich früher informieren sollen.", "I should have informed you earlier.", "Past modals"),
    ("Die Kinder spielen draußen, während ihre Eltern kochen.", "The children are playing outside while their parents cook.", "Simultaneous actions"),
    ("Sie hat ihren Termin auf nächste Woche verschoben.", "She postponed her appointment until next week.", "Time expressions"),
    ("Es gibt mehrere Möglichkeiten, dieses Problem zu lösen.", "There are several ways to solve this problem.", "Existential constructions"),
    ("Ich erinnere mich nicht mehr an seinen Namen.", "I no longer remember his name.", "Negation"),
    ("Das Haus wurde vor mehr als hundert Jahren gebaut.", "The house was built more than a hundred years ago.", "Passive voice"),
    ("Er hat mich gebeten, ihm die Wahrheit zu sagen.", "He asked me to tell him the truth.", "Infinitive clauses"),
    ("Wir werden das Ergebnis morgen bekannt geben.", "We will announce the result tomorrow.", "Future tense"),
    ("Sie interessiert sich sowohl für Kunst als auch für Musik.", "She is interested in both art and music.", "Paired conjunctions"),
    ("Ich kann mich nicht entscheiden, welches Angebot besser ist.", "I cannot decide which offer is better.", "Embedded questions"),
    ("Das Problem ist schwieriger, als es zunächst scheint.", "The problem is more difficult than it initially seems.", "Comparatives"),
    ("Er hat den Wettbewerb gewonnen, obwohl er verletzt war.", "He won the competition although he was injured.", "Contrast"),
    ("Bitte lassen Sie mich wissen, falls Sie Fragen haben.", "Please let me know if you have any questions.", "Formal communication"),
]

technical = [
    ("Der Server konnte keine Verbindung zur Datenbank herstellen.", "The server could not establish a connection to the database.", "Database connectivity"),
    ("Bitte starten Sie die Anwendung nach der Installation neu.", "Please restart the application after installation.", "Installation"),
    ("Die API gibt einen Fehlercode zurück, wenn die Anfrage ungültig ist.", "The API returns an error code if the request is invalid.", "API errors"),
    ("Das Modell wurde mit einem überwachten Lernverfahren trainiert.", "The model was trained using supervised learning.", "Machine learning"),
    ("Die Daten werden vor der Verarbeitung verschlüsselt.", "The data is encrypted before processing.", "Encryption"),
    ("Stellen Sie sicher, dass alle Abhängigkeiten installiert sind.", "Make sure that all dependencies are installed.", "Dependencies"),
    ("Die Abfrage dauerte länger als erwartet.", "The query took longer than expected.", "Query performance"),
    ("Der Algorithmus erkennt Muster in großen Datenmengen.", "The algorithm identifies patterns in large datasets.", "Algorithms"),
    ("Die Anwendung unterstützt mehrere Benutzer gleichzeitig.", "The application supports multiple users simultaneously.", "Concurrency"),
    ("Der Zugriff auf diese Ressource wurde verweigert.", "Access to this resource was denied.", "Access control"),
    ("Die Trainingsdaten müssen vor der Modellierung bereinigt werden.", "The training data must be cleaned before modeling.", "Data preprocessing"),
    ("Die Genauigkeit des Modells wurde auf dem Testdatensatz gemessen.", "The model's accuracy was measured on the test dataset.", "Model evaluation"),
    ("Die Datei konnte aufgrund fehlender Berechtigungen nicht gespeichert werden.", "The file could not be saved due to insufficient permissions.", "Permissions"),
    ("Der Dienst wird automatisch neu gestartet, wenn ein Fehler auftritt.", "The service is automatically restarted when an error occurs.", "Fault recovery"),
    ("Die Antwortzeit der API beträgt durchschnittlich 120 Millisekunden.", "The average API response time is 120 milliseconds.", "Latency"),
    ("Die Datenbank enthält mehr als eine Million Datensätze.", "The database contains more than one million records.", "Database scale"),
    ("Bitte überprüfen Sie die Konfigurationsdatei auf Syntaxfehler.", "Please check the configuration file for syntax errors.", "Configuration"),
    ("Die GPU beschleunigt das Training neuronaler Netze.", "The GPU accelerates the training of neural networks.", "GPU computing"),
    ("Die Anwendung speichert Protokolle zur Fehleranalyse.", "The application stores logs for error analysis.", "Logging"),
    ("Der Speicherverbrauch steigt mit der Batchgröße.", "Memory usage increases with batch size.", "Batch processing"),
    ("Das Modell zeigt Anzeichen von Überanpassung.", "The model shows signs of overfitting.", "Overfitting"),
    ("Die Lernrate wurde nach jeder Epoche reduziert.", "The learning rate was reduced after each epoch.", "Learning rate"),
    ("Die Vorhersagen werden mit den tatsächlichen Werten verglichen.", "The predictions are compared with the actual values.", "Prediction evaluation"),
    ("Der Datensatz wurde in Trainings-, Validierungs- und Testdaten aufgeteilt.", "The dataset was split into training, validation, and test data.", "Dataset splitting"),
    ("Die Anwendung verwendet einen Cache, um wiederholte Berechnungen zu vermeiden.", "The application uses a cache to avoid repeated computations.", "Caching"),
    ("Die Verbindung wurde nach 30 Sekunden automatisch beendet.", "The connection was automatically terminated after 30 seconds.", "Timeouts"),
    ("Das System unterstützt sowohl synchrone als auch asynchrone Anfragen.", "The system supports both synchronous and asynchronous requests.", "Asynchronous processing"),
    ("Die Gewichte des neuronalen Netzes wurden aktualisiert.", "The weights of the neural network were updated.", "Neural networks"),
    ("Die Ergebnisse wurden in einer CSV-Datei gespeichert.", "The results were saved in a CSV file.", "Data export"),
    ("Die Pipeline führt mehrere Verarbeitungsschritte nacheinander aus.", "The pipeline executes several processing steps sequentially.", "Pipelines"),
    ("Die Fehlermeldung enthält keine vertraulichen Informationen.", "The error message does not contain any confidential information.", "Security"),
    ("Die Daten werden regelmäßig auf einem externen Server gesichert.", "The data is regularly backed up to an external server.", "Backups"),
    ("Die Schnittstelle akzeptiert ausschließlich JSON-Anfragen.", "The interface only accepts JSON requests.", "Data formats"),
    ("Das System konnte den angegebenen Pfad nicht finden.", "The system could not find the specified path.", "File paths"),
    ("Die neue Version ist mit älteren Geräten kompatibel.", "The new version is compatible with older devices.", "Compatibility"),
    ("Der Prozess wurde aufgrund unzureichenden Arbeitsspeichers beendet.", "The process was terminated due to insufficient memory.", "Memory management"),
    ("Die Funktion gibt eine leere Liste zurück, wenn keine Ergebnisse gefunden werden.", "The function returns an empty list if no results are found.", "Function behavior"),
    ("Die Datenübertragung erfolgt über eine sichere Verbindung.", "Data transmission takes place over a secure connection.", "Network security"),
    ("Das Modell wurde auf einem separaten Validierungsdatensatz bewertet.", "The model was evaluated on a separate validation dataset.", "Validation"),
    ("Die Anwendung darf keine personenbezogenen Daten protokollieren.", "The application must not log personal data.", "Data privacy"),
    ("Die Vorhersagequalität verschlechterte sich nach dem letzten Update.", "Prediction quality deteriorated after the latest update.", "Model monitoring"),
    ("Der Container wurde erfolgreich in der Cloud bereitgestellt.", "The container was successfully deployed to the cloud.", "Deployment"),
    ("Die Anfrage wurde aufgrund einer Zeitüberschreitung abgebrochen.", "The request was cancelled due to a timeout.", "Timeout handling"),
    ("Der Quellcode wird automatisch auf mögliche Sicherheitslücken überprüft.", "The source code is automatically checked for potential security vulnerabilities.", "Security scanning"),
    ("Die Datenbankindizes verbessern die Geschwindigkeit der Suchanfragen.", "Database indexes improve the speed of search queries.", "Database optimization"),
    ("Die Eingabedaten dürfen keine fehlenden Werte enthalten.", "The input data must not contain missing values.", "Data validation"),
    ("Das Modell verwendet einen Transformer mit mehreren Aufmerksamkeitsschichten.", "The model uses a Transformer with multiple attention layers.", "Transformer architecture"),
    ("Die Auswertung berücksichtigt sowohl BLEU als auch chrF.", "The evaluation considers both BLEU and chrF.", "Translation metrics"),
    ("Die Platzhalter müssen während der Übersetzung unverändert bleiben.", "The placeholders must remain unchanged during translation.", "Placeholder preservation"),
    ("Eine fehlende Verneinung kann die Bedeutung des Satzes umkehren.", "A missing negation can reverse the meaning of the sentence.", "Meaning preservation"),
]

assert len(general) == 50
assert len(technical) == 50

rows = []

for domain, examples in [("general", general), ("technical", technical)]:
    for index, (source, reference, focus) in enumerate(examples, start=1):
        rows.append({
            "id": f"{'GEN' if domain == 'general' else 'TECH'}-{index:03d}",
            "domain": domain,
            "source": source,
            "reference": reference,
            "focus": focus,
        })

path = Path("data/exercises/translation_exercises.csv")
path.parent.mkdir(parents=True, exist_ok=True)

with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["id", "domain", "source", "reference", "focus"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} exercises to {path}")
print("General:", sum(row["domain"] == "general" for row in rows))
print("Technical:", sum(row["domain"] == "technical" for row in rows))
