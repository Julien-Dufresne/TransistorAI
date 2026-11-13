
"""Gold-standard regression examples for TransistorAI."""

CASES = [
    # Correct translations and valid paraphrases (8)
    ("T01", "correct", "Ich habe zwei Bücher.",
     "I have two books.", "I have two books.", []),

    ("T02", "paraphrase", "Das Auto ist sehr schnell.",
     "The car is very fast.", "The car is really quick.", []),

    ("T03", "paraphrase", "Wir fahren nächstes Wochenende nach Berlin.",
     "We are going to Berlin next weekend.",
     "We are driving to Berlin next weekend.", []),

    ("T04", "paraphrase", "Sie ist sehr glücklich.",
     "She is very happy.", "She is delighted.", []),

    ("T05", "correct", "Der Hund schläft auf dem Sofa.",
     "The dog is sleeping on the sofa.",
     "The dog is sleeping on the sofa.", []),

    ("T06", "paraphrase", "Kannst du mir helfen?",
     "Can you help me?", "Could you help me?", []),

    ("T07", "paraphrase", "Ich habe Hunger.",
     "I am hungry.", "I'm hungry.", []),

    ("T08", "paraphrase", "Das Wetter ist heute schön.",
     "The weather is nice today.",
     "It's lovely weather today.", []),

    # Grammar and tense (5)
    ("T09", "grammar", "Ich habe gestern gearbeitet.",
     "I worked yesterday.", "I work yesterday.", ["grammar"]),

    ("T10", "grammar", "Sie geht jeden Tag zur Schule.",
     "She goes to school every day.",
     "She go to school every day.", ["grammar"]),

    ("T11", "grammar", "Er hat das Buch gelesen.",
     "He read the book.", "He has read the book yesterday.", ["grammar"]),

    ("T12", "grammar", "Wenn ich Zeit hätte, würde ich kommen.",
     "If I had time, I would come.",
     "If I have time, I would come.", ["grammar"]),

    ("T13", "grammar", "Wir waren gestern zu Hause.",
     "We were at home yesterday.",
     "We was at home yesterday.", ["grammar"]),

    # Vocabulary and meaning (5)
    ("T14", "meaning", "Ich vermisse meine Familie.",
     "I miss my family.", "I visit my family.", ["meaning"]),

    ("T15", "meaning", "Das ist verboten.",
     "That is forbidden.", "That is allowed.", ["meaning"]),

    ("T16", "meaning", "Er hat den Zug verpasst.",
     "He missed the train.", "He caught the train.", ["meaning"]),

    ("T17", "vocabulary", "Ich brauche einen Regenschirm.",
     "I need an umbrella.", "I need a raincoat.", ["vocabulary"]),

    ("T18", "meaning", "Sie hat die Prüfung bestanden.",
     "She passed the exam.", "She failed the exam.", ["meaning"]),

    # Numbers and negation (5)
    ("T19", "numbers", "Ich habe drei Katzen.",
     "I have three cats.", "I have two cats.", ["numbers"]),

    ("T20", "numbers", "Das kostet fünf Euro.",
     "It costs five euros.", "It costs fifteen euros.", ["numbers"]),

    ("T21", "numbers", "Wir haben vier Tickets gekauft.",
     "We bought four tickets.", "We bought three tickets.", ["numbers"]),

    ("T22", "meaning", "Ich mag keinen Kaffee.",
     "I don't like coffee.", "I like coffee.", ["meaning"]),

    ("T23", "meaning", "Er kommt heute nicht.",
     "He is not coming today.",
     "He is coming today.", ["meaning"]),

    # Omissions and additions (4)
    ("T24", "omissions", "Ich habe gestern Pizza gegessen.",
     "I ate pizza yesterday.", "I ate pizza.", ["omissions"]),

    ("T25", "omissions", "Sie hat einen kleinen Hund.",
     "She has a small dog.", "She has a dog.", ["omissions"]),

    ("T26", "additions", "Er trinkt Wasser.",
     "He drinks water.", "He drinks cold water.", ["additions"]),

    ("T27", "omissions", "Wir treffen uns morgen früh.",
     "We will meet tomorrow morning.",
     "We will meet tomorrow.", ["omissions"]),

    # Multiple errors and complex meaning (3)
    ("T28", "multiple", "Ich habe gestern drei Bücher gekauft.",
     "I bought three books yesterday.",
     "I buy two books yesterday.",
     ["grammar", "numbers"]),

    ("T29", "multiple", "Sie hat gestern nicht gearbeitet.",
     "She did not work yesterday.",
     "She work yesterday.",
     ["grammar", "meaning"]),

    ("T30", "multiple", "Obwohl es regnete, gingen wir spazieren.",
     "Although it was raining, we went for a walk.",
     "In case it is not raining, we are going to walk.",
     ["meaning", "grammar"]),
]

assert len(CASES) == 30
assert len({case[0] for case in CASES}) == 30
