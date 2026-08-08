# Ergebnisnotiz (automatisch erzeugt)

## Datenbasis
- Quelle: results/coordination_baseline_2026-08-01/experiments
- Beobachtungen: 24 Runs
- Bedingungen: autogen, crewai, langgraph
- Designstand: 4 Aufgaben x 3 Frameworks x 2 Runs

## Zentrale deskriptive Befunde
- Erfolgsrate:
  - autogen: 1.00 (n=8)
  - crewai: 1.00 (n=8)
  - langgraph: 0.75 (n=8)
- Mittlere Latenz:
  - autogen: 0.0018 s
  - crewai: 15.4066 s
  - langgraph: 1.9754 s
- Dominanter MAST-Mode in allen Frameworks: 3.2 Weak Verification

## Inferenzbefunde (aktueller Stand)
- Success (ANOVA): p = 0.1216, kein statistisch signifikanter Unterschied auf alpha=0.05.
- Latency (ANOVA): p = 5.61e-05, deutlicher Framework-Effekt (eta^2 = 0.6063).
- Pairwise Latenz (Welch): alle drei Vergleiche signifikant; groesste mittlere Latenz bei crewai.

## Power-Interpretation (auf Pilotruns)
- Fuer Erfolgsraten-Differenzen von 15 Prozentpunkten (z. B. 0.75 vs 0.90) liegt der Richtwert bei ca. N=97 pro Gruppe (alpha=0.05, power=0.80).
- Fuer 20 Prozentpunkte (0.75 vs 0.95): ca. N=45 pro Gruppe.
- Fuer kontinuierliche Endpunkte mit mittlerem Effekt (d=0.5): ca. N=64 pro Gruppe.
- Schluss: Der aktuelle Datensatz ist fuer robuste Erfolgsraten-Inferenz unterpowert und sollte als explorativ berichtet werden.

## Methodische Limitationen fuer den Text
- Kleines N pro Zelle (n=2 pro Aufgabe und Framework).
- Fuer cost_usd liegt aktuell keine Varianz vor (faktisch 0), daher keine sinnvolle Kosteninferenz.
- Langgraph zeigt nicht-triviale Erfolgsstreuung, aber fuer belastbare Between-Framework-Aussagen ist N zu klein.

## Sofort nutzbare Schreibbotschaft
- "Die vorlaeufige Baseline zeigt einen robusten Framework-Unterschied in der Latenz, waehrend Unterschiede in der Erfolgsrate mit dem aktuellen Stichprobenumfang nicht signifikant nachweisbar sind. Entsprechend wird die Baseline als explorative Evidenz berichtet und fuer die confirmatory Success-Rate-Analyse ein hoehere Run-Zahl empfohlen."
