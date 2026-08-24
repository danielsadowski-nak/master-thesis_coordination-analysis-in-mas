# Reviewer Briefing (Phase C v4)

## Ziel
Sie annotieren 60 unabhängige Fälle aus Multi-Agent-Läufen.
Ihre Einschätzung wird für eine Human-vs-Judge-Validierung der MAST-Taxonomie verwendet.

## Ihre Datei
- Reviewer 1: reviewer1_form.html
- Reviewer 2: reviewer2_form.html

Bitte nur die Ihnen zugewiesene Datei verwenden.

## Ablauf in 6 Schritten
1. HTML-Datei lokal im Browser öffnen.
2. Reviewer-ID einmal links eintragen.
3. Für jedes Item drei Pflichtfelder ausfüllen:
   - Aufgabe erfolgreich gelöst: Ja oder Nein
   - Primäre MAST-Fehlermodi: mindestens eine Auswahl oder NO_FAILURE_MODE
   - Kurze Evidenz-Zusammenfassung: 1 bis 3 Sätze
4. Optional Notizen ergänzen.
5. Bis 60 von 60 Items vollständig annotiert sind.
6. CSV herunterladen und zurückgeben.

## Bewertungsregeln (kurz)
- Nur Evidenz aus Final Output oder Trace verwenden.
- Keine Annahmen aus dem Task-Titel allein ableiten.
- Lieber wenige, klar belegte Fehlermodi als Overlabeling.
- Aufgabe gilt nur als erfolgreich, wenn das Ergebnis die Kriterien erfüllt (nicht nur behauptet).

## Qualitätskriterien pro Item
Ein Item gilt als vollständig, wenn:
- Ja/Nein gesetzt ist,
- mindestens ein MAST-Modus gewählt ist (oder NO_FAILURE_MODE),
- die Evidenz-Zusammenfassung nicht leer ist.

## Datenschutz und Unabhängigkeit
- Bitte keine Abstimmung mit anderen Reviewern während der Erstannotation.
- Keine Veränderung an Metadaten oder Item-Reihenfolge.

## Abgabe
- Exportieren Sie am Ende die CSV aus dem Formular.
- Dateiname (empfohlen):
  - phase_c_v4_reviewer1_annotations.csv
  - phase_c_v4_reviewer2_annotations.csv

## Bei Problemen
- Browser neu laden (Zwischenstand bleibt lokal gespeichert).
- Falls CSV-Export fehlschlägt: JSON-Backup im Formular herunterladen und mit zurückgeben.
