# Bereinigte Baseline-Zusammenfassung (ohne MetaGPT)

MetaGPT wurde ausgeschlossen, da die Runs im Scaffold/Fallback-Modus liefen und daher nicht als valide Framework-Leistung interpretierbar sind.

## Deskriptive Kennzahlen

| Framework | n | Success-Rate | 95%-CI (Success) | mittlere Latenz (s) |
|---|---:|---:|---:|---:|
| autogen | 240 | 0.750 | [0.695, 0.805] | 132.504 |
| crewai | 240 | 1.000 | [1.000, 1.000] | 11.435 |
| langgraph | 240 | 0.529 | [0.466, 0.593] | 2.591 |

## Inferenz (Success)

- autogen vs crewai: p=1.196e-16, Effekt cohens_d=-0.815
- autogen vs langgraph: p=3.535e-07, Effekt cohens_d=0.472
- crewai vs langgraph: p=6.969e-35, Effekt cohens_d=1.331

## Inferenz (Latenz)

- autogen vs crewai: p=2.424e-71, Effekt cohens_d=2.357
- autogen vs langgraph: p=9.311e-77, Effekt cohens_d=2.534
- crewai vs langgraph: p=2.587e-86, Effekt cohens_d=2.662

## Hinweis zur Interpretation

- Kostenmetriken sind im Lauf als 0.0 protokolliert und daher nicht inferenziell belastbar.
- SciPy Precision-Loss-Warnungen betreffen teils nahezu identische Verteilungen; Effektgrößen und CIs priorisieren.