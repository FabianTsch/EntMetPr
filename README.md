# EntMetPr
Entwicklungsmethodik Projekt

# Schnittstellen

## Bildverarbeitung

- Kamerabild -> Segmentierung: Rektifiziertes Bildausschnitt der Arbeitsfläche, skaliert auf 2px pro mm
    - Evtl. Anpassung der Randbereiche
- Segmentierung -> Objekterkennung/ Merkmalsextraktion: Binärbild (np.array)
- Objekterkennung -> Klassifizierung: Liste von Teilbildern (RGB)
    - Anpassen der Bildgröße vor Übergabe an CNN
    - RGB-Bilder mit 3x224x224 im Bereich [0,1]
    - Skalierung, Erweiterung der Bildränder mit Randpixeln, ...
- Definition von Features für Objekterkennung und Aufgreifen?
