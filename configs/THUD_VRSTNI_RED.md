# Prvi del: vrstni red poskusa

1. **Izbor in preverjanje:** `python3 -m llm_object_classifier.thud_experiment --phase prepare`; vhodna nespremenjena THUD JSONL in SHA256SUMS, izhod `results/thud-vmesno/inputs/`.
2. **Pilot, nato opis/detekcija:** `--phase pilot`, `--phase baseline`. Osnova `stage1/thud-part1.json` (trije modeli); ločeni `discovery`/`supplied`, brez spreminjanja izvornih slik.
3. **Prompt/temperatura, nato reasoning:** `--phase additional`. Opisovalnik `prompts/thud-description-{long,short}.txt`; razvrščevalnik `prompts/thud-{long-examples,long-no-examples,short-no-examples}.txt`. Načrt v `thud_experiment.py` spreminja po eno spremenljivko.
4. **Vrednotenje:** `thud_metrics.py`, samodejna statistika in nepopravljena surova napaka. Ročna preglednica istega podnabora ostaja osnutek.
5. **Poročilo:** `python3 -m llm_object_classifier.thud_report`, `bash scripts/compile_thud_report.sh`.

Vse skupaj: `bash scripts/thud_experiment.sh --phase all`. To ni 2. del nadgradnje iz CODEX_NALOGA.md. Obstoječe konfiguracije in stari NYU poskusi ostanejo združljivi.
