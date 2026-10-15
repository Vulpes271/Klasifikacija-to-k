# Dosedanji poskusi z vmesnimi rezultati

Aktivni potek je opisan v [WORKFLOW.md](../WORKFLOW.md). Te poskuse ohranimo zaradi
sledljivosti in ponovitve dosedanjih rezultatov, ločeno od novih korakov.

| Starejša pot | Namen in pripadajoči rezultati |
| --- | --- |
| `scripts/test_vision_pipeline.sh` | Dosedanji združeni slika -> opis -> reasoning; npr. `results/vision-prvi` |
| `scripts/test_vision_benchmark.sh` | BF16 opis -> primerjava BF16/Q4 reasoning; npr. `results/vision-celotni` |
| `scripts/test_qwen35.sh` | Prvi majhen test razvrščanja iz oznak, brez slikovnega modela |
| `python -m llm_object_classifier run/baseline/evaluate` | Prvotni primerjalnik oznak in ročno vrednotenje |
| `python -m llm_object_classifier prepare-thud` | Doslej dokumentirani sintetični pretvornik; realna shema čaka na vzorec |

Dokumentacija: [združeni GPU benchmark](../../README_GPU.md),
[nastavitve združenega testa](../../README_NASTAVITVE.md),
[prvi Docker preizkus](../../README_DOCKER.md),
[prvotna navodila za razvrščanje oznak](README_LABELS.md).

Lokalnih `results/`, `data/` in modelov ne brišemo in jih ne prenašamo v Git.
Prvotni eksperimenti imajo svoje uporabljene prompte v manifestih; sprememba novega
prompta ne spremeni že izmerjenih rezultatov.
