# BF16 razširitev istih 40 Florence vhodov

Izvajalnik ponovno uporabi manifest, zamrznjene dokumente, slike ter natančna shranjena besedila obeh sistemskih in vseh uporabniških promptov v `results models/prompt_comparison40`. Nove datoteke so samo v podmapi `bf16/`; izvorni manifest ostane nespremenjen.

Modela sta `qwen3.5:4b-bf16` in `qwen3.5:9b-bf16`, preverjena z metapodatkom BF16 in digestom uteži. Manjkajoči model se prenese, nameščeni se ponovno uporabi. Izvajanje je zaporedno, z enim kratkim GPU preflightom na model pri kontekstu 8192. Ni CPU ali Q4/Q8 nadomestitve.

```bash
bash scripts/prompt_comparison_bf16.sh run
```

Obseg je 160 glavnih položajev: 40 slik × 2 modela × 2 prompta, samo seme 42. Ostale možnosti so iz obstoječega manifesta (T=0.3, največ 2048 izhodnih tokenov, kontekst 8192, isti samplerji, think=false in format=json). Dva kratka 16-token GPU preflighta sta shranjena ločeno in izključena iz glavnih položajev. Zaključeni položaji se ne ponavljajo; podpis se pred ponovno uporabo preveri. Drugje najdeni odgovori se ponovno uporabijo samo ob preverjenem enakem vhodu, promptu, modelu/digestu in nastavitvah.

Vhod se ne krajša. Konservativna ocena konteksta lahko uporabi že izmerjene tokene enakega besedila, če sta besedilni BPE in pogovorna predloga preverjeno enaka, z dodatno rezervo 128 tokenov. Pretvorbe BF16/Q4 imajo lahko drugačne neuporabljene audio/vision posebne tokene; ti niso razglašeni za enakovredne. Besedila s takimi razlikami uporabijo splošno konservativno preverjanje. Ocena ni točno BF16 štetje; po vsakem klicu se shranijo dejanski števci. Preflight dodatno preveri oceno na najdaljšem vhodu. Nezadosten/nepreverljiv kontekst se zabeleži kot preskok.

Surovi odgovori, dejanske zahteve, čas, tokeni, stanje GPU, podpis, schema/finish napake in natančnost ostanejo v `bf16/responses/`, `bf16/models/` in `bf16/preflights/`. `bf16/summary.json` vsebuje števce, ki niso accuracy. `bf16/index.html` in `all-region-classes.csv` vključujeta tudi stare rezultate in ponovitve.

Za ročno ocenjevanje:

```bash
python3 -m llm_object_classifier.review_viewer
```

Ocene so zasebne v `reviews/manual_review.sqlite3`. Razširitev ohrani identiteto nabora; ob dodajanju modelov se pred migracijo shrani varnostna kopija. Uteži in ocene ostanejo izključene iz Gita.
