# Slikovni nabori za teste

Oba izbora vsebujeta po 50 slik iz ene serije THUD, z oznakami in spremljevalnimi
datotekami. Vsak izbor je v svoji mapi.

| Mapa | Serija | Namen |
| --- | --- | --- |
| [thud_dense_test50](thud_dense_test50/README.md) | Real_Scenes/store/dynamic/Capture_1 | **Priporočeni izbor:** polni prizori s policami, izdelki in opremo; prednost ima malo praznega prostora. |
| [thud_agtea_test50](thud_agtea_test50/README.md) | Real_Scenes/agtea/dynamic/Capture_1 | Prvotni izbor z več ločenimi oznakami in raznolikimi vrstami objektov. |

Vsaka mapa vsebuje:

- `images/`: 50 izvirnih PNG slik.
- `annotations/`: pripadajoče izvorne oznake LabelMe brez podvojenih slik v base64.
- `objects.jsonl`: zapise s potmi do slik in oznak ter okvirji objektov.
- `gallery.html`, `preview/` in `contact_sheet_*.jpg`: vizualni pregled.
- `README.md`, `summary.json`, `selection.csv`: opis, statistiko in merila izbora.
- `SHA256SUMS.txt`: kontrolne vrednosti slik.
- Datoteko s primerjavo izvornih serij.

Za lokalni pregled odpri `gallery.html` v izbrani mapi. Poti do slik in oznak
v JSONL so relativne glede na mapo izbora, zato nabora delujeta tudi po kloniranju
repozitorija brez celotnega izvornega THUD nabora.

Za ponovitev izbora uporabi [skripto](../scripts/select_thud_test_images.py);
ponovitev zahteva celotno pripadajočo izvorno serijo v lokalnem `data/THUD_Robot`.
Izvorni celotni nabor ostaja izključen iz Gita.

Oznake `Products` v trgovinskem naboru pogosto združujejo več izdelkov v en
okvir. Število pravokotnih oznak zato ni število vseh vidnih fizičnih predmetov.
Referenčni razredi mobilnosti `fixed/movable/uncertain` še niso podani;
posebnosti uporabe z vizualnim cevovodom so opisane v README posameznega izbora.

## Zagon z ločenim prvim korakom

Na veji `feature/separate-vision-stages` sta oba nabora povezana z novim bralnikom,
ki posreduje izvorne `bbox_xyxy`, lokalne ID-je in imena oznak. Ni treba ponovno
pripravljati `frames.jsonl` ali priklopiti celotne baze/zunanjega diska.

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
bash scripts/stage1_describe.sh thud-store-hitri \
  --config configs/stage1/thud-compare.json \
  --input data/thud_dense_test50/objects.jsonl --limit 2
```

Za celoten nabor uporabi novo ime poskusa in `--limit 50`. Za drugi izbor zamenjaj
vhod z `data/thud_agtea_test50/objects.jsonl` in ime poskusa.
Konfiguracija THUD uporablja do 8192 izhodnih tokenov in kontekst 16384, ker je
na sliki lahko do 37 označenih regij. Daljši opisi lahko zahtevajo več časa in VRAM.
Prvi korak ne izvaja razvrščanja fiksno/premično. Navodila: [WORKFLOW](../docs/WORKFLOW.md).
