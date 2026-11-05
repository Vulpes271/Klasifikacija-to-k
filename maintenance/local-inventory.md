# Lokalni popis projekta — 2026-11-05

Popis pred commitom tega dokumenta; datum Europe/Ljubljana. Samo metapodatki, `du`, `find`, Git in `ollama list`. Brez brisanja, testov ali inferenc; vsebine velikih datotek in baz ocen niso bile odprte.

## Git

- Veja: `cleanup/florence-qwen`.
- Lokalni HEAD: `08448390589397a0228f41d59a851690d97d254e`.
- Oddaljeni `origin/cleanup/florence-qwen`: `08448390589397a0228f41d59a851690d97d254e` (potrjeno z `git ls-remote`; enak tudi lokalni remote-tracking SHA).
- Pred popisom: nobenih sprememb sledenih datotek ali pripravljenih sprememb; samo `?? vision-launch.log`. Ta datoteka ni vključena v commit.
- Ta commit vsebuje samo `maintenance/local-inventory.md`; navedena SHA opisujeta stanje pred njim.

## Poraba projekta

Celotno vidno drevo: **5.002.436.608 B (4.66 GiB)**, vključno z `.git`. Velikosti so dodeljeni bloki (`du -x -B1`), brez sledenja simbolnim povezavam. Ne vključujejo Docker volumna. Meritve so posnetek pred zapisom popisa; Git in dokument bosta po objavi nekoliko večja.

| Mapa v korenu | Velikost | Git/lokalni status |
| --- | ---: | --- |
| `results/` | 2.108.596.224 B (1.96 GiB) | Ignorirano; samo lokalno |
| `models/` | 1.573.060.608 B (1.47 GiB) | Ignorirano; samo lokalno |
| `.git/` | 516.698.112 B (492.76 MiB) | Interni Git podatki |
| `analiza rezultatov vmesna predstavitev/` | 316.784.640 B (302.11 MiB) | Git (5468 sledenih datotek; možni lokalni dodatki) |
| `results models/` | 312.008.704 B (297.55 MiB) | Git (6471 sledenih datotek; možni lokalni dodatki) |
| `data/` | 112.484.352 B (107.27 MiB) | Git (431 sledenih datotek; možni lokalni dodatki) |
| `reports/` | 54.308.864 B (51.79 MiB) | Git (237 sledenih datotek; možni lokalni dodatki) |
| `reviews/` | 2.904.064 B (2.77 MiB) | Ignorirano; samo lokalno |
| `maintenance/` | 1.740.800 B (1.66 MiB) | Git (77 sledenih datotek; možni lokalni dodatki) |
| `llm_object_classifier/` | 1.093.632 B (1.04 MiB) | Git (58 sledenih datotek; možni lokalni dodatki) |
| `scripts/` | 204.800 B (0.20 MiB) | Git (34 sledenih datotek; možni lokalni dodatki) |
| `tests/` | 204.800 B (0.20 MiB) | Git (24 sledenih datotek; možni lokalni dodatki) |
| `configs/` | 167.936 B (0.16 MiB) | Git (36 sledenih datotek; možni lokalni dodatki) |
| `docs/` | 53.248 B (0.05 MiB) | Git (5 sledenih datotek; možni lokalni dodatki) |
| `examples/` | 20.480 B (0.02 MiB) | Git (4 sledenih datotek; možni lokalni dodatki) |
| `docker/` | 16.384 B (0.02 MiB) | Git (3 sledenih datotek; možni lokalni dodatki) |
| `.github/` | 12.288 B (0.01 MiB) | Git (1 sledenih datotek; možni lokalni dodatki) |
| `.agents/` | 0 B (0.00 MiB) | 0 B v vidnem okolju; vsebina ni pregledovana |
| `.codex/` | 0 B (0.00 MiB) | 0 B v vidnem okolju; vsebina ni pregledovana |
| `.aws/` | 0 B (0.00 MiB) | 0 B v vidnem okolju; vsebina ni pregledovana |

Največje neposredne podmape zahtevanih večjih map:

| Nadrejena mapa | Največje podmape |
| --- | --- |
| `results/` | `results/thud-vmesno` — 846.999.552 B (807.76 MiB); `results/thud-part2` — 412.844.032 B (393.72 MiB); `results/loco` — 257.343.488 B (245.42 MiB); `results/ponovitve` — 221.859.840 B (211.58 MiB); `results/experiment200` — 136.421.376 B (130.10 MiB); `results/stage2-presentation` — 105.480.192 B (100.59 MiB) |
| `data/` | `data/thud_dense_test50` — 47.017.984 B (44.84 MiB); `data/thud_agtea_test50` — 42.336.256 B (40.38 MiB); `data/loco_test100` — 16.375.808 B (15.62 MiB); `data/loco_test30` — 5.750.784 B (5.48 MiB) |
| `models/` | `models/florence-2-large` — 1.556.283.392 B (1.45 GiB); `models/retired-metadata` — 16.773.120 B (16.00 MiB) |
| `reviews/` | `reviews/backups` — 57.344 B (0.05 MiB) |

### Največjih 20 lokalnih datotek ali map

Hierarhični seznam; nadrejene in podrejene velikosti se prekrivajo. Ne seštevajte jih. Celotni koren ni ponovno vključen.

| Pot glede na koren | Velikost | Status |
| --- | ---: | --- |
| `results` | 2.108.596.224 B (1.96 GiB) | Ignorirano; samo lokalno |
| `models` | 1.573.060.608 B (1.47 GiB) | Ignorirano; samo lokalno |
| `models/florence-2-large` | 1.556.283.392 B (1.45 GiB) | Ignorirano; samo lokalno |
| `models/florence-2-large/model.safetensors` | 1.553.567.744 B (1.45 GiB) | Ignorirano; samo lokalno |
| `results/thud-vmesno` | 846.999.552 B (807.76 MiB) | Ignorirano; samo lokalno |
| `.git` | 516.698.112 B (492.76 MiB) | Interni Git podatki |
| `.git/objects` | 514.052.096 B (490.24 MiB) | Interni Git podatki |
| `results/thud-part2` | 412.844.032 B (393.72 MiB) | Ignorirano; samo lokalno |
| `.git/objects/pack` | 377.835.520 B (360.33 MiB) | Interni Git podatki |
| `analiza rezultatov vmesna predstavitev` | 316.784.640 B (302.11 MiB) | Git (5468 sledenih datotek; možni lokalni dodatki) |
| `results models` | 312.008.704 B (297.55 MiB) | Git (6471 sledenih datotek; možni lokalni dodatki) |
| `results/loco` | 257.343.488 B (245.42 MiB) | Ignorirano; samo lokalno |
| `results/ponovitve` | 221.859.840 B (211.58 MiB) | Ignorirano; samo lokalno |
| `results/loco/llm` | 216.801.280 B (206.76 MiB) | Ignorirano; samo lokalno |
| `results/ponovitve/stage1` | 215.408.640 B (205.43 MiB) | Ignorirano; samo lokalno |
| `analiza rezultatov vmesna predstavitev/industrija LOCO` | 179.412.992 B (171.10 MiB) | Git (3242 sledenih datotek; možni lokalni dodatki) |
| `results models/florence_caption_grounding_test10` | 153.485.312 B (146.38 MiB) | Git (2281 sledenih datotek; možni lokalni dodatki) |
| `results/loco/llm/runs` | 142.622.720 B (136.02 MiB) | Ignorirano; samo lokalno |
| `results/experiment200` | 136.421.376 B (130.10 MiB) | Ignorirano; samo lokalno |
| `analiza rezultatov vmesna predstavitev/industrija LOCO/02_llm` | 128.512.000 B (122.56 MiB) | Git (1948 sledenih datotek; možni lokalni dodatki) |

## Aktivna konfiguracija in lokalni modeli

Poti iz `configs/pipeline.json` (obstoj preverjen, brez zagona):

| Nastavitev | Relativna pot / ime | Stanje |
| --- | --- | --- |
| `input` | `data/input` | ne obstaja; privzeta predloga, ob ustvaritvi ignorirano |
| `output` | `results/pipeline/run` | ne obstaja; privzeta predloga, ob ustvaritvi ignorirano |
| `florence.model_dir` | `models/florence-2-large` | obstaja; Ignorirano; samo lokalno |
| `llm.system_file` | `configs/prompts/reasoning-evidence-system.txt` | obstaja; Git |
| `llm.instruction_file` | `configs/prompts/reasoning.txt` | obstaja; Git |

- Florence: `microsoft/Florence-2-large`, revizija `21a599d414c4d928c9032694c424fb94458e3594`; **uteži samo lokalno, ignorirane, potrebne za aktivno pot**.
- Izbrani besedilni model: `qwen3.5:27b-q4_K_M`; Gemma je izbirna alternativa. `base_url=auto` uporabi projektni vsebnik; ne gre za dodatno datotečno pot.

Projektna Ollama `klasifikacija-ollama-1`, samo `ollama list`:

| Tag | ID | Velikost po Ollami |
| --- | --- | ---: |
| `qwen3.5:27b-q4_K_M` | `64ae5b1d651f` | 17 GB |
| `gemma3:27b-it-q4_K_M` | `6be2e2ba6843` | 17 GB |

Dejanska poraba modelov: **34.892.128.256 B (32.50 GiB)** v projektnem Docker volumnu `klasifikacija_ollama-models`. Ta volumen je zunaj projektnega drevesa in ni v Gitu. Tagi manjših Qwenov, BF16 in Gemme 4B niso več prisotni. Delujoči Florence Docker image je konfiguriran kot `thud-florence:local`; njegovih plasti nismo merili.

## Prejšnje čiščenje: trenutno stanje poti

Preverjeno po `maintenance/cleanup-florence-qwen.json`; nič ni bilo ponovno odstranjeno.

| Pot iz zapisnika | Trenutno stanje |
| --- | --- |
| `results/thud-extended/models/nvila` | Ni prisotno. |
| `results/stage1-presentation/models/qwen3-vl` | Ni prisotno. |
| `results/lilocbench/cache` | Ni prisotno. |
| `results/lilocbench/indices` | Ni prisotno. |
| `llm_object_classifier/__pycache__` | Obstaja, 241.664 B (0.23 MiB); Ignorirano; samo lokalno. Ponovno ustvarjen Python predpomnilnik. |
| `scripts/__pycache__` | Ni prisotno. |
| `tests/__pycache__` | Ni prisotno. |
| `results/lilocbench/prepare.log` | Ni prisotno. |
| `results/lilocbench/selection.log` | Ni prisotno. |
| `results/lilocbench/florence.log` | Ni prisotno. |
| `results/lilocbench/text.log` | Ni prisotno. |
| `results/lilocbench/progress.md` | Ni prisotno. |

Neutežna arhivska metadata sta še prisotna: `models/retired-metadata/nvila` (5.160.960 B (4.92 MiB)) in `models/retired-metadata/qwen3-vl` (11.608.064 B (11.07 MiB)); oboje ignorirano in samo lokalno. Del majhnih konfiguracij je ločeno ohranjen v Gitu pod `maintenance/retired-model-configs/`; enakovrednost vseh datotek ni preverjena.

## Simbolne povezave

Preverjenih je vseh šest povezav v vidnem drevesu. Cilji obstajajo; noben zunanji cilj ni bil pregledovan.

| Povezava | Razrešen cilj | Zunaj projekta? |
| --- | --- | --- |
| `results/thud-vmesno/nvila-diagnostic/venv/lib64` | `results/thud-vmesno/nvila-diagnostic/venv/lib` | Ne |
| `results/thud-vmesno/nvila-diagnostic/venv/bin/python` | Sistemski Python interpreter (absolutna pot namenoma izpuščena) | Da |
| `results/thud-vmesno/nvila-diagnostic/venv/bin/python3.14` | Sistemski Python interpreter (absolutna pot namenoma izpuščena) | Da |
| `results/thud-vmesno/nvila-diagnostic/venv/bin/python3` | Sistemski Python interpreter (absolutna pot namenoma izpuščena) | Da |
| `results/thud-vmesno/nvila-diagnostic/venv/bin/𝜋thon` | Sistemski Python interpreter (absolutna pot namenoma izpuščena) | Da |
| `results/thud-extended/models/florence` | `models/florence-2-large` | Ne |

## Samo lokalne ocene, rezultati in uteži

- **Ročne ocene: ohraniti.** `reviews/` je v celoti ignoriran; baze niso v Gitu. Njihova vsebina ni bila odprta.

| Datoteka ocen | Velikost (`du`) | Status |
| --- | ---: | --- |
| `reviews/backups/manual_review-before-model-addition-20261104T185013748012Z.sqlite3` | 53.248 B (0.05 MiB) | Ignorirano; zasebno, ohraniti |
| `reviews/manual_review.sqlite3` | 2.809.856 B (2.68 MiB) | Ignorirano; zasebno, ohraniti |
| `reviews/manual_review.sqlite3-shm` | 32.768 B (0.03 MiB) | Ignorirano; zasebno, ohraniti |
| `reviews/manual_review.sqlite3-wal` | 0 B (0.00 MiB) | Ignorirano; zasebno, ohraniti |

- **Lokalni rezultati:** celotna `results/` je ignorirana in ni sledena v Gitu. To ne dokazuje, da so vse datoteke edinstvene: nekatere objavljene kopije so v `results models/`, `analiza rezultatov vmesna predstavitev/`, `reports/` in `maintenance/smoke-test/`. Enakovrednost in povezave niso preverjene; izvirnike ohraniti.
- **Lokalni izbrani podatki:** `data/nyu20` je ignoriran in nima sledenih datotek; obseg 995.328 B (0.95 MiB). Namen in povezanost s starimi poročili je še treba preveriti. Ostali večji podnabori `data/` imajo datoteke v Gitu.
- **Uteži:** aktivna Florence in oba modela v Ollama volumnu so samo lokalni. Git vsebuje konfiguracije/revizije, ne uteži. Ohraniti.

## Možni kandidati za poznejše preverjanje čiščenja

**To ni dovoljenje za brisanje.** Ni bilo preverjanja vsebine, enakovrednosti kopij ali povezav vseh izvozov s poročili. Nejasne datoteke ostajajo.

| Kandidat | Velikost | Status | Razlog / naslednje preverjanje |
| --- | ---: | --- | --- |
| `llm_object_classifier/__pycache__` | 241.664 B (0.23 MiB) | Ignorirano; samo lokalno | Ponovljiv Python predpomnilnik, ponovno ustvarjen po prejšnjem čiščenju. |
| `results/experiment200/source-coco.json` | 103.526.400 B (98.73 MiB) | Ignorirano; samo lokalno | Ni naveden v aktivni konfiguraciji. Izvor, ponovljivost prenosa in potrebo pri reprodukciji je še treba preveriti; vsebina ni odprta. |
| `results/pipeline-smoke-cleanup` | 1.937.408 B (1.85 MiB) | Ignorirano; samo lokalno | Začasni enoslikovni preizkus; kopije so tudi v maintenance/smoke-test. Pred brisanjem je treba preveriti njihovo popolnost in enakovrednost. |
| `results/vision-caption-cache` | 327.680 B (0.31 MiB) | Ignorirano; samo lokalno | Ime nakazuje predpomnilnik; njegov namen in morebitne edinstvene opise je še treba preveriti. |
| `results/thud-vmesno/nvila-diagnostic/venv` | 32.768 B (0.03 MiB) | Ignorirano; samo lokalno | Opuščeno diagnostično okolje; samo 32 KiB. Preveriti arhivske odvisnosti in zunanje Python povezave. |
| `models/retired-metadata` | 16.773.120 B (16.00 MiB) | Ignorirano; samo lokalno | Neutežni podatki odstranjenih modelov; preveriti, ali lokalno prilagojene konfiguracije/tokenizerji obstajajo samo tu. Za zdaj ohraniti. |
| `vision-launch.log` | 98.304 B (0.09 MiB) | Lokalno, nesledeno | Nesleden lokalni dnevnik. Namen in morebitne edinstvene diagnostične podatke je še treba preveriti; vsebina ni odprta. |
| `results/thud-vmesno/**/comparison.html` in `results/thud-part2/**/comparison.html` | 1.124.401.152 B (1.05 GiB) skupaj (51 datotek) | Ignorirano; samo lokalno | Veliki pregledni izvozi; pred morebitno obnovitvijo ali brisanjem je treba preveriti generatorje, poročilne povezave in ohranitev surovih odgovorov. |

Vse ostale lokalne eksperimentalne mape `results/` ostajajo zaradi nepreverjene edinstvenosti. `.git`, objavljeni rezultati/podatki, aktivne uteži, `reviews/`, izvozi ocen in varnostne kopije niso predlagani za brisanje. V tem opravilu ni bilo nobenega čiščenja.

## Dopolnitev 2026-11-05: omejeno arhiviranje HTML in odstranitev predpomnilnika

Izhodišče: veja `cleanup/florence-qwen`, commit `41c200be30dd4eaa1cd1ecd83cb7669b9ee79bf9`. Ta dopolnitev opisuje novo, izrecno dovoljeno čiščenje; zgornji popis ostaja zgodovinski posnetek pred njim.

- Najdenih je bilo 51 navadnih datotek z natančnim imenom `comparison.html` pod `results/thud-vmesno/` in `results/thud-part2/`; simbolnim povezavam nismo sledili.
- Arhiviranih in nato odstranjenih: **51 HTML datotek**. Sledenih ali poročilno odvisnih izvozov ni bilo v izboru; izločenih/nejasnih: 0.
- Odstranjen predpomnilnik: `llm_object_classifier/__pycache__` (241.664 B dodeljenih blokov). Pot je bila preverjena kot nesledena običajna lokalna mapa znotraj projekta.

### Preverjene odvisnosti

Pregledanih je bilo 138 ohranjenih poročilnih/dokumentacijskih datotek ter URL-ji obeh PDF poročil. Aktivni `llm_object_classifier/review_viewer.py` bere manifeste, JSON in slike; teh starih HTML izvozov ne nalaga. `stage1_describe.save_tables` jih samo ustvarja. Generatorja THUD poročil bereta JSON in ustvarita ločena `manual-review.html`; ta sta ohranjena. Generične navedbe korenskih map surovih rezultatov niso sklici na izbrane `comparison.html`.

Relevantne skripte/moduli (samo pregled, brez zaganjanja): `llm_object_classifier/review_viewer.py`, `llm_object_classifier/pipeline.py`, `llm_object_classifier/stage1_describe.py`, `llm_object_classifier/description_report.py`, `llm_object_classifier/thud_experiment.py`, `llm_object_classifier/thud_part2.py`, `llm_object_classifier/thud_report.py`, `llm_object_classifier/thud_part2_report.py`, `llm_object_classifier/thud_extended_report.py`, `scripts/thud_experiment.sh`, `scripts/thud_part2.sh`, `scripts/compile_thud_report.sh`, `scripts/thud_status.py`, `scripts/thud_ponovitve.py`.

### Lokalni arhiv zunaj repozitorija

- Pot glede na koren projekta: `../Klasifikacija-to-k-archives/comparison-html-2026-11-05-1793872843792162011.tar.xz`.
- Format: en `tar.xz`; izvorne relativne poti so ohranjene. Vsebuje 51 HTML datotek in `ARCHIVE-MANIFEST.json` (52 članov skupaj). Arhiv ni dodan v Git.
- Velikost arhiva: 82.702.100 B (78.87 MiB); dejansko dodeljeno: 82.706.432 B (78.88 MiB).
- SHA256 arhiva: `df3fc5e0711d82d80ff50a416907737f20081bb0f2e6adc12b779244dc2c5c70`.
- Celoten arhiv je bil ponovno prebran; preverjeni so bili množica članov, tipi, dolžine, manifest in SHA256 vseh HTML datotek. Nato so bile ponovno preverjene kontrolne vsote in identitete izvirnikov. Šele po uspehu vseh preverjanj so bili izvirniki odstranjeni. Njihova vsebina se je obravnavala samo kot bajti za arhiviranje/kontrolne vsote, brez pregleda odgovorov.

### Dejanski prihranek prostora

| Meritev | Dodeljeni prostor |
| --- | ---: |
| Odstranjeni HTML izvorniki | 1.124.401.152 B (1072.31 MiB) |
| Odstranjeni predpomnilnik | 241.664 B (0.23 MiB) |
| Izmerjeno zmanjšanje dveh izvornih dreves in predpomnilnika | 1.124.642.816 B (1072.54 MiB) |
| Rast zunanje arhivske mape, skupaj z arhivom in režijo imenika | 82.710.528 B (78.88 MiB) |
| **Neto sproščeno po odštetju arhiva in režije** | **1.041.932.288 B (993.66 MiB)** |

Izračun uporablja `du -x -B1` pred/po posegu in razliko dodeljenih blokov zunanje arhivske mape. Ne uporablja seštevanja navideznih velikosti ali spremembe prostega prostora celotnega diska, ki bi vključevala druge procese. Kasnejša majhna rast zapisnika/Git objektov ni del meritve čiščenja.

Metapodatki preostalih 29.171 datotek v projektu so ostali nespremenjeni. Vsebina surovih rezultatov, slik, uteži in ocen ni bila pregledovana; kode in poročila smo brali samo zaradi odvisnosti. Ohranjeni so vsi drugi rezultati in sami eksperimentalni imeniki, slike, manifesti, surovi odgovori, LILocBench, `results/experiment200/source-coco.json`, `models/florence-2-large`, `reviews/`, Docker okolja/volumni ter oba 27B modela. Modelov in testov nismo zaganjali. Pipeline in filtri niso spremenjeni. V commit je vključena samo ta dopolnitev zapisnika.
