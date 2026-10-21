# Vmesno poročilo THUD (1. del naloge)

Poročilo je osnutek za poznejši ročni pregled. Prvih šest strani opisuje 1. del;
nadaljevanje 2. dela je ločen dodatek, ustvarjen iz novih shranjenih rezultatov.

Celoten ponovljiv zagon (potrebuje GPU Docker, tri nameščene tage in Python s Pillow/Matplotlib):

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
docker compose exec -T ollama ollama pull qwen3.5:4b-bf16
docker compose exec -T ollama ollama pull qwen3.5:4b-q4_K_M
docker compose exec -T ollama ollama pull gemma3:4b-it-q4_K_M
bash scripts/thud_experiment.sh --phase all
```

Zaganjalnik najprej preveri podatke in izvede pilot (2 sliki × 2 nabora × 2 načina × 3 modeli = 24 merjenih klicev); shrani linearno oceno pred osnovnim zagonom. Osnova ima 600 merjenih klicev in 12 ločenih ogrevalnih klicev. Dodatni prvi korak ima 80 klicev; dodatni drugi korak 120. Drugi korak ogreva ločeno. Osnovni podrobni T=0 opis se uporabi ponovno na istih 20 slikah. Seed 43 se doda samo pri T=.3. Modeli se izvajajo zaporedno.

Faze lahko poganjaš ločeno z `--phase prepare|pilot|baseline|additional`; `all` na koncu ustvari in prevede poročilo. Pri neposrednem Python zagonu nastavi `--base-url`; lupinski zaganjalnik naslov prebere iz projektnega kontejnerja. Napačen GPU/CPU offload ustavi izvajanje; odprava izgubljenega GPU je v `docs/WORKFLOW.md`.

Vsaka skupina ima nespremenljivo konfiguracijo in `complete.json` tudi ob neveljavnih odgovorih. Ponavljanje dokončanih skupin ne sproži novih klicev. Prekinjeni stage1 nadaljuje le manjkajoče odgovore in ohrani tudi napake. Pri prekinjenem stage2 se ohrani celotna stara mapa in ustvari ločeno `attemptN`; `resumed-attempt.json` določi dejanski zadnji poskus. Spremenjen vhod ali konfiguracija se zavrne; nov poskus potrebuje novo izhodno korensko mapo v kodi. Dveh zaganjalnikov hkrati ne poganjaj.

Samostojna ponovna generacija poročila iz shranjenih odgovorov:

```bash
python3 -m llm_object_classifier.thud_verify --require-complete
python3 -m llm_object_classifier.thud_report
bash scripts/compile_thud_report.sh
```

Prevajalnik je izoliran v `Dockerfile.latex`, če lokalni `pdflatex` manjka. `porocilo.tex`, `porocilo.pdf`, slike, agregati CSV/JSON in revizija podatkov tvorijo samostojen paket. Vrednosti so vedno izračunane iz odgovorov, ne ročno prepisane. Za generiranje je poleg osnovnih odvisnosti potreben Matplotlib (`requirements-report.txt`).

Veliki surovi izhodi, manifesti s prompti in metapodatki, zamrznjeni opisi, ogrevanja in dnevniki ostanejo lokalno v `results/thud-vmesno/`. Samostojni HTML za istih 20 slik je `results/thud-vmesno/manual-review.html`; `manual-review.csv` ima prazna ročna polja in status `pending`. Regeneracija združi ročna polja po ključu SHA256 slike/model/način in ohrani urejene vrednosti. Izbrani vizualni komentarji so posebej v `manual-notes.json`; ne predstavljajo sistematične anotacije kakovosti/stabilnosti.

Prekrivanje imen: male črke, normalizirani presledki, presek unikatnih imen. Brez sinonimov, instance matching, IoU ali razredov stabilnosti. To ni precision/recall/F1/mAP/accuracy. Neuspehi ostanejo v imenovalcih; primerjave odločitev uporabljajo le pare veljavnih odgovorov istega SHA256 opisa in tipiziranega ID.

`verification.json` preveri vseh 32 skupin osnovnih/dodatnih poskusov, enake slikovne SHA256 med modeli in načinoma, nespremenjene opise, odsotnost slike v drugem koraku in GPU dodelitve. Napaka modelove sheme je merjeni izid, ne neuspeh revizije.

Sintaktično berljivi, a po shemi zavrnjeni odgovori imajo ločene razčlenjene kandidate v `results/thud-vmesno/rejected-candidates.jsonl`; indeks je `rejected-index.json`. Prvotni odgovori ostanejo nespremenjeni, status `error` se ohrani in kandidatov drugi korak ne uporabi.

Dodatna izvedena Q4 ponovitev drugega koraka (20 klicev + 2 ogrevanji, isti zamrznjeni opisi, dolgi prompt brez primerov, T=.3, seed 42):

```bash
python3 -m llm_object_classifier.thud_q4_repeat --base-url http://172.18.0.2:11434
python3 -m llm_object_classifier.thud_report
bash scripts/compile_thud_report.sh
```

Naslov po potrebi preberi z `docker inspect` projektne Ollame. Revizija pri prisotnih Q4 skupinah preveri tudi ti dve dodatni skupini (skupaj 34 skupin, 820 merjenih odgovorov brez pilota).

## 2. del: bogatejši samostojni opis

```bash
bash scripts/thud_part2.sh --phase all
```

Zagon uporabi že nameščene tri iste modele in nespremenjene slikovne manifeste.
Faze `prepare|pilot|describe|reason|all` imajo ločene rezultate v
`results/thud-part2/`. Dokončane skupine in posamezni odgovori se ponovno uporabijo,
tudi ob zavrnitvi. Spremenjen prompt ali vhod se zavrne; nova različica zahteva
ločen poskus. Prvi neuspešni pilot je ohranjen v `diagnostic-pilot-v1/`, dopolnitev
promptov pa pojasnjena v `part2-pilot-revision.json`.

`configs/stage1/thud-part2.json` in `thud-description-rich.txt` določata shemo
`rich-description-v2`: poleg starih polj še videz, položaj v sliki, zakritost,
podpore/stike, vidne dokaze namestitve in odnose med lokalnimi ID-ji. Besedilna
polja uporabljajo `unknown`, če podatek ni razviden. Tipovna JSON predloga nima
vsebinskih primerov objektov. Napovedanih okvirjev ni. Stari opisi ostanejo
združljivi; stara izhodna drevesa se ne spreminjajo.

Vseh 100 slik se uporabi za nove opise treh modelov. Drugi korak uporabi isti
podnabor 20 slik, BF16 razvrščevalnik in obstoječi kratki prompt brez primerov pri
T=.3, seed42. Vsi strukturirani dokazi in odnosi se prenesejo v celoti; slike se
ne odpirajo. Starih 20 BF16 presoj se uporabi ponovno, manjkajočih 40 kontrol na
starih opisih Q4/Gemma se doda enkrat. Zavrnjeni opisi ostanejo eksplicitni preskoki.

Pregled: `python3 scripts/thud_part2_status.py`. Lokalni
`results/thud-part2/manual-review.html` prikaže stare/nove opise, slike,
preklopne referenčne okvirje, odnose in odločitve. Sistematična ročna polja v
`part2-manual-review.csv` so osnutek; delne vizualne pripombe so posebej označene.
Enako ime ni potrjeno ujemanje instance, skupinske regije pa niso število predmetov.

Poročilo iz obstoječih odgovorov, brez GPU klicev:

```bash
python3 -m llm_object_classifier.thud_part2_report --require-complete
python3 -m llm_object_classifier.thud_report
bash scripts/compile_thud_report.sh
```

Dodatek `part2.tex` je vključen v glavni PDF/LaTeX. Tabele `part2-aggregates.*`,
`part2-image-comparisons.*`, `part2-decision-pairs.*`, `part2-summary.json` in
`part2-verification.json` se izračunajo iz odgovorov. Preverjanje vključuje enake
slikovne bajte, celotno besedilo dokazov/odnosov, SHA256 zamrznjenih opisov,
GPU dodelitve in nespremenjenost vseh starih odgovorov iz lokalnega indeksa.
Več odnosov ali veljavna shema ne pomenita pravilnosti; brez stabilnostnih
referenc ni accuracy, brez napovedanih okvirjev ni IoU/mAP.
