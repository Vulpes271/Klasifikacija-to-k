# Vmesno poročilo THUD (1. del naloge)

Poročilo je osnutek za poznejši ročni pregled. Drugi del naloge se ne izvaja.

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
