# Florence → Qwen: opisovanje in razvrščanje regij

Aktivna pot je **slika → Florence-2 MORE_DETAILED_CAPTION → celoten opis v
CAPTION_TO_PHRASE_GROUNDING → Qwen3.5 27B Q4_K_M → fixed/movable/uncertain z razlogom**.
Privzeto se izvede samo Qwen. Gemma 3 27B Q4_K_M je izbirna alternativa.
Regije so fraze, lahko skupine ali prekrivajoči deli; niso potrjene instance.
Veljavnost zapisa ni pravilnost klasifikacije.

## Okolje in namestitev

Na tem računalniku so že pripravljeni `models/florence-2-large`, slika Docker
`thud-florence:local` in GPU Ollama `klasifikacija-ollama-1`. Ohranjena taga sta
`qwen3.5:27b-q4_K_M` in `gemma3:27b-it-q4_K_M`. Ni samodejnega prenosa modelov.
Python gostitelja potrebuje le Pillow; PyTorch/Transformers tečeta v obstoječem
Florence vsebniku (CUDA 12.8, podpora RTX 5090). Za ločeno lokalno Python okolje:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Nadaljnje ukaze lahko izvedete z `.venv/bin/python` namesto `python3`.
Če projektno Ollamo ustavite, jo ponovno vključite z
`docker compose -f compose.yaml -f compose.gpu.yaml up -d ollama`.
Na drugem računalniku je treba najprej pripraviti iste lokalne uteži in okolje;
[arhiv okolij](README_DOCKER.md) ter [Florence Dockerfile](docker/thud-florence.Dockerfile)
ohranjata podrobnosti. Za nov projekt izberite veljavno pot uteži v konfiguraciji.

## Konfiguracija in en zagon

Vse aktivne nastavitve so v **[configs/pipeline.json](configs/pipeline.json)**.
Poti v JSON so glede na koren projekta; `--input` in `--output` sta glede na trenutni imenik.
Spreminjate model, poti, prompt, temperaturo, seme, kontekst, izhodni proračun in
Florence nastavitve. `llm.selected` je `qwen` ali `gemma`. Privzeti evidence-based
prompt v `configs/prompts/reasoning-evidence-system.txt` je dobesedno enak novemu
promptu objavljene primerjave 40 slik. `reasoning.txt` vsebuje uporabniško navodilo.

Iz korena projekta:

```bash
python3 -m llm_object_classifier.pipeline --config configs/pipeline.json --input "/pot/do/slik" --output "results/moj-zagon"
```

Podmape slik so vključene; podprti so JPG/PNG/WebP/BMP/TIFF. Podvojene SHA256
zavrne. Izhod mora biti zunaj vhodne mape. Kopije originalov dobijo nevtralne ID-je.
Referenčnih oznak ne bere. Za nadaljevanje ponovite **isti ukaz**: preverjeni
zaključeni klici, tudi napake/preskoki, se ne ponovijo. Spremenjen podpis zahteva novo
izhodno mapo. Kode, promptov in nastavitev vmes ne spreminjajte.

Isti ukaz s `--stage florence` izvede samo oba slikovna klica.
`--stage llm --output results/moj-zagon` uporabi shranjene Florence rezultate.
`--stage export --output results/moj-zagon` samo obnovi JSON/CSV/HTML, brez GPU,
Dockerja, uteži in inferenc. Pri izbranem modelu morajo nastavitve ustrezati zagonu.

Za Gemmo na istem shranjenem Florence koraku, brez ponovnega opisovanja:

```bash
python3 -m llm_object_classifier.pipeline --config configs/pipeline.json --stage llm --model gemma --florence-from results/moj-zagon --output results/moj-zagon-gemma
```

Kontekst je privzeto 8192, največji izhod 2048, T=0.3, seme 42. Native tokenizer
prešteje vsebino; dodatnih 256 tokenov je označena konservativna ocena ovojnice.
Nezadosten/nepreverljiv proračun povzroči zabeležen preskok, ne krajšanja vhoda.
Po klicu se preverijo dejanski tokeni. Celoten GPU, natančnost Q4_K_M in digest
se preverijo pred generacijo; CPU preklopa ni. Zaključek, JSON, točni ID-ji,
podvojitve, razredi in neprazen `reason` se preverijo s skupnim validatorjem.

## Izhodi in ročni pregled

- `run.json`, `source/`, okolja in metapodatki: konfiguracija, dejanske različice,
  uteži/digesti, izvorne kode ter SHA slik.
- `images/`, `native/`: originali in celotni opis/grounding s tokeni, časi, VRAM in napakami.
- `frozen/`, `prompts/`: nespremenljivi vhodi in dejanski LLM prompti. `not_visible`
  pomeni manjkajoč dokaz pritrditve; ne pomeni, da je objekt nepritrjen.
- `responses/`, `errors/`: nespremenjeni odgovori, tehnična veljavnost in napake.
- `annotated/`, `index.html`, `regions.csv`, `results.json`, `summary.json`: izris
  vrnjenih okvirjev, celotna besedila, razredi/razlogi in tehnični števci.

`index.html` odprete neposredno v brskalniku. Ročno ocenjevanje:

```bash
python3 -m llm_object_classifier.review_viewer --results-dir results/moj-zagon
```

Privzeto dobi vsak nov nabor svojo bazo `reviews/pipeline/<ID-nabora>.sqlite3`.
Ena referenca velja za vse povezane odgovore; pregled razlogov je neobvezen.
Stari nabori, njihove anonimne preslikave, reference, baze in varnostne kopije so
združljivi in nedotaknjeni. Ocene ostajajo izključene iz Gita; podrobnosti so v
[README_REVIEW_VIEWER.md](README_REVIEW_VIEWER.md).

## Arhiv in preverjanje

Stara navodila so v [README_EXPERIMENTS_ARCHIVE.md](README_EXPERIMENTS_ARCHIVE.md),
pregled arhivskih izvajalnikov v [docs/ARCHIVE.md](docs/ARCHIVE.md).
Objavljeni rezultati, izbrani podatki, kalibracije, prompti in surovi zapisi ostajajo.
Pred čiščenjem je narejen Git checkpoint `checkpoint/before-florence-qwen-cleanup`.
Ciljni zapis odstranitev in dejansko sproščenih blokov je
[maintenance/cleanup-florence-qwen.json](maintenance/cleanup-florence-qwen.json).

Testi brez inferenc in brez dejanskih baz ocen:

```bash
python3 -B -m unittest tests.test_pipeline tests.test_llm_checks tests.test_review_viewer tests.test_lilocbench -q
```
