# Nastavitve: slika -> opis -> reasoning

> Starejša, ohranjena pot. Za nova ločena koraka glej [docs/WORKFLOW.md](docs/WORKFLOW.md).


Koda temelji na veji `feature/docker-smoke`. Prvi del prejme sliko in opise
vidne objekte. Drugi del prejme oznake in opis iz prvega dela ter vrne
`fixed`, `movable` ali `uncertain`. Modela lahko izberes loceno.

## Hiter zagon na Linuxu

Uporabi obstojeci Linux racunalnik z NVIDIA GPU, Docker Compose in NVIDIA
Container Toolkit, kjer je dosedanji postopek ze deloval. Podatki so privzeto
v `data/nyu20/objects.jsonl`; priprava je opisana v [README_GPU.md](README_GPU.md).

Iz korena repozitorija:

```bash
cp configs/vision-pipeline.example.json configs/my-vision.json
# Uredi configs/my-vision.json in configs/prompts/reasoning*.txt.
bash scripts/test_vision_pipeline.sh moj-poskus --config configs/my-vision.json
```

Skripta zgradi aplikacijo, pozene teste, prenese oba izbrana modela in preveri
uporabo GPU. Vsak poskus potrebuje novo ime. Rezultati so v
`results/moj-poskus`, celoten dnevnik v `results/moj-poskus.log`.
Privzeti zagon `bash scripts/test_vision_pipeline.sh ime` se vedno deluje.

## Kaj spreminjati

| Nastavitev v JSON | Pomen |
| --- | --- |
| `vision_model` | Model za opis slike; mora podpirati vision v Ollami. |
| `reason_model` | Model za razvrscanje na podlagi besedilnega opisa. |
| `caption_detail` | `brief`, `standard` ali `detailed`. |
| `caption_temperature` | Temperatura prvega dela. |
| `temperatures` | Seznam temperatur drugega dela, npr. `[0.2]` ali `[0, 0.3, 0.7]`. |
| `reason_prompt_file` | Besedilna datoteka z navodili za drugi del. |
| `reason_system_file` | Celoten sistemski prompt drugega dela: pravila, razredi, oblika JSON. |
| `caption_num_predict` | Najvec izhodnih tokenov za opis slike. |
| `num_predict` | Najvec izhodnih tokenov za reasoning; tudi za opis, ce `caption_num_predict` ni podan. |
| `num_ctx` | Velikost konteksta za oba dela. |
| `seed` | Seme za oba dela; `seeds` lahko poda vec semen samo za reasoning. |
| `profiles` | Neobvezna pot do profilov, kot je `vision-sweep.json`; velja samo za reasoning. |
| `limit` | Najvecje stevilo slik. |
| `require_gpu` | `true` zahteva preverjeno uporabo GPU. |

Podrobnost opisa nastavlja naslednja navodila modelu:

| `caption_detail` | Opis scene | Opazanje za vsak objekt |
| --- | --- | --- |
| `brief` | Najvec 35 besed | Najvec 25 besed |
| `standard` | Najvec 80 besed | Najvec 60 besed |
| `detailed` | Najvec 160 besed | Najvec 120 besed |

To so navodila modelu, ne trdo rezanje besedila. Pri daljsih opisih in vec
objektih povecaj tudi `caption_num_predict`, npr. na `4096`. Vec tokenov in
vecji kontekst povecata porabo pomnilnika in cas izvajanja.

Poti znotraj konfiguracije so relativne glede na mapo te konfiguracije.
V `configs/my-vision.json` zato `../data/nyu20/objects.jsonl` pomeni
projektno mapo `data/nyu20/objects.jsonl`. Poti na ukazni vrstici so relativne
glede na trenutno delovno mapo. Znotraj Dockerja uporabi `/app` poti ali
projektne relativne poti; konfiguracije in prompte hrani v `configs`, ki je
priklopljen v kontejner. Za poljuben zunanji Ollama endpoint uporabi neposredni
Python zagon spodaj; pomocna skripta upravlja projektno Ollamo.

## Spreminjanje prompta za drugi del

`configs/prompts/reasoning.txt` vsebuje navodilo, ki se poslje kot uporabniski
prompt. Dodaj npr. poudarek na vidnih dokazih ali zahtevaj doloceno vrsto
utemeljitve. Program sam doda dejanske oznake, ID-je in opazanja iz prvega
dela, zato v datoteko ni treba vstavljati predlog ali JSON podatkov.

`configs/prompts/reasoning-system.txt` vsebuje celotna pravila razvrscanja.
Za spremembo definicij razredov, primerov ali dolzine razloga uredi to
datoteko. Uporabniska navodila naj bodo skladna s sistemskimi pravili.
Ohrani izhodni JSON `{"objects": [{"id": 1, "classification": "fixed",
"reason": "..."}]}`, vse vhodne ID-je in dovoljene razrede.

Obe datoteki sta UTF-8. V konfiguraciji lahko navedes tudi svoji kopiji.
Ce nastavitvi `reason_prompt_file` in `reason_system_file` izpustis, se
uporabita dosedanja vgrajena prompta. Prazna ali manjkajoca datoteka javi
napako pred klici modelov. Vsebina se prebere enkrat na zacetku poskusa in
shrani v `manifest.json`; spremembe datoteke med zagonom veljajo sele naslednjic.

## Hitre spremembe brez urejanja konfiguracije

Argumenti na ukazni vrstici imajo prednost pred JSON nastavitvami:

```bash
bash scripts/test_vision_pipeline.sh podroben-poskus \
  --config configs/my-vision.json \
  --caption-detail detailed --caption-num-predict 4096 \
  --caption-temperature 0.1 --temperatures 0.2 0.5 --limit 2
```

`--vision-model` in `--reason-model` prepiseta izbiro modelov. Skripta prenese
tudi tako izbrana modela. Ime modela mora biti dejansko razpolozljiv tag v
Ollami. V skripti je izhodna mapa dolocena z imenom poskusa in GPU je obvezen.

Neposredno izvajanje v ze pripravljeni Docker namestitvi:

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
docker compose run --rm --no-deps --entrypoint python app \
  -u -m llm_object_classifier.vision_pipeline \
  --config configs/my-vision.json --output-dir results/neposredni-poskus
```

Za izvajanje brez Dockerja, ob ze zagnani lokalni Ollami in prenesenih modelih:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m llm_object_classifier.vision_pipeline \
  --config configs/my-vision.json --base-url http://127.0.0.1:11434 \
  --output-dir results/lokalni-poskus
```

## Predpomnilnik in preverjanje

Sprememba podrobnosti, opisovalnega modela, temperature opisa ali njegovega
token limita ustvari nov opis. Sprememba samo reasoning prompta, modela ali
temperature ponovno uporabi isti opis. To omogoca primerjavo drugega dela
na enakih vhodih. `manifest.json` shrani dejanske prompte, modele in nastavitve;
`caption.json` in `reasoning-*.json` vsebujeta odgovore, `results.csv` pa tabelo.
Koda 2 pomeni, da je vsaj en odgovor odpovedal ali bil izpuscen.

```bash
python -m unittest discover -s tests -v
```

Testi pokrivajo prioriteto nastavitev, izbiro modelov, oba prompta,
podrobnost opisa, loceni temperaturi in token limita, predpomnilnik ter
dejanske HTTP zahteve proti simuliranemu lokalnemu strezniku. Obstojeci
testi preverjajo tudi format odgovorov, GPU napake in temperaturne primerjave.
Test Bash skripte uporabi simulirana ukaza Docker in nvidia-smi ter preveri
prenos izbranih modelov in pravilno posredovanje argumentov.
GitHub Actions ob pushu/PR izvede teste na Ubuntu s Pythonom 3.11 in preveri
sintakso Bash skript. Dejanska inferenca na NVIDIA GPU ostaja del Linux zagona.
