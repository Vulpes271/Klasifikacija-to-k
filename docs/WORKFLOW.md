# Linux: nova baza, primerjava opisov, nato razvrščanje

Potrebuješ delujoč Docker Compose v2, NVIDIA Container Toolkit in gonilnik. Na tvojem
laboratorijskem računalniku je BF16 na GPU že deloval. Modeli se izvajajo zaporedno.
Ta projekt ne spreminja nastavitev gonilnikov ali globalnega Docker daemona.

## 1. Pridobi kodo

```bash
cd ~/jakob/Klasifikacija-to-k
git fetch origin
git switch feature/separate-vision-stages
git pull --ff-only origin feature/separate-vision-stages
```

Če Git zaradi lokalnih sprememb zavrne preklop, jih najprej shrani; ne uporabljaj
`reset --hard` ali brisanja map z rezultati.

## 2. Pripravljena izbora THUD (zunanji disk ni potreben)

Oba izbora s po 50 slikami in pripadajočimi oznakami sta vključena v Git.
Za trenutne poskuse lahko preskočiš razdelek o zunanjem disku.

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
bash scripts/stage1_describe.sh thud-store-hitri \
  --config configs/stage1/thud-compare.json \
  --input data/thud_dense_test50/objects.jsonl --limit 2
xdg-open results/stage1/thud-store-hitri/comparison.html
```

Za oba celotna izbora:

```bash
bash scripts/stage1_describe.sh thud-store-50 \
  --config configs/stage1/thud-compare.json \
  --input data/thud_dense_test50/objects.jsonl --limit 50
bash scripts/stage1_describe.sh thud-agtea-50 \
  --config configs/stage1/thud-compare.json \
  --input data/thud_agtea_test50/objects.jsonl --limit 50
```

Vsak nabor se primerja ločeno. Lokalni frame_id se lahko med naboroma ponovi,
zato njunih JSONL ne združuj brez predpone prizora. Prvi korak uporabi že pripravljene
ID-je, vrste in okvirje, ne novega modelskega seznama objektov. Pri `Products`
lahko ena oznaka zajema skupino predmetov. Za večje število regij ima konfiguracija
THUD kontekst 16384 in do 8192 izhodnih tokenov; ostale nastavitve so enake.

## 2b. Celotna baza ali druga RGB mapa na zunanjem disku

Bazo razširi zunaj repozitorija. Za prvi test izberi eno mapo, ki vsebuje RGB
fotografije. Ohranitev izvirne strukture in imen datotek olajša poznejše povezovanje oznak.

V spodnjem ukazu **zamenjaj pot z dejansko potjo do izbrane RGB mape**:

```bash
export DATASET_DIR="/media/student/IME_DISKA/THUD/POT_DO_IZBRANE_RGB_MAPE"
test -d "$DATASET_DIR" && echo "Mapa obstaja"
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml:compose.dataset.yaml
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
mkdir -p data results
docker compose build app
```

Če `Mapa obstaja` ni izpisana, popravi pot, preden nadaljuješ. V kontejnerju bo ta
mapa dostopna kot `/datasets`, samo za branje. Poti `/media/student/...` iz Linux
namizja znotraj kontejnerja niso samodejno na voljo.

Pripravi izbor največ 30 slik, vsako dvajseto po urejenem seznamu imen:

```bash
docker compose run --rm -T --no-deps --entrypoint python app \
  -m llm_object_classifier.image_dataset \
  --images-dir /datasets --output data/selected/frames.jsonl \
  --limit 30 --every-n 20
```

Če je slik premalo, uporabi manjši `--every-n` in novo ime izhodnega JSONL. Za tipična
THUD imena lahko dodaš `--patterns '*.color.png'`. Ne izbiraj mape z RGB, globinami
in maskami hkrati; ta osnovni izbor ne ugiba, katera PNG je prava fotografija.
Datoteka vsebuje točne poti, izvorna imena in SHA256 slik; slik ne kopira in ne prenaša
celotne baze. Obstoječega izbora ne prepisuje. Spremenjeno sliko pozneje zazna.

Pomembno: pri tem koraku **ne uvažamo oznak THUD++**. Brez polja `objects` bo prvi
model sam predlagal vidne objekte. To je dovolj za začetno primerjavo opisov, ne pa
za meritev pravilnosti zaznavanja. Za uvoz dejanskih oznak potrebujemo pravi vzorec
slike in njen JSON. Obstoječi pretvornik realne sheme THUD ne ugiba.

## 3. Samo prvi korak: primerjaj slikovne modele

```bash
bash scripts/stage1_describe.sh nova-baza-hitri \
  --input data/selected/frames.jsonl --limit 3
xdg-open results/stage1/nova-baza-hitri/comparison.html
```

Nato isti izbor na vseh izbranih slikah:

```bash
nohup bash scripts/stage1_describe.sh nova-baza-30 \
  --input data/selected/frames.jsonl --limit 30 \
  > stage1-launch.log 2>&1 < /dev/null &
tail -F results/stage1/nova-baza-30.log
```

Vsak nov poskus potrebuje novo ime. `Ctrl+C` ustavi samo spremljanje dnevnika.
Skripta zgradi aplikacijo, prenese izbrane lokalne modele, preveri uporabo GPU in
zažene samo opisovanje. Lastni projektni model po pregledu oziroma ob koncu njegove
skupine sprosti iz VRAM. Tujih Ollama endpointov ne upravlja.

Privzeta konfiguracija `configs/stage1/qwen-compare.json` primerja BF16 in Q4 modela
Qwen3.5 4B. Nastavi isti prompt, isto podrobnost, temperaturo 0, seme 42, kontekst
8192 in do 4096 izhodnih tokenov. Tu kvantizacijo primerjamo **v prvem koraku**;
dosedanji benchmark jo je primerjal predvsem pri razvrščanju.

Za primerjavo drugih modelov kopiraj konfiguracijo, uredi `models` in jo podaj:

```bash
cp -n configs/stage1/qwen-compare.json configs/stage1/my-models.json
nano configs/stage1/my-models.json
bash scripts/stage1_describe.sh moji-opisi \
  --config configs/stage1/my-models.json --input data/selected/frames.jsonl --limit 30
```

- `name`: preprosto ime podmape, ki mora biti enolično.
- `model`: točen tag v Ollami oziroma ime, ki ga sprejme zunanji strežnik.
- `backend`: `ollama` ali `openai`.
- `precision`: neobvezna kontrola dejansko poročane natančnosti pri Ollami.
- `require_gpu`: za Ollamo privzeto `true`; preveri pozitiven `size_vram` za zahtevani model.
- `detail`: `brief` (35/25 besed), `standard` (80/60), `detailed` (160/120) za sceno/objekt.
- `temperatures`, `seeds`: kombinacije za opisovanje, privzeto `[0]`, `[42]`.
- `max_tokens`: največ izhodnih tokenov. `num_ctx`: kontekst pri Ollami.
- `system_file`, `instruction_file`: neobvezni UTF-8 datoteki. Relativni poti sta
  relativni glede na konfiguracijo, ne na trenutno mapo.

Prvi korak vedno izvede sveže zahteve, da predpomnilnik ne prikrije časa delovanja.
Primerjava je urejena po modelih, da zmanjša preklapljanje uteži. Čas prve zahteve
lahko vključuje nalaganje modela. Povzetek zato izpiše tudi povprečje brez prvega
uspešnega klica; to ni zagotovljena meritev ogretega modela. Metapodatki in surovi
odgovori se shranijo za podrobnejšo analizo. Modelom so poslane enake izvirne slikovne
bajte; njihovo notranje predprocesiranje se lahko razlikuje.

Za točno eno sliko podaj njen `frame_id` iz vhodnega JSONL:

```bash
bash scripts/stage1_describe.sh ena-slika \
  --input data/nyu20/objects.jsonl --frame-id nyuv2-test-000003 --limit 1
```

### Kaj dobiš

- `comparison.html`: fotografije in dejanski opisi vseh modelov, brez internetnih odvisnosti.
- `descriptions.csv`: opisi po objektih in parametrična različica.
- `summary.json`: uspehi, napake, število vrnjenih objektov in časi zahtev po modelih.
- `manifest.json`: točni prompti, izbor slik, nastavitve, metapodatki modelov/verzija Ollame.
- `qwen35-bf16/t0-s42/frame-000001.json`: celoten odgovor in sled zahteve za eno sliko.

Brez referenc noben izhod ne poroča o accuracy. Več vrnjenih objektov ali daljši opis
ne pomenita boljše kakovosti. Če seznam objektov ni priložen, ima vsak model svoje
lokalne ID-je; ID 5 pri dveh modelih ni nujno isti predmet. V HTML jih preglej po
vidnem predmetu in opisu. Za isto množico objektov med modeli kasneje dodamo dejanske
instance iz baze.

## 4. NVILA: pripravljena povezava, ločen zagon strežnika

Za novi neposredni GPU izvajalnik NVILA-8B in namenske naloge Florence glej
[ločeno nadaljevanje THUD](THUD_EXTENDED.md). Spodnji odsek opisuje zgodovinsko
API predlogo; novi native izvajalnik ne potrebuje tega strežnika.

`configs/stage1/nvila-server.example.json` je API predloga za primerjavo Qwen/NVILA.
**Ni namestitev ali že preverjen zagon NVILA na RTX 5090.** Najprej je treba postaviti
uradni VILA/NVILA strežnik v svojem okolju ter preveriti CUDA/PyTorch, natančnost in VRAM.

Ko strežnik deluje:

1. Nastavi `model` na točno ime, ki ga strežnik sprejme, in `base_url` na njegov osnovni URL.
2. Naš adapter doda `/chat/completions`. URL že vsebuje `/v1` samo, če ga zahteva tvoj strežnik.
3. Za uradni `server.py`, ki sprejema le `user`/`assistant`, uporabi `system_as_user=true`:
   sistemsko navodilo in podatki so združeni v eno uporabniško sporočilo. To je zabeleženo.
4. Pregledani uradni `server.py` ima ekspliciten seznam dovoljenih imen in vključuje
   `NVILA-15B`, ne pa `NVILA-8B`. Za 8B mora izvajalnik dejansko podpirati oziroma
   pravilno vključiti ta model; preimenovanje 8B v 15B ni ustrezna rešitev.
5. Pregledani API podpira `temperature`, `top_p`, `max_tokens`; semen, `top_k`,
   kazni ponavljanja in konteksta ne nastavlja prek istih polj. Zato jih adapter ne
   pošilja in jih označi v `unsupported_options`. Za strožjo primerjavo je treba
   uskladiti tudi te strežniške nastavitve. Privzeta temperatura 0 zmanjša vpliv vzorčenja.
6. `json_mode` in `send_seed` omogoči le, če jih strežnik dejansko podpira. GPU in
   natančnost pri tem API-ju nista samodejno preverljiva; preveri ju na strežniku.

Primer dodatne omrežne nastavitve, če je API na gostiteljskem Linux računalniku:

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml:compose.dataset.yaml:compose.external-api.yaml
bash scripts/stage1_describe.sh qwen-nvila \
  --config configs/stage1/nvila-server.example.json \
  --input data/selected/frames.jsonl --limit 3
```

`host.docker.internal` kaže na gostitelja. Strežnik mora poslušati na naslovu, ki je
dosegljiv iz kontejnerja. Če je v drugem kontejnerju, uporabi njegovo omrežno ime.
Ključ za API se lahko prebere iz `api_key_env`, nikoli se ne shranjuje v rezultate;
če ga potrebuješ v kontejnerju, ga moraš tudi izrecno posredovati z Dockerjem.

Uradna koda in opombe o API so preverjene proti:
https://github.com/NVlabs/VILA in https://github.com/NVlabs/VILA/blob/main/server.py

## 5. Drugi korak: razvrsti shranjene opise

Za zdaj tega ni treba zaganjati, če primerjaš samo opisovanje. Ko izbereš nabor opisov:

```bash
bash scripts/stage2_reason.sh reasoning-bf16-opisi \
  --captions results/stage1/nova-baza-30/qwen35-bf16/t0-s42
```

Oba razvrščevalnika dobita isti opis BF16. Slike se v tem koraku ne berejo in ne pošiljajo.
Če želiš preveriti vpliv opisovalnika, ponovi na opisih Q4 z istimi razvrščevalniki:

```bash
bash scripts/stage2_reason.sh reasoning-q4-opisi \
  --captions results/stage1/nova-baza-30/qwen35-q4/t0-s42
```

Uporabi podmapo **enega modela in ene različice**, ne cele mape prvega koraka.
Nastavitve drugega dela so v `configs/stage2/qwen-reason.json`. Prompt spreminjaj v
`configs/prompts/reasoning-system.txt`. Vsebina ob zagonu se zapiše v manifest.
Odgovori in razlogi so v `classifications.csv` ter posameznih JSON datotekah.
Napaka v opisu se izrecno zabeleži kot preskočen vhod; brez veljavnega opisa ni klasifikacije.

## 6. GPU in izhodne kode

```bash
COMPOSE_FILE=compose.yaml:compose.gpu.yaml docker compose exec ollama ollama ps
```

`100% GPU` potrjuje celoten model na grafični. Pozitiven `size_vram`, ki ga preverja
program, potrjuje uporabo GPU, ne pa nujno odsotnosti delnega prenosa na CPU.
Če se ponovi prejšnja izguba GPU, ponovno ustvari samo projektno Ollamo:

```bash
COMPOSE_FILE=compose.yaml:compose.gpu.yaml docker compose up -d --force-recreate --wait ollama
```

Modeli in rezultati ostanejo shranjeni. Drugih kontejnerjev ali GPU procesov ne ustavljaj.

Izhod 0: vsi zahtevani odgovori veljavni. Izhod 2: vsaj en neveljaven odgovor ali preskočen
vhod; ostali se vseeno izvedejo. Izhod 1: usodna napaka priprave/GPU. Napake niso tiho
popravljene; tudi podvojeni JSON ključi se v novih korakih zavrnejo, izvorni odgovor ostane.

## THUD vmesna predstavitev: celoten 1. del

Nov ponovljiv zaganjalnik `bash scripts/thud_experiment.sh --phase all` izvede
preverjanje/izbor → pilot → osnovno ločeno samostojno zaznavanje in opisovanje
regij (3 modeli × 100 slik × 2 načina) → dodatne teste na zamrznjenih 20 slikah
→ besedilno presojo shranjenih opisov → vrednotenje → PDF/LaTeX.
Glej [paket poročila](../reports/THUD_VMESNO/README.md) in
[vrstni red nastavitev](../configs/THUD_VRSTNI_RED.md). Zaganjalnik ohrani tudi
neveljavne odgovore, preveri nespremenjene vhode in omogoča nadaljevanje.
Drugi korak cevovoda je vključen v prvi del naloge; **2. del nadgradnje prvega
koraka se s tem ne začne**.

## THUD: izrecno naročeno nadaljevanje (2. del)

`bash scripts/thud_part2.sh --phase all` uporabi iste tri nameščene modele in
slike, izvede samo nove bogate opise ter manjkajoče besedilne primerjave.
Stari uspešni klici se ne ponavljajo. `description_schema: rich-v2` v konfiguraciji
vključi strogo preverjanje novih polj in grafa lokalnih ID-jev; stare konfiguracije
in opisi ostanejo združljivi. Besedilni korak sam prepozna verzionirano shemo in
prejme vse shranjene dokaze/odnose, brez slik. Oba koraka podpirata `--resume`.
Glej [navodila, rezultate in omejitve](../reports/THUD_VMESNO/README.md#2-del-bogatejši-samostojni-opis).
