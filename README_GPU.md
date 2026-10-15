# GPU test: slika → opis → LLM → fixed / movable / uncertain

Koda je na veji **feature/docker-smoke**. Na `main` je za zdaj samo prvotni README.

## Zagon na laboratorijskem Linux računalniku

Predpogoji: NVIDIA gonilnik (`nvidia-smi` deluje), Docker Compose v2 in
NVIDIA Container Toolkit za Docker. Za namestitev toolkita glej
[README_DOCKER.md, razdelek 3](README_DOCKER.md). Če je že nastavljen, ga ne nameščaj ponovno.

V obstoječem projektu:

```bash
cd ~/jakob/Klasifikacija-to-k
git switch feature/docker-smoke
git pull --ff-only origin feature/docker-smoke
bash scripts/test_vision_benchmark.sh vision-gpu-prvi both
```

Za čisto novo kopijo:

```bash
git clone --branch feature/docker-smoke https://github.com/Vulpes271/Klasifikacija-to-k.git
cd Klasifikacija-to-k
bash scripts/test_vision_benchmark.sh vision-gpu-prvi both
```

Skripta posodobi Ollamo, zgradi aplikacijo, rezervira NVIDIA GPU za kontejner,
prenese modela in po potrebi 20 slik NYUv2. Preveri teste, dejansko poročano
natančnost modela BF16 in uporabo GPU po poskusni zahtevi ter po vsakem novem
opisu/klasifikaciji. `num_gpu=999` zahteva prenos vseh slojev na GPU.
Če `/api/ps` za zahtevani model ne poroča o pozitivnem `size_vram`, se test ustavi;
ne nadaljuje tiho samo na CPU. Pozitivni `size_vram` potrjuje uporabo GPU,
ne zagotavlja, da ni delnega prenosa slojev na CPU: preveri tudi `ollama ps`.
Stanje modela, kontekst in poraba VRAM se shranijo v JSON vsake zahteve.

Za hiter preizkus z dvema slikama, nato za celotni test v ozadju:

```bash
LIMIT=2 bash scripts/test_vision_benchmark.sh vision-gpu-hitri both
nohup bash scripts/test_vision_benchmark.sh vision-gpu-celotni both > vision-gpu-launch.log 2>&1 < /dev/null &
tail -F results/vision-gpu-celotni.log
```

`Ctrl+C` prekine samo spremljanje dnevnika. Računalnik mora ostati buden.
Začetne napake preveri tudi v `vision-gpu-launch.log`.
Za vsako ponovitev uporabi novo ime; starih rezultatov skripta ne prepisuje.

Samo nekvantizirani model ali samo kvantizirani razvrščevalnik:

```bash
bash scripts/test_vision_benchmark.sh vision-bf16-prvi bf16
bash scripts/test_vision_benchmark.sh vision-q4-prvi quantized
```

## Kaj se primerja

- Opisovalnik: `qwen3.5:4b-bf16`, temperatura 0, seme 42.
- Razvrščevalnika: `qwen3.5:4b-bf16` in `qwen3.5:4b-q4_K_M`.
- BF16 je nekvantizirana 16-bitna različica uteži; ni FP32.
- Opis se pripravi enkrat na sliko in shrani v predpomnilnik. Oba LLM-ja
  prejmeta **isti opis, iste oznake in isti poziv, brez slike**.
- Tako primerjamo razvrščanje na podlagi istega opisa. Vpliv kvantizacije
  na sam opis slike ni del te primerjave.
- Temperatura: **0, 0.3, 0.7**; semeni: **42, 43**.

Štirje profili iz `configs/vision-sweep.json` spreminjajo po en parameter:

| Profil | top_p | top_k | presence_penalty |
| --- | ---: | ---: | ---: |
| reference | 0.8 | 20 | 1.5 |
| top-p-095 | 0.95 | 20 | 1.5 |
| top-k-40 | 0.8 | 40 | 1.5 |
| presence-0 | 0.8 | 20 | 0 |

Vsak profil se preizkusi pri vseh temperaturah in semenih:
**24 kombinacij × 20 slik × 2 modela = 960 zahtev za razvrščanje**, poleg največ
20 novih opisov slik. Pri temperaturi 0 so lahko semena in vzorčevalni parametri
brez učinka; ponovitve pomagajo preveriti stabilnost, niso neodvisni vzorci.
Kontekst je 8192 tokenov, največ 2048 izhodnih tokenov, dodatno razmišljanje je
izklopljeno. Profili veljajo za razvrščevalnik; opisovalnik ima stalne nastavitve.
Vsak klic ima svoj pogovor, zato ne nabira zgodovine prejšnjih poskusov.

Uradna modela sta ob pripravi navedena kot približno 9.3 GB (BF16) in 3.3 GB
(Q4_K_M), poleg tega je potreben VRAM za izvajanje in kontekst. Na RTX 5090 32 GB
je 4B razumna začetna velikost; dejansko razpoložljivost VRAM preveri z
`nvidia-smi`. Modela se izvajata zaporedno. Slika Ollame uporablja `latest` zaradi
podpore aktualnim GPU; manifest shrani različico Ollame in metapodatke modela.
Za strogo ponovljivost shrani tudi digest Docker slike in pripni različice.

## Rezultati

```bash
cat results/vision-gpu-prvi/bf16/summary.json
cat results/vision-gpu-prvi/q4/summary.json
head -n 10 results/vision-gpu-prvi/bf16/results.csv
```

Vsak model dobi svojo mapo:

- `manifest.json`: modeli in natančnost uteži, pozivi, vsi profili ter nastavitve.
- `summary.json`: uspehi, napake, izpuščene slike in število objektov po kombinacijah.
- `results.csv`: ena vrstica na objekt in kombinacijo, skupaj z opisom in razlogom.
- `frame-001/caption.json`: surov ter razčlenjen opis slike.
- `frame-001/reasoning-reference-t0.3-s42.json`: odgovor, poziv, parametri, čas in GPU.

Predpomnilnik `results/vision-caption-cache` se deli med primerjavami, vendar se
uporabi le, kadar se ujemajo slika, oznake, poziv, model in nastavitve opisovanja.
Sprememba konteksta ali števila izhodnih tokenov zato sproži nov opis.
Čas predpomnjenega opisa je čas prvotnega generiranja, ne branja predpomnilnika.

Izhodna koda 0 pomeni, da so vsi odgovori prestali preverjanje formata;
2 pomeni, da je vsaj en opis/odgovor odpovedal ali bil izpuščen. Druga natančnost
se ob kodi 2 vseeno izvede. Napaka GPU ali priprave ustavi celoten postopek.
To niso meritve pravilnosti klasifikacije: za accuracy potrebuješ ročne reference.
NYUv2 vzorec ima semantične razrede, ne posameznih instanc objektov.
`fixed` pomeni trajno fiksiran objekt; `movable` se lahko premakne med obiski;
`uncertain` pomeni premalo dokazov. Mirovanje na eni fotografiji ni dokaz fiksnosti.

## Lastne nastavitve

Profilom lahko dodaš tudi `num_ctx`, `num_predict` ali `num_gpu`. Primer ločenega
zagona (po pripravi kontejnerjev):

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
docker compose run --rm --no-deps --entrypoint python app \
  -u -m llm_object_classifier.vision_pipeline \
  --input data/nyu20/objects.jsonl --output-dir results/vision-lastni \
  --vision-model qwen3.5:4b-bf16 --reason-model qwen3.5:4b-bf16 \
  --temperatures 0 0.2 0.5 0.8 --seeds 42 43 44 \
  --profiles configs/vision-sweep.json --num-ctx 8192 --num-predict 2048 \
  --require-gpu --require-unquantized --limit 20
```

Ustavitev projektne Ollame po končanem testu (rezultati in modeli ostanejo):

```bash
COMPOSE_FILE=compose.yaml:compose.gpu.yaml docker compose down
```

## Preverjanje kode in viri

```bash
python -m unittest discover -s tests -v
```

Testi preverjajo ločitev slike in besedilnega LLM, pošiljanje parametrov,
različne izhodne datoteke, skupen predpomnilnik, spremembo konteksta,
zavrnitev CPU in napačno označene kvantizirane uteži ter izolacijo napak.
V pripravljalnem okolju ni Dockerja ali NVIDIA GPU; dejanska inferenca in
Docker build se preverita ob zagonu zgornje skripte na laboratorijskem računalniku.

Uradni viri:
- https://ollama.com/library/qwen3.5:4b-bf16
- https://ollama.com/library/qwen3.5:4b-q4_K_M
- https://docs.ollama.com/docker
- https://docs.ollama.com/api/ps
- https://docs.docker.com/compose/how-tos/gpu-support/
