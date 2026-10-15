# Majhen prvi test v Dockerju (Linux terminal)

> Starejša, ohranjena pot. Za nova ločena koraka glej [docs/WORKFLOW.md](docs/WORKFLOW.md).


Ta postopek pripravi Python program, lokalno Ollamo in majhen model.
Prenesemo **10 resničnih RGB slik NYUv2 in njihove semantične maske**,
ne celotnega arhiva. Prenos slik je reda nekaj MB; skripta izpiše dejansko velikost.
Privzeti model **Qwen2.5 0.5B** ima približno **398 MB**.
Docker slike in njihovi razširjeni sloji potrebujejo dodaten prostor: rezerviraj
približno 5–8 GB. Ne prenašaj velikih arhivov THUD za ta preizkus.

To je test delovanja programa. Model je majhen in njegove odločitve niso
referenčne oznake za znanstveno vrednotenje.

## 1. Predpogoji

Potrebuješ Git, Docker Engine in Docker Compose v2:

```bash
git --version
docker --version
docker compose version
docker info
```

Če Docker že deluje, preskoči namestitev. Ukazi spodaj so za **Ubuntu 22.04/24.04/26.04**,
ne za vse distribucije Linuxa. Če imaš že drugačno namestitev Dockerja,
je ne zamenjuj s temi ukazi. Za Debian uporabi uradna navodila spodaj.

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Po dodajanju v skupino `docker` se odjavi in ponovno prijavi v Linux,
da naslednji ukazi delujejo brez `sudo`. Skupina docker omogoča skrbniški dostop.

Viri: [Ubuntu](https://docs.docker.com/engine/install/ubuntu/),
[Debian](https://docs.docker.com/engine/install/debian/).

## 2. Prenesi projekt

Koda je v veji `feature/docker-smoke`. Repozitorij je zaseben, zato potrebuješ
GitHub dostop. Pri HTTPS uporabi GitHub token/credential manager, ne gesla računa.
Tokena ne vstavljaj v URL. Če imaš nastavljene SSH ključe, lahko uporabiš SSH URL.

```bash
git clone --branch feature/docker-smoke https://github.com/Vulpes271/Klasifikacija-to-k.git
cd Klasifikacija-to-k
mkdir -p data results
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
```

Če projekt že imaš lokalno in nimaš neshranjenih sprememb:

```bash
cd /pot/do/Klasifikacija-to-k
git fetch origin
git switch --track origin/feature/docker-smoke
mkdir -p data results
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
```

## 3. Izberi CPU ali NVIDIA GPU

**Za prvi majhen test priporočam CPU.** Ne potrebuje CUDA, NVIDIA Container
Toolkita ali GPU. To tudi prepreči zasedanje laboratorijske RTX 5090.

```bash
export COMPOSE_FILE=compose.yaml
```

**Če želiš NVIDIA GPU**, mora `nvidia-smi` že delovati na gostitelju.
Docker ne namesti gonilnika grafične kartice. Za RTX 5090 uporabi gonilnik,
ki podpira Blackwell. Nato enkrat namesti NVIDIA Container Toolkit
(Ubuntu/Debian; poseg zahteva skrbniški dostop):

```bash
nvidia-smi
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
```

Na skupnem laboratorijskem računalniku uskladi ponovni zagon Dockerja s skrbnikom,
ker ustavi tudi druge Docker kontejnerje. Če je toolkit že nastavljen, uporabi samo
zadnji `export COMPOSE_FILE=...`.

## 4. Pripravi Docker in model

Isti ukazi veljajo za CPU in GPU, glede na izbrani `COMPOSE_FILE`:

```bash
docker compose config
docker compose build app
docker compose up -d --wait --wait-timeout 120 ollama
docker compose exec ollama ollama pull qwen2.5:0.5b
docker compose exec ollama ollama list
```

Uteži se prenesejo enkrat v Docker volumen. Nobena koda ne kliče plačljivega
OpenAI API-ja. Ollama API ni izpostavljen na omrežju gostitelja.

## 5. Prenesi samo 10 označenih slik

```bash
docker compose run --rm --no-deps --entrypoint python app scripts/download_nyu_sample.py --count 10 --output-dir data/nyu-small
```

Skripta prek Hugging Face Dataset Viewerja prenese samo zahtevane RGB slike in
maske. Ne prenese celotnega Parquet niza ali 2,8 GB MAT datoteke. Vhod je
`data/nyu-small/objects.jsonl`; poleg so JPEG slike, PNG maske in `source.json`.

**Ta majhna različica ima 40 semantičnih razredov, nima ločenih instanc.**
Dva stola na sliki sta zato en zapis `chair`, ne dva izmišljena objekta.
To je primerno za začetni test razvrščanja seznamov razredov. Nima poz kamere
in oznak fixed/movable. Izvirne NYUv2 instance ali THUD dodamo v poznejši fazi.
Ne označuje vseh predmetov: piksel 0 je neoznačen in ga izpustimo.

Vir: [NYUv2 mirror](https://huggingface.co/datasets/wyrx/NYUDepthv2_seg),
[izvirna baza in pogoji/citiranje](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
Zemljevid imen uporablja ID 0..39, maske pa 1..40; pretvornik upošteva razliko.
Semantični PNG ni fotografija: ohranimo številčne indekse palete.

Če želiš samo sezname in maske, brez RGB, pri prenosu dodaj `--labels-only`.
Za nov izbor izberi drugo mapo in npr. `--offset 100 --count 10`.
Obstoječega `objects.jsonl` skripta ne prepiše.

Če internet ali Dataset Viewer ne deluje, lahko še vedno testiraš z
`examples/demo.jsonl`. To je izmišljen demonstracijski nabor, ne resnična baza.

## 6. Zaženi prvi test modela

```bash
docker compose run --rm app run --input data/nyu-small/objects.jsonl --config configs/docker-qwen-small.json --output-dir results --run-name qwen-small --limit 10
```

Rezultati so na gostitelju in ostanejo po ustavitvi Dockerja:

```bash
cat results/qwen-small/summary.json
head -n 10 results/qwen-small/results.csv
```

Preveri število `ok` in `error`. Zaključen ukaz sam po sebi ne pomeni,
da je model vrnil veljavne odgovore. Napake in surovi odgovori so v `results.jsonl`.
Majhen model lahko napačno razvršča ali izpusti ID; program to zabeleži.
Ob ponovitvi isti `--run-name` dobi končnico, npr. `qwen-small-2`.

Test z demonstracijskimi podatki (brez prenosa baze):

```bash
docker compose run --rm app run --input examples/demo.jsonl --config configs/docker-qwen-small.json --output-dir results --run-name demo-small
```

## 7. Referenčna metoda brez modela in testi programa

```bash
docker compose run --rm --no-deps app baseline --input data/nyu-small/objects.jsonl --output-dir results --run-name nyu-baseline
docker compose run --rm --no-deps --entrypoint python app -m unittest discover -s tests -v
```

Za resnične slike ne poročamo o accuracy brez ročnih referenčnih oznak.
Demonstracijski `examples/reference.csv` velja samo za `examples/demo.jsonl`.
Trenutna evaluacija ocenjuje uspešno obdelane objekte; pri primerjavi modelov
vedno poročaj tudi o napakah in pokritosti. To še ni končni znanstveni benchmark.

## 8. Ustavitev in ponovni zagon

Po testu Python kontejner sam konča (`--rm`). Ollama ostane v ozadju.
Za ustavitev vsega v tem projektu:

```bash
docker compose down
```

Model in rezultati ostanejo shranjeni. V novem terminalu ponovno nastavi
`LOCAL_UID`, `LOCAL_GID` in ustrezni `COMPOSE_FILE`, nato:

```bash
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export COMPOSE_FILE=compose.yaml
docker compose up -d --wait --wait-timeout 120 ollama
```

Za GPU namesto zadnjega CPU `export` uporabi `compose.yaml:compose.gpu.yaml`.
Če se prenos slik že izvaja, `Ctrl+C` prekine ukaz; za prekinitev vseh
projektnih kontejnerjev lahko v drugem terminalu uporabiš `docker compose down`.

Če želiš **izbrisati tudi prenesene modele** in sprostiti prostor:

```bash
docker compose down --volumes
```

To izbriše projektni volumen Ollame. Host mapi `data/` in `results/` ostaneta.
Ne uporabljaj globalnega `docker system prune`, če imaš druge Docker projekte.

## 9. Malo večji model

Ko prvi test deluje, prenesi Qwen2.5 1.5B in spremeni samo ime v konfiguraciji:

```bash
docker compose up -d --wait --wait-timeout 120 ollama
docker compose exec ollama ollama pull qwen2.5:1.5b
sed 's/qwen2.5:0.5b/qwen2.5:1.5b/' configs/docker-qwen-small.json > configs/docker-qwen-1.5b.json
docker compose run --rm app run --input data/nyu-small/objects.jsonl --config configs/docker-qwen-1.5b.json --output-dir results --run-name qwen-1.5b
docker compose down
```

## 10. Isti test s Qwen3.5 0.8B

V isti projektni mapi uporabi drugo konfiguracijo; podatkov ni treba ponovno
prenašati. Model `qwen3.5:0.8b` je majhen model za prvi test (trenutni prenos
približno 1–1.3 GB). Uporabimo iste objekte, isti poziv, temperaturo 0 in
omejitev 2048 izhodnih tokenov. Za hiter test je dodatno razmišljanje izklopljeno
z `reasoning_effort: none` prek Ollaminega OpenAI združljivega vmesnika.

Če projekt že imaš, najprej pridobi spremembe:

```bash
git pull --ff-only origin feature/docker-smoke
bash scripts/test_qwen35.sh
```

Skripta posodobi Ollama sliko, zažene storitev, zgradi Python aplikacijo,
prenese model in zažene test na `data/nyu-small/objects.jsonl`. Rezultati se
shranijo v novo mapo `results/qwen3.5-small-DATUM-URA/`, katere pot izpiše na
koncu. Rezultati prejšnjega modela ostanejo v `results/qwen-small/`.
Skripta tudi zavrne že obstoječo mapo, da ne pomeša ponovitev.

Za predvidljivo ime mape ali GPU uporabi:

```bash
# CPU, poimenuj mapo results/qwen3.5-prvi:
bash scripts/test_qwen35.sh qwen3.5-prvi

# GPU, če je NVIDIA Container Toolkit že nameščen (razdelek 3):
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
bash scripts/test_qwen35.sh qwen3.5-gpu

# Ustavitev storitve; podatki in modeli ostanejo:
docker compose down
```

Primerjaj klasifikacije v `results.csv`, število napak in čase v povzetkih.
Ta vzorec ima semantične oznake objektov, nima pa referenčnih oznak
`fixed/movable/uncertain`. Zato iz njega še ne moremo samodejno izračunati,
kateri model je pravilnejši; najprej ročno določi reference za iste objekte.
Manjše število napak JSON pomeni zanesljivejši format, ne nujno boljše
klasifikacije. Modela sta tudi različnih velikosti (0.5B in 0.8B).

Viri: [model Qwen3.5 v Ollami](https://ollama.com/library/qwen3.5:0.8b),
[nadzor razmišljanja](https://docs.ollama.com/api/openai-compatibility).
Dejanski zagon Qwen3.5 moraš preveriti na svojem računalniku; v okolju priprave
ni Dockerja ali GPU.

## Preverjanje te dostave

Preverjeni so Python testi, dejanski prenos majhnega vzorca in priprava JSONL.
V okolju izdelave ni Docker daemon-a ali GPU, zato Docker build in dejanski
zagon Ollame nista bila izvedena. `docker compose config` in zgornji prvi test
sta preverjanji na tvojem Linux računalniku. `latest` pri Ollami namenoma
uporablja aktualno podporo za nove GPU; za končni benchmark zabeleži/pini
uporabljeno sliko in digest modela (`docker compose exec ollama ollama list`).
