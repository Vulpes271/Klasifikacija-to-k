# Opisovanje in razvrščanje objektov za robotsko zaznavanje

**Naloga za Codex na Linuxu:** [cilji, načrt poskusov in poročilo za predstavitev](CODEX_NALOGA.md).
Dokument loči takojšnji eksperimentalni del od nadaljnje nadgradnje prvega koraka.

Aktivna pot je razdeljena na tri jasne korake. Slike in rezultati ostanejo lokalno;
Git vsebuje kodo, nastavitve in dokumentacijo.

| Korak | Namen | Ukaz / modul | Rezultati |
| --- | --- | --- | --- |
| 0. Izbor slik | Iz nove RGB mape naredi ponovljiv seznam posnetkov | `python -m llm_object_classifier.image_dataset` | `data/selected/frames.jsonl` |
| 1. Opisi in primerjava | Vsak slikovni model opiše iste slike | `bash scripts/stage1_describe.sh IME ...` | `results/stage1/IME/` |
| 2. Razvrščanje | Besedilni LLM oceni že shranjene opise, brez slike | `bash scripts/stage2_reason.sh IME ...` | `results/stage2/IME/` |

**Začni z [navodili za Linux in novo bazo](docs/WORKFLOW.md).**
Za zdaj je smiselno urediti izbor podatkov in primerjati prvi korak. Drugega lahko
zaženeš pozneje na točno izbranih opisih, brez ponovnega opisovanja slik.

## Vključena izbora THUD: 2 x 50 slik

Oba izbora sta že v repozitoriju: `data/thud_dense_test50` (trgovina) in
`data/thud_agtea_test50` (raznoliki predmeti). [Podrobnosti in zagon](data/README.md).

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
bash scripts/stage1_describe.sh thud-prvi \
  --config configs/stage1/thud-compare.json \
  --input data/thud_dense_test50/objects.jsonl --limit 2
```

Za teh 100 izbranih fotografij zunanji disk ni potreben. Celotna izvorna baza
ostaja lokalno in se ne nalaga v Git.

## Hiter preizkus na že prenesenih slikah NYUv2

Če imaš že `data/nyu20/objects.jsonl`, lahko novo ločitev preveriš takoj:

```bash
export COMPOSE_FILE=compose.yaml:compose.gpu.yaml
bash scripts/stage1_describe.sh opisi-preizkus --input data/nyu20/objects.jsonl --limit 2
xdg-open results/stage1/opisi-preizkus/comparison.html
```

Privzeto primerja Qwen3.5 4B BF16 in Q4. Primerjava Qwen/NVILA je pripravljena
kot ločena API predloga; strežnik NVILA mora biti najprej samostojno pripravljen
in preverjen. Ni samodejnega prenosa NVILA skozi Ollamo.

## Kaj urejati

- **Prvi korak:** `configs/stage1/qwen-compare.json` (modeli, podrobnost, temperatura, tokeni).
- **NVILA API predloga:** `configs/stage1/nvila-server.example.json` (glej omejitve v navodilih).
- **Drugi korak:** `configs/stage2/qwen-reason.json`.
- **Sistemski prompt razvrščanja:** `configs/prompts/reasoning-system.txt`.
- **Uporabniško navodilo razvrščanja:** `configs/prompts/reasoning.txt`.
- **Opisovalni prompt:** `llm_object_classifier/description_prompts.py`; za eksperiment lahko
  v stage1 konfiguraciji nastaviš `system_file` na svojo UTF-8 datoteko.

## Izhodi in vrednotenje

Prvi korak ustvari `comparison.html` s fotografijami in dejanskimi opisi vseh modelov,
`descriptions.csv`, `summary.json`, manifest s prompti in metapodatki modelov ter
JSON vsakega odgovora, vključno z napakami. HTML je samostojen in deluje brez strežnika.

Drugi korak ustvari `classifications.csv`, manifest, povzetek in izvorne odgovore z
utemeljitvami. Predmeti iz neoznačenih fotografij so predlog prvega modela, ne oznake baze.
Brez ročnih referenc ne poročamo o accuracy. Veljaven JSON ne pomeni pravilnega opisa.

## Dosedanji poskusi in združljivost

[Pregled starejših poskusov](docs/legacy/README.md) loči začetne teste, ki so že dali
vmesne rezultate. Ostanejo ponovljivi z obstoječimi ukazi in v svojih starih mapah;
novi prvi/drugi korak se z njimi ne mešata. Rezultatov in lokalnih podatkov koda ne briše.

## Preverjanje kode

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
```

HTTP testi uporabljajo lokalni simulirani strežnik za oba API-ja; preverjajo dejanske
zahteve, ločitev slike in besedila, enake vhode med modeli, ohranjanje napak, zavrnitev
CPU in podvojenih ključev JSON. Docker build in inferenco na NVIDIA GPU je treba
preveriti na ciljnem Linux računalniku.
