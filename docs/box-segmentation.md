# Izbirno filtriranje okvirjev in SAM 2 segmentacija

`--stage segment` bere obstoječe LILocBench `manifest.json`, `frozen/`,
`images/` in `region-classes.csv`. Ne kliče Florence/LLM in ne spreminja vhodov.
Korak ni del privzetega `--stage all`. Izhodna mapa mora biti nova in ločena od
vhodnih rezultatov. Slike se izberejo izrecno z `--frames`.

Združevanje zahteva IoU >= 0,95, razliko vsakega roba <= 2 % manjše pripadajoče
stranice ter skladno identiteto v oznaki in celotnem opisu. Pogoja za koordinate
morata veljati med vsemi člani skupine. Odstranijo se samo začetni členi
`a/an/the`; brez splošnih sinonimov ali pretvorbe množine. Dodatno je podprt
izrecni sklic `a cardboard box` → `The box`, kadar opis uvede natanko eno
škatlo v ednini in vsebuje `The box`. Ob negotovosti ostaneta obe regiji.
Reprezentativni okvir je prvi izvorni okvir, ne unija. Vsi izvorni okvirji,
ID-ji, oznake, opisi in razredi vseh shranjenih modelov ostanejo v `sources`.
Nasprotujoči razredi se ne popravljajo; manjkajoči razred ostane `null`.

Prekrivanje različnih predmetov ne združuje regij. Logotip dobi metapodatek
`part_of` samo pri izrecnem opisu `box with the logo ... on it` in natanko enem
ustreznem okvirju škatle, ki ga vsebuje. Ohranita se obe regiji in obe maski.
Drugi izrazi povezave del–celota se za zdaj konzervativno ne povezujejo.

Vsaka ohranjena regija, tudi `fixed` ali brez razreda, dobi ločen SAM 2
box prompt. Uporablja se samo SAM 2.1 Hiera Small z eno izhodno masko
(`multimask_output=False`), brez samodejnega generiranja regij ali sprememb
razredov. Maska ni pravokotnik, odrezan iz slike, temveč napoved segmentacijskega
modela; ne odšteva se drugih mask in ne pripravlja mask za izločanje.

## Enkratna priprava

Pregled lokalnega `models/` in knjižnic v `thud-florence:local` ni našel
nameščenega SAM modela. Izolirano okolje temelji na obstoječem PyTorch 2.7.1,
Florence in LLM okolji ostaneta nespremenjeni. Dockerfile fiksira revizijo
uradne implementacije SAM 2. Namestitev ne prevaja izbirne CUDA razširitve.

Iz korena projekta:

```bash
docker build -f docker/Dockerfile.sam2 -t klasifikacija-sam2:local .
mkdir -p models/sam2.1-hiera-small
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/models/sam2.1-hiera-small:/weights" klasifikacija-sam2:local \
  python -c "import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt', '/weights/sam2.1_hiera_small.pt')"
```

V tem delovnem okolju sta slika vsebnika in checkpoint že pripravljena;
priprave ni treba ponavljati. Uteži so lokalne in ignorirane z Gitom.
Model in API: [uradni SAM 2](https://github.com/facebookresearch/sam2),
[model SAM 2.1 Hiera Small](https://huggingface.co/facebook/sam2.1-hiera-small).

## Zagon na dveh obstoječih slikah

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:$PWD" -w "$PWD" klasifikacija-sam2:local \
  python -m llm_object_classifier.pipeline --stage segment \
  --config configs/pipeline.json --results-from 'results models/lilocbench' \
  --frames view-000044 view-000086 --output results/lilocbench-box-segmentation-test2
```

Ta izhod že obstaja po osnovnem preizkusu. Za ponovitev izberi novo prazno
izhodno mapo. V okolju z nameščenim SAM 2 deluje tudi isti `python -m ...`
ukaz brez Docker ovoja. Pri odsotnosti CUDA se uporabi CPU.

- `masks/<frame_id>/region-<id>.png`: binarna maska, 0 ozadje / 255 predmet,
  izvirna širina in višina slike, brez spremembe koordinat.
- `segmentation.json`: povezave mask z okvirji, izvornimi ID-ji in razredi;
  kontrolne vsote vhodov, uteži, izvajalnika in mask ter rezultat modela SAM.
  `sam_score` je interna ocena modela, ne meritev pravilnosti.
- `previews/<frame_id>/region-<id>.jpg`: prekrivni prikaz posamezne maske.
  `fixed` je zelen; `uncertain`, `movable` in manjkajoči ali nasprotujoči
  razredi so rdeči. Barve temeljijo na shranjenem Qwen 27B razredu
  (`segmentation.preview_model`); razredi drugih modelov ostajajo ohranjeni.

Osnovni preizkus uporablja samo `view-000044` in `view-000086` (640 × 480).
Iz 25 izvornih regij je nastalo 23 mask: združena sta samo para ID-jev
`[1, 4]` na prvi in `[2, 4]` na drugi sliki. Drugi okvir škatle na drugi
sliki (ID 5), ki se koordinatno ne ujema, je ohranjen. Oba logotipa imata
ločeno masko in opisno potrjeno povezavo do škatle.
Vizualni pregled glavnih škatel potrjuje, da maski izpustita vidno steno.
Vseh 32 regresijskih testov je uspešnih.
To ni validacija lokalizacije ali meritev pravilnosti za celoten nabor.
Posegov v ocene, LiDAR ali geometrijo parov ni.
