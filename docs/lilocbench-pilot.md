# LILocBench: majhen A/B preizkus sledenja z AMCL

Iz korena repozitorija, na tem računalniku:

```bash
python3 -m llm_object_classifier.pipeline --stage localize \
  --sequence lt_changes_0 --start 1733414028 --end 1733414058 \
  --data data/lilocbench-pilot --output results/lilocbench-pilot
```

`start/end` sta absolutna časovna žiga UNIX v sekundah, vključeni sta obe meji.
Sekvenca je parameter; samodejni prenos je namenoma omejen na `lt_changes_0`.
Odsek je omejen na največ 40 sekund in mora vsebovati vsaj pet že klasificiranih
RGB slik. `--preflight-only` izvede samo preverjanje 5–10 parov, brez AMCL.
Za svež par zagonov izberi novo `--output` mapo; isti izhod ponovno uporabi
shranjeni oceni A/B samo ob skladni konfiguraciji in podpisu vhodov replaya.
Maske se ponovno uporabijo tudi pri novi izhodni mapi. Ni glasovanja med slikami.

Potrebni so Docker, obstoječa slika `nbv-sim-jazzy-r4p:latest` z ROS Jazzy /
AMCL 1.3.12 in po potrebi `klasifikacija-sam2:local` z že nameščenim
SAM 2.1 Hiera Small; slednjo opisuje [box-segmentation.md](box-segmentation.md).
Za gostitelja: `python3 -m pip install -r requirements-localization.txt`.
Na tem računalniku je vse pripravljeno. AMCL in SAM tečeta brez omrežja.
Potrebni so tudi shranjeni Florence/Qwen rezultati v `results models/lilocbench/`.
Če manjka model/vsebnik ali vhod, izvajalnik ne izbere drugega modela.
Nastavitve so v `configs/lilocbench-pilot.json`; nastavitve segmentacije v
`configs/pipeline.json` so enake obstoječemu koraku. Florence/LLM se ne kličeta.

## Odsek in podatki

**Pred A/B** je bil izbran 30-sekundni interval `1733414028–1733414058`:
hodnik → ena pisarna s prestavljenim pohištvom in škatlami → hodnik.
Izbira je temeljila na že shranjenih opisih/slikah in poteku GT poti, ne na
napakah lokalizacije. Zapis je v `data/lilocbench-pilot/selection.json` in
`results/lilocbench-pilot/selection.json`. Slednji se zapiše pred lokalizacijo.

Uporabljenih je 450 zaporednih sprednjih skenov (15 Hz) ter osem obstoječih
RGB slik `view-000034/036/038/040/042/044/046/048`. Maska `044` je bila že
pripravljena; preostalih sedem je segmentiranih z obstoječim SAM modelom.
Klasifikacije so shranjene Qwen3.5 27B Q4 (`qwen35-27b-q4`), brez nove inference.

Lokalno so bili že GT, intrinzična/ekstrinzična kalibracija in RGB slike.
Samo za manjkajoče sprednje skene, odometrijo in specifikacijo LiDAR je uporabljen
HTTP Range dostop do članov uradnega ZIP. Dodan je uradni occupancy grid z YAML.
Prenesenih je bilo **19.134.841 B** podatkov (poleg majhnih izvornih datotek za
pregled evaluatorja/AMCL). Med pripravo je bilo prostih 123 GB; ekstrakcija je
zasedla približno 99 MB skupaj s cacheom. Ni prenosa `mapping`, depth, celotnega
arhiva ali dataseta. Prenos ima omejitev 12 GB in preveri prostor za ekstrakcijo.
Uspešni prenosi bajtnih obsegov so shranjeni za nadaljevanje; prekinjena ekstrakcija
se ponovno sestavi iz teh obsegov. Dnevnik je `data/lilocbench-pilot/http-cache/downloads.json`.

## Projekcija in filter

Pred AMCL so preverjeni vsi izbrani pari (osem): `preflight.json` in
`projection_check.png`. Najbližji odmik RGB od sredine skena je 2,6–29,5 ms.
Kalibracijsko drevo je uradno `base_link → scan_front_link` oziroma
`base_link → camera_front_color_optical_frame`; projekcija uporablja uradni
`plumb_bob` model z radialnim in tangencialnim popačenjem. Časovni zamik in
posamezni žarki so kompenzirani s planarno interpolirano kolesno odometrijo.
GT ne sodeluje pri projekciji ali izbiri odstranjenih zadetkov. Zaradi višinske
razlike kamere in LiDAR se ne uporablja zgolj horizontalni kot.

Za vsak sken je veljavna samo najbližja posamična RGB maska z odmikom največ
50 ms od sredine skena. Zunaj tega okna, za kamero, izven slike ali pri neveljavni
projekciji ostane meritev. Odstranijo se samo piksli mask, katerih vsi izvorni
zapisi imajo `movable` za izbrani Qwen model. Pri prekrivanju z masko
`fixed/uncertain`, manjkajočim ali konfliktnim razredom se zadetek ohrani.
Originalne regije, razredi in ID-ji ostanejo v segmentacijskem cacheu.

V B je odstranjen zadetek `NaN`, nikoli meritev do največjega dosega. AMCL
`likelihood_field` preskoči NaN. A ohrani izvirni vektor meritev; obe veji imata
iste indekse žarkov. Odstranjenih je 789/362.949 veljavnih zadetkov (**0,2174 %**),
oziroma 8,896 % zadetkov na skenih z veljavno masko. Veljavno časovno masko ima
11/450 skenov (**2,444 %**). Vsi ostali skeni ostanejo v časovnem zaporedju.
Depth ni potreben: to je projekcijski filter končnih LiDAR zadetkov, brez
preverjanja okluzij s kamerino globino. Zato je diagnostični prikaz pomemben.

## Lokalizacija in ocenjevanje

Obe veji uporabljata isti uradni 5-cm zemljevid, odometrijo, časovne korake,
AMCL `OmniMotionModel` + `likelihood_field`, 100 žarkov, 1000–10000 delcev in
seme **42**. Ni samodejnega tuninga. AMCL standardno nastavlja `srand48` tudi iz
stenske ure; majhna knjižnica `amcl_seed.c` prek `LD_PRELOAD` fiksira samo
sejanje na 42 in ohrani generator `drand48` ter preostali AMCL nespremenjen.
Nastavitve in koda so isti za A/B. Celoten vhod replaya je podpisan.

Prva interpolirana GT poza inicializira obe veji z isto kovarianco
`diag(0,2 m, 0,2 m, 10°)^2`. Vhod replaya vsebuje samo to začetno GT pozo,
zemljevid, odometrijo in skene. Kasnejši GT se uporablja le za ocenjevanje in
prikaz. Ni naknadne poravnave trajektorije.

Za 2D napaki se neposredno izvedejo nespremenjene numerične funkcije
`poses_to_2d`, `ape`, `ape2d` iz uradnega evaluatorja. Datoteka in SHA sta lokalno
v provenance; njegove CLI/HTML odvisnosti niso potrebne. GT je interpoliran
na **iste čase vseh skenov A/B**, z linearnim položajem in Slerp orientacijo;
vrzeli nad 150 ms in ekstrapolacija niso dovoljene. To je pilot na časih skenov,
ne celoten uradni challenge z več začetnimi časi in zagoni.

Pragi so strogo `<`, kot zahtevano. V deležih je imenovalec vsak čas skena z GT;
manjkajoča ocena šteje kot neuspeh. RMSE/max upoštevata razpoložljive pare
GT–ocena. Skupni uspeh zahteva hkrati položaj <10 cm in |yaw| <4°.

| Meritev | A: originalni skeni | B: filtrirani skeni |
|---|---:|---:|
| Položaj RMSE / max [cm] | 4,015 / 7,800 | 3,887 / 7,736 |
| Položaj <5 / <10 cm [%] | 79,56 / 100,00 | 82,67 / 100,00 |
| Yaw RMSE / max [°] | 2,083 / 4,465 | 2,117 / 4,805 |
| Yaw <2 / <4° [%] | 60,00 / 95,56 | 54,89 / 97,56 |
| Skupni uspeh | 430/450 = **95,56 %** | 439/450 = **97,56 %** |
| Manjkajoče ocene / časi brez GT | 0 / 0 | 0 / 0 |

Razlika je **+2,00 odstotni točki**. En par na kratkem odseku z redkimi
maskami ne potrjuje splošnega izboljšanja; yaw RMSE in največja yaw napaka
sta pri B nekoliko slabša. V okviru tega pilota ni dodatnih zagonov/tuninga.

## Izhodi

V `results/lilocbench-pilot/` so `metrics.json`, `poses.csv` (450 vrstic podatkov),
`localization_map.png`, `lidar_overlay.png`, `projection_check.png`, `preflight.json`,
`provenance.json`, skupni `replay/` ter ločena `A/` in `B/` z ocenami in dnevniki.
`localization_map.png` ima isti izrez in merilo za A/B; GT je zelen ob skupnem
uspehu in rdeč ob neuspehu, ocenjena pot pa modra. V `lidar_overlay.png` so zadetki
postavljeni z **GT pozami samo za diagnostiko**, kar piše v naslovu in legendi;
prikazana sta zemljevid in primer RGB maske.

Večji vhodi/cache/uteži/izhodne slike so ignorirani (`data/`, `models/`, `results/`).
V Git gredo le koda, testi, konfiguracija in navodila. `reviews/` ostaja nedotaknjen.

Uradni viri:
[dataset, kalibracija in zemljevid](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/),
[članek, sliki 5/6 in tabela III](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/trekel2025iros.pdf),
[evaluator](https://github.com/PRBonn/LILocBench),
[AMCL likelihood field](https://github.com/ros-navigation/navigation2/blob/jazzy/nav2_amcl/src/sensors/laser/likelihood_field_model.cpp).
