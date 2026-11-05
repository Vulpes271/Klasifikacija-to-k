# Čiščenje aktivne veje — 2026-11-05

## Arhiv potrjen pred odstranjevanjem

Celoten začetni posnetek je objavljen na veji
[`archive/experiments-before-prune-2026-11-05`](https://github.com/Vulpes271/Klasifikacija-to-k/tree/archive/experiments-before-prune-2026-11-05),
commit `04eed71ac18d3684c094e4fe9f8449283171addd`. Pred prvo odstranitvijo je bil
SHA potrjen z `git ls-remote`. Veje `main` in `feature/separate-vision-stages` niso spremenjene.

## Obseg

- Sledenih datotek pred posegom: **12.873**.
- Odstranjenih starih sledenih datotek: **9.631**.
- Ohranjenih obstoječih sledenih datotek: **3.242**.
- Dodan je samo ta novi zapisnik; po čiščenju skupaj **3.243** sledenih datotek.

Odstranjeni so stari slikovni/besedilni poskusi iz
`analiza rezultatov vmesna predstavitev/`, nepovezani deli `results models/`,
stari izbori THUD/LOCO v `data/`, poročila, eksperimentalne konfiguracije in prompti,
izvajalniki, navodila, primeri in testi. Odstranjevale so se samo konkretne sledene
datoteke; lokalne nesledene ali ignorirane vsebine niso bile množično brisane.

## Preverjene neposredne odvisnosti

Ohranjen je celoten `results models/lilocbench/` (1.700 sledenih datotek).
Spremenjen je samo njegov README, da za reprodukcijo pravilno napoti na arhivsko
vejo; podatki, slike, odgovori, manifesti, prompti, poze in kalibracije so nespremenjeni.

Pregledovalnik še potrebuje naslednje stare poti; zato ostajajo samo njegovi
konkretni vhodi, ne nekdanji izvajalniki ali statični primerjalni izvozi:

| Imenik | Sledenih datotek po posegu | Namen |
| --- | ---: | --- |
| `results models/prompt_comparison40` | 507 | Izvorna manifesta, 40 zamrznjenih opisov, odgovori vseh različic in objavljeni izvozi ročnih ocen |
| `results models/florence_grounding_llm_comparison10` | 801 | Manifest 200-slikovnega pregleda, zamrznjeni opisi in 600 odgovorov |
| `results models/florence_caption_grounding_test10` | 200 | Originalne slike, ki jih pregledovalnik izbere pred označenimi nadomestili |

Vse te poti, bajti izvornih manifestov, ID-ji regij, identitete naborov in
anonimne identitete odgovorov ostajajo enaki. Zato dosedanje ocene ostajajo
povezane z istimi regijami; resnične baze niso bile odprte prek pregledovalnika.

Aktivna koda obsega Florence native runtime, izris/prenos regij, besedilni
backend, skupne validatorje in podpise, Ollama tokenizer ter pregledovalnik.
Izjema `GPURequiredError` je prestavljena v skupni GPU modul, da backend ne
uvaža celotnega opuščenega VLM izvajalnika. Stari privzeti CLI paketa je
zamenjan z aktivnim pipelineom. Prompt za razrede, njegova natančna vsebina,
Florence nalogi in nastavitve `configs/pipeline.json` se niso spreminjali.
Compose ohranja isto Ollama storitev in volumen; opuščeni profil CPU benchmark
aplikacije je odstranjen iz definicije. Obstoječi Docker vsebniki, slike in
volumni niso bili spreminjani ali odstranjeni.

## Preverjanje

- 25 ohranjenih regresijskih testov uspešnih: 24 v peskovniku, lokalni HTTP test
  zaradi omejitve vtičnic uspešno ponovljen zunaj njega. Vsi uporabljajo začasne
  podatke/ocene in ponarejene modelne odzive. Nobenih modelnih inferenc.
- Vsi relativni uvozi aktivnega paketa imajo ohranjene ciljne module.
- `python3 -m llm_object_classifier --help` deluje in kaže aktivni CLI.
- Pregledovalnik brez odpiranja baze še naloži 40 slik / 414 regij,
  200 slik / 1.985 regij ter LILocBench 100 slik / 1.442 regij.
- Identitete naborov, zaporedje ključev regij, množica model/prompt/seme in vse
  poti odgovorov/slik se ujemajo s posnetkom pred posegom.
- SHA256 vseh 3.206 ohranjenih podatkovnih datotek in štirih lokalnih datotek
  ocen se ujema s posnetkom pred posegom. README dokumentacija je iz te
  podatkovne primerjave izključena, ker je namenoma posodobljena.

Uteži `models/florence-2-large`, oba 27B modela, zasebni `reviews/`, lokalni
surovi rezultati ter `vision-launch.log` niso bili odstranjeni. Arhiviranja
HTML iz commita `04eed71` nismo ponavljali in zunanjega arhiva nismo spreminjali.
Zgodovina Git objektov se s to spremembo ne zmanjšuje; arhiv ostaja dostopen.
