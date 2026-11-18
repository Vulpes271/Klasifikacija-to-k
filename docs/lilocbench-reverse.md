# LILocBench: obratni preizkus in tri kamere

**Rezultati so zaključeni (280/280).** [Prečiščeno poročilo s slikami](lilocbench-reverse/README.md)
in [PDF](lilocbench-reverse/porocilo.pdf) najprej prikazujeta običajni AMCL brez
dodatne odometrijske premostitve. Običajni AMCL še vedno uporablja kolesno
odometrijo. Spodaj so tehnične podrobnosti ponovitve.

```bash
MPLCONFIGDIR=/tmp/mpl-lilocbench python3 -u maintenance/run_reverse_experiment.py
```

Konfiguracija: `configs/lilocbench-reverse.json`. Rezultati, slike, zemljevidi in
trajektorije: `results/lilocbench-reverse/`. Podatki in modeli se ponovno uporabijo;
novi RGB, odzivi in maske se hranijo v `data/lilocbench-reverse/`. Izvajanje potrebuje
omrežje za omejene prenose uradnega arhiva in dostop do obstoječega Docker/GPU.
Ollama izvajalnik `klasifikacija-ollama-1` mora delovati. Modeli, prompti in
parametri AMCL ostanejo iz obstoječega preizkusa. Zaključeni paketki in zagoni
se pri nadaljevanju ne ponavljajo. `progress.json` beleži trenutno stopnjo,
`error.json` morebitno prekinitev, `metrics.json.complete` zaključek vseh zagonov.

Nadzornik `maintenance/run_reverse_experiment.py` prepreči dvojni zagon,
arhivira nedokončane maske in največ trikrat ponovi isti SAM korak po nativnem
sesutju 139/134. Ne ponavlja prejetih odgovorov LLM, klasifikacijskih napak ali
kontekstnih preskokov. `job.json` pove, ali izračun dejansko teče, je končan ali
ustavljen; `runtime-recoveries/` hrani sled okrevanj. Živo stanje lahko osvežuje
`python3 maintenance/watch_reverse_status.py`; prikaže ga `STATUS.md`.

Med izračunom mask lahko `maintenance/run_reverse_controls.py` na CPU izvede
120 kanoničnih kontrol, ki ne potrebujejo novih filtrov. Končni poskus iste
izhode ponovno uporabi. Po zaključku vseh 280 zagonov
`maintenance/finalize_reverse_report.py` izdela jedrnate tabele, parne razlike,
ločene geometrijske in maskirane pokritosti ter dodatne razdalje iz GT.
Možnost `--wait` počaka na `job.json.status=complete`; pri napaki se ustavi.

`guard_reverse_continuation.py` obravnava tudi potrjeno izgubo GPU dostopa
znotraj Ollama vsebnika (`Failed to initialize NVML: Unknown Error`). Ponovno
zažene samo ta vsebnik in nato zahteva uspeh izvirnega preverjanja modelnega
digesta, konteksta in celotnega GPU nalaganja; CPU izvajanje ostane zavrnjeno.
To je [dokumentirana težava NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html).
Če še ni prejetih odgovorov, lahko arhivira diagnostični `model-metadata.json`,
kadar se razlikuje zgolj vrstni red istih privzetih `PARAMETER` vrstic, ki jih
Ollama izpiše po ponovnem zagonu. Digest, predloga, različica strežnika in vse
vrednosti se morajo ujemati; spremenjene nastavitve se ne normalizirajo.
Prejeti odgovori in izvorni modelni podpisi se ohranijo.

Stopnje: `prepare`, `preflight`, `models`, `project`, `maps`, `geometry`, `amcl`,
`report`. Privzeti `all` jih izvede v tem zaporedju. Celotni preizkus ima 280 AMCL
zagonov: sedem kombinacij zemljevida in vhodnih skenov × naravne/umetne vrzeli ×
običajni AMCL/premostitev × deset semen. Osem ROS procesov lahko deluje hkrati,
vsak v lastnem Docker omrežnem prostoru. Modelski paketki so zaporedni.

Uradni [opis podatkov](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/)
potrjuje skupni koordinatni sistem javnega GT. `lt_changes_0` ima prestavljeno
pohištvo in dodane škatle, `static_0` predstavlja statično osnovno stanje;
`mapping` je ločena referenčna vožnja. To ni kontroliran par povsem prazne in
polne sobe. Osnovno stanje vsebuje pohištvo, zato sklepi veljajo za spremembe
glede na opazovano referenčno geometrijo. Vse tri vožnje obiščejo izbrano sobo;
števila skenov so v `sequence-audit.json`.

Vsak žarek ohrani svoj indeks. Projekcija uporablja odometrijsko kompenzacijo
časa posameznega žarka in slike ter ločeno kalibracijo vsake kamere. Izbran je
najbližji shematsko veljaven maskiran RGB znotraj ±50 ms sredine skena. Brez
veljavne maske vidna kamera zaščiti svoje žarke. Vsaj ena oznaka `movable` brez
veta `fixed`, `uncertain`, konflikta ali manjkajoče maske odstrani žarek enkrat.
Neoznačeno ozadje pri sicer veljavni maski ne glasuje. Zunaj unije vidnih polj
žarek ostane. Kot v obstoječem preizkusu ni globinskega testa zakrivanja.

Referenca točk je celotna vožnja `static_0`. Vsi veljavni konci žarkov imajo
GT interpoliran na čas žarka z največjo vrzeljo 150 ms. Enaka razdalja 10 cm
velja za vse tri variante. Delež ujemanj in odvečnih točk uporablja število
ohranjenih zadetkov, brez naključnega vzorčenja za metrike. Prikaz je lahko
razredčen. Neujemanje šteje kot odvečno samo v prosti celici referenčne mreže.
Nikoli opažene celice so ločene od opaženih, a neodločenih/zasedenih celic.
Odstranjeni žarki ne posodabljajo niti konca niti prostega prostora.
Manjkajoče referenčne točke so sekundarna metrika in vključujejo razliko v
opazovanosti voženj.

Premostitev sproži manj kot 100 veljavnih zadetkov v obdelanem skenu.
Zanesljivost sidra določa izključno objavljena kovarianca AMCL: standardni
odklon največ 20 cm za x/y in 10° za yaw. Med vrzeljo se skeni ne objavljajo,
izhodišče pa je zadnja tako sprejeta AMCL poza. Njeno SE(2) odometrijsko razliko
prenesemo v koordinatni sistem zemljevida. Ko zadetkov zadostuje, AMCL dobi
nov sken in akumulirano odometrijo. GT ni del odločitve o premostitvi.
Umetne vrzeli so vnaprej določene pri 40 s (0,5 s), 90 s (1 s), 180 s (2 s)
glede na začetek vsake vožnje in so enake za obe metodi. Običajni AMCL med njimi
prejema enake skene z neveljavnimi zadetki. Naravne vrzeli so poročane ločeno;
če jih pri tem pragu ni, se jih ne razglaša za uspešen preizkus naravnih vrzeli.

`gaps.json` vsebuje napake, delež uspeha, manjkajoče ocene, varen začetni
odsek, najdaljše neprekinjeno trajanje/razdaljo pod pragoma in čas do začetka
vsaj ene sekunde neprekinjenega skupnega uspeha po vrzeli. Neopažen GT ne šteje
za potrjeno varen del vrzeli; navedeno je tudi njegovo število. Manjkajoča
lokalizacijska ocena ob razpoložljivem GT šteje kot neuspeh.

Metrike položaja <10 cm, orientacije <4° in obeh skupaj se izpišejo ločeno,
s povprečjem in populacijskim odklonom semen 42–51. Slike kažejo seme 42;
rdeča označuje neuspeh na referenčni poti (tudi pri manjkajoči oceni), oranžna
odometrijsko premostitev, modra oceno in zelena GT. Rezultati se ne uporabljajo
za izbiranje pragov, vrzeli ali semen.

Testi: `python3 -m unittest tests.test_localization_reverse tests.test_filtered_map
tests.test_localization_pilot -q`. Lokalni kratek ROS preizkus je v
`results/lilocbench-reverse/smoke/`; ne predstavlja metrike celotne sekvence.

Podatki in izpeljani zemljevidi: Trekel et al., LILocBench, IROS 2025,
CC BY-NC-SA 4.0.
