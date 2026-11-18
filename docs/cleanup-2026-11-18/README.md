# Čiščenje aktivnega projekta, 18. november 2026

Prvi ločeni commit te naloge. Pred posegom je bilo stanje objavljeno in preverjeno na [arhivski veji](https://github.com/Vulpes271/Klasifikacija-to-k/tree/archive/before-cleanup-relocalization-2026-11-18), commit `0510c225320da56bc2b89e2e135711beb57c9618`.

Lokalni arhiv: `/home/student/jakob/archives/Klasifikacija-to-k-0510c225-20261118`. `manifest.json` vsebuje vsako izvorno pot, tip, velikost in SHA256. Vsi bajti so bili preverjeni pred odstranitvijo; podvojene vsebine imajo v arhivu eno kopijo z ohranjenimi potmi. Zasebne ocene so v arhivu in dodatni konsistentni SQLite kopiji; aktivni `reviews/` je ohranjen in nespremenjen. Manifest in ocene niso objavljeni.

Iz projektne kopije je odstranjenih **1,092,761,371 B (1.018 GiB)** in **1509 sledenih datotek**. Arhiv hrani 1,049,378,271 B unikatnih vsebin ter manifeste in SQLite varnostno kopijo. Zmanjšanje projektne kopije ni enako sprostitvi diska: unikatne vsebine ostajajo v lokalnem arhivu, Git zgodovina pa je nespremenjena.

[Celotni seznam odstranjenih poti, velikosti in števila datotek](removed-paths.csv) · [Arhivska revizija in kontrolne vsote ohranjenih vhodov](archive-validation.json) · [Preverjanje zagona](startup-validation.json)

## Preverjene odvisnosti

- `results models/lilocbench`: `configs/lilocbench-pilot.json`, segmentacija in pregledovalnik ga še uporabljajo. Izvorne slike, zamrznjeni opisi, odgovori, kalibracije in maske so ohranjeni; odstranjena je samo stara statična HTML galerija in morebitne podvojene native vizualizacije.
- Stari 40/200-slikovni nabori so bili odvisnost starega privzetega pregledovalnika. Privzeta pot je spremenjena na LILocBench. Adapterji in testi starih formatov ostanejo; po preselitvi v arhiv so preverjene nespremenjene identitete naborov in vseh regij. Ocene se zato lahko nadaljujejo z izrecnimi arhivskimi potmi.
- `data/nyu20`, začetni tar arhivi, opuščeni THUD/LOCO/vision izhodi in `models/retired-metadata` niso vhodi aktivnega pipelinea. Ohranjen je lokalni arhiv vseh njihovih vsebin, tudi unikatnih in manifestov.
- Dejanske preostale tenzorske uteži Florence in SAM sta aktivni; njuna SHA256 sta preverjeno nespremenjena. Opuščene NVILA/Qwen-VL uteži so bile odstranjene že prej; preostala metadata so zdaj arhivirana. Qwen in Gemma v Ollami ter Docker volumni so ohranjeni.
- Testi opuščenih izvajalnikov na začetku že niso bili prisotni. Vseh 82 obstoječih aktivnih testov ostaja in je uspešnih, vključno z lokalnim HTTP izvozom pregledovalnika. Preverjena sta oba CLI zagona brez inferenc; pregledovalnik privzeto naloži 100 slik in 1442 regij.
- Ohranjen je celoten `predstavitev/`, vsa končna poročila v `docs/`, Docker in vsi LILocBench skeni, maske, kalibracije, cache projekcij in štiri mreže. Slednje ter raw skeni imajo preverjene nespremenjene kontrolne vsote.

Stare mape, ustvarjene z drugim lastnikom, so bile po preverjenem arhiviranju obdelane z obstoječo Docker sliko brez omrežja, samo za popravek lastništva pregledanih poti. Modelski proces ni bil zagnan.

## Obnova

Za stare izvajalnike uporabite ločen checkout arhivske veje. Lokalne podatke odprite iz podmape `tree/` arhiva ali jih obnovite v ločeno kopijo po zasebnem `manifest.json`. `RESTORE.md` v arhivu vsebuje navodila. Arhivskih ocen ne prepisujte čez aktivne brez primerjave. `README_REVIEW_VIEWER.md` poda ukaz za pregled starih naborov.

Skripta `maintenance/archive_project_cleanup.py` preveri SHA256 vseh kopij in referenčne identitete pred odstranitvijo. `--resume` nadaljuje samo iz obstoječega preverjenega manifesta. Postopek ni običajen zagonski ukaz pipelinea.
