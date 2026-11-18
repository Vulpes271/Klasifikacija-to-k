# Inventar repozitorija in lokalnih vhodov

Posnetek sledenih poti iz `git ls-files` pred objavo tega poskusa; izhodišče `ba62e13edc2d02090e2da5e0e5e46054a143973c`. Velikosti so logični bajti datotek v delovnem drevesu, ne velikost Git zgodovine ali zasedba diskovnih blokov. Vključene so nove, že dodane datoteke poskusa. Seznam vsebuje 3919 datotek in 317 map z agregati (nadrejene mape vključujejo podmape). Ni bilo brisanja.

[Vsaka sledena datoteka z velikostjo in namenom](inventory-files.csv) · [Vsaka mapa z vsoto velikosti](inventory-directories.csv) · [Lokalni vhodi in cache](local-inputs.csv)

| Mapa na vrhu | Datoteke | Velikost [MiB] |
|---|---:|---:|
| results models | 3208 | 186.48 |
| docs | 599 | 45.31 |
| predstavitev | 24 | 8.40 |
| llm_object_classifier | 28 | 0.33 |
| maintenance | 18 | 0.21 |
| tests | 17 | 0.07 |
| configs | 12 | 0.01 |
| docker | 2 | 0.00 |
| data | 1 | 0.00 |
| .github | 1 | 0.00 |

## Za zagon in ponovitev

Obvezna koda je označena `potrebno za zagon` v inventarju; seznam nastane iz lokalnih Python uvozov začetne skripte. Potrebni sta obe konfiguraciji, `requirements-localization.txt` opisuje del okolja; dejansko uporabljeni paketi so NumPy, SciPy, PyYAML in Matplotlib. Nastavitve in ukazi so v [README](README.md).

Surovi `static_0/raw.npz`, opis laserja `raw.json`, kalibracija `transformations.yaml` ter mreži `three` in `empty` morajo biti lokalno pripravljeni. Vse natančne poti, velikosti in razpoložljivost so v `local-inputs.csv`. Ti vhodi niso sledeni v Gitu. Za branje rezultatov so dovolj README, objavljeni CSV in PNG. Za ponovni izris brez lokalizacije potrebujemo lokalne CSV po kombinacijah in revizije v `data/lilocbench-tracking/full` ter mreži; `--stage render` ne izvaja sledenja.

## Kandidati za ločeno odločitev o čiščenju

Starejši `results models/` in starejša poročila LILocBench so označeni kot kandidati za pregled arhiviranja. Niso samodejno odvečni: vsebujejo izvirne rezultate in so lahko povezani iz drugih poročil. Pred premikom preveriti povezave in ohraniti arhiv.

Lokalni maskovni/LLM cache, starejši izhodi AMCL, začasne projekcije in uteži niso potrebni za ta sekvenčni test, če ohranimo oba zemljevida, testne skene in kalibracijo. `data/lilocbench-reverse` vsebuje tudi geometrijo in maske za druge poskuse, zato brisanje ni priporočilo brez pregleda. Nov `data/lilocbench-tracking` podvaja objavljene meritve in hrani tehnični preizkus; po arhiviranju revizij je kandidat za čiščenje, vendar je trenutno potreben za ukaz `--stage render`.

Mapi `results/lilocbench-reverse` in `data/lilocbench-filtered-map` vsebujeta potrebne vhode skupaj z drugimi izhodi. Nikoli ju ne odstraniti v celoti samo na podlagi skupne velikosti. CSV loči konkretne potrebne datoteke. O drugih vsebinah projekta ta inventar ne podaja ocene za brisanje.

Osvežitev po dodajanju nameravanih datotek v Git: `python3 maintenance/inventory_tracking.py`, nato ponovno dodaj posodobljene inventarje. Skripta samo bere in piše svoje poročilo, ničesar ne briše.
