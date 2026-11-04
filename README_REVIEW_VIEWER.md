# Lokalno ročno ocenjevanje

Pregledovalnik bere obstoječe rezultate; modelov ne zaganja in izvornih datotek ne spreminja. Deluje brez GPU in interneta. Potrebuje Python 3.10 ali novejši s standardnim modulom `sqlite3`. **Dodatnih pip odvisnosti ni.** Če Python še ni nameščen, je ukaz na Ubuntu/Debian:

```bash
sudo apt install python3
```

Iz korena projekta zaženite:

```bash
python3 -m llm_object_classifier.review_viewer
```

Odpre brskalnik na `http://127.0.0.1:8765/` in privzeto naloži `results models/prompt_comparison40`. Ustavite s Ctrl+C; enak ukaz nadaljuje pri zadnji izbrani sliki/regiji in ohrani ocene. Če samodejno odpiranje brskalnika ni na voljo, odprite izpisani naslov ročno.

Če obstaja `results models/prompt_comparison40/bf16/manifest.json`, samodejno vključi še **Qwen3.5 4B BF16 in 9B BF16**, vsakega s starim in novim promptom ter semenom 42. Izvorni manifest, SHA zamrznjenih vhodov, ID-ji regij in identiteta nabora ostanejo isti. Zato vaša že podana referenca velja tudi za nove modele; referenc ni treba ponoviti. Obstoječe anonimne oznake ostanejo iste, novi odgovori dobijo dodatne oznake.

Ob prvem odpiranju baze po dodajanju modelov pregledovalnik pred spremembo ustvari konsistentno SQLite varnostno kopijo v `reviews/backups/` (oziroma `backups/` ob lastni poti baze). Nato doda samo nove identitete in izračunana ujemanja. Prejšnjih referenc, opomb, ocen utemeljitev, zgodovine in položaja ne združuje ali prepisuje. Med razvojem se dejanska baza ne odpira; migracija se izvede, ko sami zaženete pregledovalnik.

Skupna referenca se vnese samo enkrat. Privzeti način **Samo referenčni razredi** skriva odgovore in po uspešnem shranjevanju odpre naslednjo neocenjeno regijo. Tipke **1/2/3/4** pomenijo FIXED/MOVABLE/UNCERTAIN/NE MOREM OCENITI. Neveljaven ali manjkajoč celoten odgovor se v izračunih posebej označi in ne dobi avtomatskega ujemanja. Delni razred in izvorna utemeljitev ostaneta vidna za pregled, s tehničnim opozorilom.

Utemeljitve so poljubne in imajo svoj način ter napredek. Dobesedno enake utemeljitve z enakim napovedanim razredom iste regije se povežejo tudi med BF16/Q4 modeli, prompti in ponovitvami. Ena izrecna ocena velja za vse navedene povezane odgovore. Različne stare ocene ostanejo nespremenjene, dokler sami ne izberete nove skupne ocene. Neocenjene utemeljitve ne preprečijo dokončanja referenčnega pregleda.

Skupni statični pregled vseh petih modelov in njihovih ponovitev: `results models/prompt_comparison40/bf16/index.html`; skupni izvoz regij: `bf16/all-region-classes.csv`. Oba sta pregled rezultatov, brez vaših zasebnih ocen. CSV/JSON osebnih ocen izvozite iz Python pregledovalnika.

Za obstoječi izvoz 200 slik:

```bash
python3 -m llm_object_classifier.review_viewer --results-dir "results models/florence_grounding_llm_comparison10"
```

Ta izvoz vsebuje samo stari prompt, zato pregledovalnik zanj prikaže tri shranjene odgovore. Novi prompt in dodatne ponovitve so na voljo tam, kjer so dejansko shranjeni v izboru 40 slik.

Poti in vrata so nastavljivi, na primer:

```bash
python3 -m llm_object_classifier.review_viewer --results-dir "results models/prompt_comparison40" --florence-dir "results models/florence_caption_grounding_test10" --db reviews/moje-ocene.sqlite3 --port 8766 --no-browser
```

`--florence-dir` določa mapo izvirnikov; če posamezni izvirnik manjka, vmesnik uporabi shranjeno označeno sliko in to izpiše. Izbrani okvir poudari brez spreminjanja njegovih koordinat. Nabor in zamrznjene vhode preveri s SHA256. Odgovore poveže z regijo po njenem izvornem ID in preveri SHA vhodnega dokumenta; manjkajočih podatkov ne dopolnjuje.

## Ocenjevanje in zasebni podatki

Privzeti način **Samo referenčni razredi**: za regijo enkrat izberite FIXED / MOVABLE / UNCERTAIN / NE MOREM OCENITI. Po uspešnem zapisu se samodejno odpre naslednja regija brez referenčne odločitve (po potrebi od začetka nabora). Ko takih regij ni več, ostanete na trenutni. Odgovori ostanejo skriti. Ob napaki se prehod ustavi in pokaže se možnost ponovnega shranjevanja; po ponovnem poskusu lahko nadaljujete z naslednjo regijo ali ponovno izbiro reference.

V načinu **Pregled utemeljitev** po referenčni odločitvi po želji odprite odgovore. Ročne ocene utemeljitev PRAVILNA / NAPAČNA / NE MOREM OCENITI so neobvezne. **UNCERTAIN je referenčni razred**; NE MOREM OCENITI pusti referenčni razred prazen. Nobena ročna ocena ni predizpolnjena. Prejšnje ocene lahko spremenite ali počistite.

Ujemanje napovedanega razreda z referenco se izračuna in shrani v ločeno tabelo `computed_matches`, tudi za skrite odgovore in ponovitve. Ob spremembi ali čiščenju reference se posodobi v isti transakciji. Pri NE MOREM OCENITI, prazni referenci ali neveljavni napovedi je ujemanje prazno; razlog določa `computed_match_status`. Izračun ni nova ročna ocena in ne spreminja prejšnjih ročnih ocen klasifikacije.

Združujejo se samo dobesedno enake, neprazne utemeljitve z enakim veljavnim napovedanim razredom znotraj iste regije in istega SHA vhodnega dokumenta. Presledkov ali besedila se ne normalizira; manjkajočih odgovorov se ne združuje. Ena izrecna ocena skupine v transakciji oceni vse navedene povezane odgovore, tudi ponovitve v drugih zavihkih. Vse identitete in posamezne ocene ostanejo sledljive. Če se stare ročne ocene povezane skupine razlikujejo ali so delno prazne, se to pokaže; samo odpiranje skupine jih ne prepiše.

Odgovori imajo stabilne anonimne oznake A/B/C …; dodatne ponovitve so na ločenem zavihku. Novi modeli dobijo dodatne oznake, obstoječe oznake se ne preštevilčijo. Preslikava model/prompt/seme je enkrat ustvarjena in shranjena za vsak nabor. Gumb jo razkrije na zahtevo. Ob prehodu na drugo regijo se odgovori znova skrijejo; ob ponovnem zagonu imena ostanejo skrita. Celotna izvorna besedila utemeljitev ostanejo nespremenjena, tudi če sama omenjajo model.

Vsak klik in sprememba opombe se takoj zapišeta v SQLite transakcijo skupaj z zgodovino sprememb (`synchronous=FULL`, WAL). V primeru neuspelega zapisa vmesnik ohrani čakajoče spremembe, pokaže napako in omogoči ponovni poskus. Pred zapiranjem počakajte na napis, da so vse spremembe shranjene. Ne uporabljajte več zavihkov hkrati za isto ocenjevanje.

Privzeta baza je **`reviews/manual_review.sqlite3` v korenu projekta**. Mapa `reviews/`, vključno s pomožnimi SQLite datotekami in morebitnimi izvozi, je izključena iz Gita. Pri lastni poti `--db` sami poskrbite za izključitev iz Gita. Baze ni dovoljeno zapisati v izvorni mapi rezultatov ali Florence. Nabori imajo v isti bazi ločene identifikatorje na podlagi manifesta. Za varnostno kopijo po ustavitvi strežnika kopirajte bazo skupaj z morebitnima datotekama `-wal` in `-shm`, ali uporabite izvoz.

CSV vsebuje trenutne ocene po regijah in odgovorih, JSON tudi zgodovino sprememb in anonimno preslikavo. Oba izvoza **vsebujeta resnične identitete modelov in promptov**. Prenos sprožite s povezavama v vmesniku; izvoze shranite v `reviews/`, če jih želite obdržati zunaj Gita.

Izvoz različice 2 loči `reference_class`, `predicted_classification`, `computed_match` (true/false ali prazno), `computed_match_status` in ročno `reason_rating`. Stari `classification_rating` ostaja izključno prejšnja ročna ocena razreda; program ga ne predizpolnjuje ali prepisuje z izračunom. Čas izračuna je `computed_at`, čas ročne ocene `modified_at`. Izračunano ujemanje pomeni samo primerjavo z vašo referenco.

Ob prvem zagonu nove različice se doda tabela izračunov in iz obstoječih referenc dopolnijo izračuni. Stare ročne ocene, opombe, zgodovina, preslikava ter položaj ostanejo ohranjeni. Obstoječa baza se privzeto odpre v referenčnem načinu; izbira načina se nato shranjuje.

Napredka sta ločena: referenčni šteje referenčne odločitve, vključno z NE MOREM OCENITI; napredek utemeljitev šteje ročno ocenjene posamezne povezane odgovore, tudi dodatne ponovitve. Filter **Samo neocenjene** v referenčnem načinu potrebuje samo referenčno odločitev; v pregledu utemeljitev zahteva še ocene vseh utemeljitev. Neocenjene utemeljitve ne zadržijo referenčnega označevanja. Za popravljanje prejšnjih ocen izklopite filter.

## Bližnjice

Kratka vrstica ostaja ob navigacijskih gumbih; **Pomoč / bližnjice** odpre celotna navodila.

| Tipka | Dejanje |
|---|---|
| 1 / 2 / 3 / 4 | FIXED / MOVABLE / UNCERTAIN / NE MOREM OCENITI |
| ← / → | Prejšnja / naslednja regija |
| Shift + ← / → | Prejšnja / naslednja slika |
| Preslednica | Prikaži/skrij odgovore po referenčni odločitvi v načinu Pregled utemeljitev |

Bližnjice ne delujejo med pisanjem opombe, vnosom v polje ali urejanjem besedila, ob odprti pomoči ter med shranjevanjem ali nalaganjem. Držanje tipke ne povzroča ponavljanja ocen. V referenčnem načinu odgovori ostajajo skriti tudi pri preslednici.

## Preverjanje brez uporabniških ocen

```bash
python3 -m unittest tests.test_review_viewer -v
```

Testi uporabljajo začasne slikovne rezultate in začasno bazo; ne zapisujejo v vašo bazo. Preverijo oba formata, nedestruktivno nadgradnjo, spremembo reference in izračunov, dobesedne skupine in skupinsko shranjevanje, oba načina, nadaljevanje ter HTTP/CSV/JSON izvoz. Bližnjice in sintakso vmesnika izvedejo z obstoječo sistemsko knjižnico JavaScriptCore, če je na voljo; sicer je ta test izpuščen. Ta knjižnica ni odvisnost pregledovalnika in je ni treba nameščati.
