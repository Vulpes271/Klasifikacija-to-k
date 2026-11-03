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

Najprej podajte referenčni razred ali izberite **NE MOREM OCENITI**, nato posebej odprite odgovore. **UNCERTAIN je razred**; NE MOREM OCENITI pusti referenčni razred prazen. Nobena ocena ni predizpolnjena. Razred in celotno utemeljitev vsakega odgovora ocenjujete ločeno. Prejšnje ocene lahko kadarkoli spremenite ali počistite.

Glavni odgovori so anonimno premešani A–F (A–C pri 200 slikah); dodatne ponovitve so na ločenem zavihku. Preslikava model/prompt/seme je enkrat ustvarjena in shranjena za vsak nabor. Gumb jo razkrije na zahtevo. Ob prehodu na drugo regijo se odgovori znova skrijejo; ob ponovnem zagonu imena ostanejo skrita. Celotna izvorna besedila utemeljitev ostanejo nespremenjena, tudi če sama omenjajo model.

Vsak klik in sprememba opombe se takoj zapišeta v SQLite transakcijo skupaj z zgodovino sprememb (`synchronous=FULL`, WAL). V primeru neuspelega zapisa vmesnik ohrani čakajoče spremembe, pokaže napako in omogoči ponovni poskus. Pred zapiranjem počakajte na napis, da so vse spremembe shranjene. Ne uporabljajte več zavihkov hkrati za isto ocenjevanje.

Privzeta baza je **`reviews/manual_review.sqlite3` v korenu projekta**. Mapa `reviews/`, vključno s pomožnimi SQLite datotekami in morebitnimi izvozi, je izključena iz Gita. Pri lastni poti `--db` sami poskrbite za izključitev iz Gita. Baze ni dovoljeno zapisati v izvorni mapi rezultatov ali Florence. Nabori imajo v isti bazi ločene identifikatorje na podlagi manifesta. Za varnostno kopijo po ustavitvi strežnika kopirajte bazo skupaj z morebitnima datotekama `-wal` in `-shm`, ali uporabite izvoz.

CSV vsebuje trenutne ocene po regijah in odgovorih, JSON tudi zgodovino sprememb in anonimno preslikavo. Oba izvoza **vsebujeta resnične identitete modelov in promptov**. Prenos sprožite s povezavama v vmesniku; izvoze shranite v `reviews/`, če jih želite obdržati zunaj Gita.

Napredek in filter »Samo neocenjene« štejeta regijo kot dokončano, ko ima referenčno odločitev (tudi NE MOREM OCENITI) in obe oceni za vse shranjene odgovore, vključno z dodatnimi ponovitvami. To je zgolj stanje ročnega pregleda.

## Preverjanje brez uporabniških ocen

```bash
python3 -m unittest tests.test_review_viewer -v
```

Testi uporabljajo začasne slikovne rezultate in začasno bazo; ne zapisujejo v vašo bazo. Preverijo oba formata, slepi prikaz, referenčno odločitev, ločene ocene, popravke, ponovno nalaganje, navigacijo in lokalni HTTP/CSV/JSON izvoz.
