# Ročne ocene 40 slik — 2026-11-04

Izvoz je objavljen po izrecnem naročilu uporabnika. Vseh 40 slik ima shranjene odločitve za vseh 414 Florence regij. `uncertain` je referenčni razred; `unassessable` (NE MOREM OCENITI) pomeni prazno referenčno oznako.

`manual-reviews.csv` in `manual-reviews.json` vsebujeta dejanske človeške reference, opombe, oznake težav Florence ter izrecne ročne ocene odgovorov. JSON ohrani tudi celotno zgodovino sprememb, stabilno anonimno preslikavo in zadnji položaj. Samodejno izračunana ujemanja niso ročne ocene in niso vključena; izračunajo se iz že objavljenih napovedi.

Izvoz je konsistenten posnetek SQLite, preverjen proti izvornemu manifestu in SHA zamrznjenih vhodov. Delovna baza `reviews/manual_review.sqlite3`, njene varnostne kopije in stranske datoteke ostanejo lokalne ter izključene iz Gita. Za izvedbo izvoza ni bila spremenjena nobena uporabniška ocena in niso bili zagnani modeli.
