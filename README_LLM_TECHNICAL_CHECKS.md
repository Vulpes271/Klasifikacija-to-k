# Tehnično preverjanje shranjenih odgovorov

Oba izvajalnika (`florence_grounding_llm` in `prompt_comparison40`) ter lokalni pregledovalnik uporabljajo `llm_checks.validate_response`. Veljaven celoten odgovor zahteva `done=true`, zaključek `stop`, veljaven JSON brez podvojenih ključev, točne tipe in množico ID-jev, brez podvojenih ID-jev, dovoljen razred in neprazen niz `reason`. `description` se nikoli ne pretvori v `reason`. Nepopoln odgovor lahko še vedno prispeva nedvoumne posamezne klasifikacije za pregled; to ne pomeni veljavnosti celotnega odgovora ali pravilnosti vsebine.

Števci ločijo prejete API odgovore, veljavne celotne odgovore, napake API, napake sheme/zaključka in preskoke. Prejeti odgovori vključujejo veljavne in neveljavne; teh števcev ne seštevajte kot ločenih kategorij. Opozorila o kontekstu so dodatna tehnična dimenzija. Pravilnost vsebine ostaja neocenjena.

## Revizija brez modelov

Iz korena projekta:

```bash
python3 -m llm_object_classifier.audit_llm_results --output-dir "results models/technical_revision_v1"
```

Ukaz prebere shranjene odgovore in preveri njihove izvorne SHA. Ne kliče Ollame ali modelov, ne odpre baze ocen ter ne piše v zgodovinske rezultate. Nova mapa vsebuje `summary.json`, `responses.csv` in `source-sha256.json`. Če se ob ponovitvi revizija razlikuje, izberite novo izhodno mapo. 120 starih odgovorov v primerjavi promptov je podmnožica osnovnih 600, ne 120 novih generacij.

## Prihodnje nadaljevanje

Pred uporabo vsakega obstoječega položaja izvajalnik preveri SHA zamrznjenega dokumenta in identiteto slike, natančni sporočili, model, digest nameščenih uteži, možnosti generiranja ter `think`/`format`. Digest prebere iz metapodatkov Ollame, brez nalaganja modela. Neujemanje ali manjkajoči preverljivi podatki ustavijo nadaljevanje; obstoječi rezultat ostane nedotaknjen. Podrobnosti se zapišejo v podmapo `_technical_checks` ob odgovorih.

Pri starih odgovorih brez lastnega SHA slike se slika lahko preveri prek SHA celotnega zamrznjenega dokumenta, ki vsebuje njen SHA; to je izrecno navedeno. Če digest uteži ali natančni prompt/nastavitve manjkajo in jih ni v shranjenem podpisu, preverjanje ne uspe. Revizija brez Ollame preveri le zgodovinsko skladnost digestov, ne trenutno nameščenih uteži.

Pred novim klicem se preveri proračun konteksta. Ollama v tem izvajalniku nima lokalnega točnega tokenizerja: uporabljena je **konservativna hevristika** enega ocenjenega tokena na UTF-8 bajt obeh sporočil in pogovorne predloge, plus 512 tokenov rezerve. To ni točno štetje ali dokazana zgornja meja. Lahko zavrne vhod, ki bi dejansko ustrezal. Če ocena z rezerviranim izhodom preseže kontekst ali pogovorna predloga ni znana, se položaj shrani kot preskok brez generacije. Kontekst in besedilo se ne spreminjata samodejno. Takšnega preskoka nadaljevanje ne ponovi.

Po klicu se shranijo dejanska `prompt_eval_count` in `eval_count`, preveri vsota z rezerviranim izhodom, dejanska dolžina zaporedja ter poročani kontekst. Manjkajoči podatki so `unverifiable`. Dejanski števci sami ne dokazujejo, da strežnik ni krajšal vhoda. Proračunsko opozorilo se ne pretvarja v napako JSON ali v presojo kakovosti.

Pregledovalnik pokaže napako celotnega odgovora tudi ob regiji z uporabnim razredom. Obstoječih ročnih ocen ne popravlja. Anonimne oznake in zasebna baza ostanejo ohranjene.

## Omejeni testi

```bash
python3 -m unittest tests.test_llm_checks tests.test_review_viewer -v
```

Uporabljajo začasne datoteke, lažni Backend in začasne baze. Štiri znane neveljavne shranjene odgovore preberejo samo za preverjanje; njihove bajte preverijo pred in po branju. Nobene inference ali spremembe dejanskih ocen.
