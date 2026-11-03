# Omejen test prompta na 40 zamrznjenih vhodih

Odprite index.html. 20 LOCO (4/okolje), 10 AGTEA in 10 trgovinskih THUD;
manifest in izbor desetih ponovitvenih slik sta zamrznjena pred novimi klici.
Izbor temelji samo na SHA256 s fiksno soljo/semenom 42, ne na odgovorih modelov.
heldout160.json je preostalih 160 slik, ki novega prompta ne prejmejo.

Ukaz: `bash scripts/prompt_comparison40.sh run`. Nadaljevanje preskoči vse
shranjene poskuse, tudi napake. Sam izvoz: isti ukaz z argumentom `export`.
Stari prompt: 120 kopiranih celotnih odgovorov. Novi prompt: 120 glavnih
klicev s semenom 42 ter 60 ponovitev s semenoma 43/44 na desetih slikah.
T=0.3, kontekst 8192, največ 2048 tokenov, top_p=0.8, top_k=20,
presence_penalty=1.5; iste Qwen3.5 4B/27B in Gemma 3 27B Q4_K_M uteži.
Florence, predpreizkusi, ogrevanja in ostalih 160 novih promptov niso izvedeni.
Točne zahteve, celotni odgovori, čas, GPU stanje in napake so v responses/;
baseline/ ohranja izvorne stare odgovore. prompts/ in frozen/ vsebujeta
neskrajšana besedila ter identične opisne vhode s SHA256 v manifestu.

prompt-differences.csv: razredi po regijah za stari/novi prompt in semena.
repeatability-regions.csv: primerljive trojice razredov 42/43/44.
repeatability-summary.csv: za vsak model delež regij z enakim razredom pri
vseh treh semenih; imenovalec so samo primerljive trojice, neprimerljive so
ločeno preštete. Regije so lokalni Florence zapisi, tudi skupine/prekrivanja.
To je ponovljivost, ne accuracy ali potrditev fizične identitete/pravilnosti.
Ponovljivost se nanaša na vrnjene veljavne razrede/ID-je. Manjkajoče utemeljitve
so napake celotne JSON sheme, ločeno v response-errors.csv; razredi in surovi
odgovori ostanejo nespremenjeni. Časi so celotni API klici, lahko z nalaganjem.
manual-review-empty.csv ima prazne človeške ocene; ob izvozu se ne prepisuje.
Vsebinske kakovosti modelov ne ocenjujemo. Končni števci: verification.json.
