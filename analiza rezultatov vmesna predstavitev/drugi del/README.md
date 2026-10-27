# Drugi korak: isti zamrznjeni opisi, trije besedilni modeli

Odprite [index.html](index.html) lokalno po prenosu repozitorija. Pregled poteka:
**slika → izvorni VLM in njegova ponovitev → LLM → kratki/dolgi prompt → tri
LLM ponovitve z odločitvami in celotnimi razlogi**. Slika je samo za človekov
pregled; LLM je ne prejme. Prvi korak ni ponovno izveden. PDF ni del te naloge.

## Zaključeni rezultati

Preverjenih je **3240/3240 položajev**: 2807 veljavnih po shemi, 289 napak in
144 preskokov. Opravljenih je **3096 glavnih inferenčnih API poskusov** in devet
arhiviranih neuspešnih kontekstnih poskusov, skupaj **3105**. Prejetih je 3104
Ollama odzivov; en API poskus je vrnil opisani HTTP 500 namesto generiranega
odgovora. Piloti so vključeni, fizičnih ogrevanj ni, uspešnih klicev nismo
ponavljali. Število veljavnih zapisov ni število pravilnih odločitev.

| LLM | Prompt | Veljavni po shemi | Napake | Preskoki | Glavni API poskusi |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen BF16 | kratki | 460 | 56 | 24 | 516 |
| Qwen BF16 | dolgi | 476 | 40 | 24 | 516 |
| Qwen Q4 | kratki | 459 | 57 | 24 | 516 |
| Qwen Q4 | dolgi | 496 | 20 | 24 | 516 |
| Gemma Q4 | kratki | 422 | 94 | 24 | 516 |
| Gemma Q4 | dolgi | 494 | 22 | 24 | 516 |

Pri primerljivih veljavnih odzivih na **istem zamrznjenem vhodu** je strinjanje
razredov med LLM ponovitvami 90,88 % (24.754 objektnih parov), med promptoma
80,90 % (11.442 parov) in med LLM 78,29 % (21.677 parov). To so pogojne mere
strinjanja, ne natančnost; neveljavni odzivi in preskoki so ločeno prikazani
v imenovalcih položajev in neprimerljivih parov. Pari ponovno uporabljajo istih
20 slik, zato niso neodvisna opazovanja.

Med različnimi VLM je nastalo le 686 nepotrjenih parov edinstvenih enakih imen,
med njihovimi ponovitvami pa 8583. To **niso potrjena ujemanja fizičnih objektov**.
Celotne razlike, neujemajoča in dvoumna imena ostajajo v primerjalnih datotekah.
Preverjanje pokriva 540 skupnih datotek po šest položajev, SHA256, dejanske
nastavitve in prompte, 10.688 HTML povezav, prazen obrazec z 3240 vrsticami in
888 nespremenjenih datotek prvega koraka. Preverjena je tudi natančna obnovitev
vseh 3240 zapisov samo iz javnega paketa, brez lokalnih map izvajalnika.

## Zamrznjeni vhodi

V [frozen/index.json](frozen/index.json) je vseh 180 kombinacij 20 slik,
treh VLM (Florence-2-large, NVILA-8B, Qwen3-VL-8B-Instruct) in treh njihovih
ponovitev. Vsaka datoteka `frozen/frame-NNNNNN.json` je zamrznjena pred inferenco,
ima izvorne poti in SHA256 vseh uporabljenih native odzivov ter stabilne lokalne
ID-je. SHA256 celotne datoteke in kanoničnega opisnega dokumenta sta v indeksu.
Vseh 18 LLM kombinacij za isti vhod bere **isto datoteko in iste ID-je**.
Tudi uporabniški prompt je pri kratkem in dolgem sistemskem promptu enak.

Uporabnih je **172 vhodov**. Pri osmih NVILA odgovorih ni zanesljivo izluščljivega
seznama; teh 8 × 18 = **144 položajev je preskočenih**, brez nadomestitve z drugo
ponovitvijo ali izmišljanja objektov. Preostalih **3096 položajev** je namenjenih
dejanskim klicem. Skupaj je načrtovanih **3240 položajev**, vključno s preskoki.
Dejanska zaključna števila in statusi so v [verification.json](verification.json).
Pred zaključkom ta datoteka izrecno vsebuje `complete=false`.

- **Florence:** obstoječi native pretvornik združi celoten caption ter vse OD in
  dense zapise. Vsak zapis ohrani `source_task` in `source_region_id`. Zaporedni
  transportni ID se ne ponovi v dokumentu; ID regije lahko velja samo znotraj
  svoje naloge. Prekrivanje med OD/dense ali znotraj naloge **ne potrjuje različnih
  fizičnih objektov**. Obe nalogi ostaneta, brez tihega združevanja. Bbox je
  besedilni podatek o Florencejevi napovedi, ne referenca. Celotni native odzivi
  so tudi v `native_task_outputs`. `not_visible` je transportna oznaka za
  nepodano pritrditev; `identifiable=true` pomeni, da je native model regijo
  poimenoval, ne človekove potrditve zaznave.
- **NVILA:** uporabljeno je obstoječe dobesedno izluščenje celih JSON dokumentov
  iz zaporedja. Nobenega popravljanja narekovajev ali dopolnjevanja trditev.
  Obstoječi transportni pretvornik pripravi seznam iz samo tako izluščenih
  objektov. `native_raw_text` ohrani **celoten originalni neveljavni odgovor**,
  tudi neporabljeni konec. Vsak tak vhod je označen kot delni in nevalidiran.
  Observation je dobesedni celoten izvorni objekt kot JSON tekst. Transportni
  ID je zaporedni indeks; `source_object_id` ohrani izvirni ID, če je prisoten.
  Manjkajoče ali neveljavne evidence se prenesejo kot `not_visible`, manjkajoči
  identifiable kot true samo v smislu vrnjene oznake. To niso nove vidne trditve.
  Omejitve so izrecno vključene tudi v vhodni dokument za LLM.
- **Qwen3-VL:** podan je celoten prej validirani opis z nespremenjenimi
  originalnimi lokalnimi ID-ji, opisi, lokacijami in odnosi.

Izvorni VLM lahko halucinira. LLM odloča na njegovem opisu, ne na fotografiji.
Ohranjen neveljaven zapis ni popravljen VLM rezultat; validiran transportni
dokument ne pomeni potrjene vsebinske pravilnosti.

## LLM, prompti in ponovitve

Uporabljeni so že nameščeni `qwen3.5:4b-bf16`, `qwen3.5:4b-q4_K_M` in
`gemma3:4b-it-q4_K_M` prek obstoječega Ollama izvajalnika in preverjanja GPU.
Dejanski quantization, digest, različica strežnika, predloga in model_info so v
`provenance/run-manifests/`; vsak odziv ima še lastno GPU stanje in VRAM.

Obstoječa sistemska prompta brez vsebinskih primerov sta nespremenjena v
`prompts/short-no-examples.txt` in `prompts/long-no-examples.txt`. Splošno
uporabniško navodilo je v `prompts/reasoning.txt`. Oba uporabljata iste definicije
fixed/movable/uncertain in zahtevata vsak podani ID točno enkrat ter kratko
angleško utemeljitev. Shematski zapis strukture in ohranitev tipa ID nista
vsebinska primera razvrstitve objektov.

Vsi klici: T=0.3, **LLM-r1/r2/r3 = seed 42/43/44**, max_tokens=4096,
privzeti num_ctx=16384, top_p=0.8, top_k=20, presence_penalty=1.5, num_gpu=999,
think=false, format=json. To so ponovitve vzorčenja med semeni, ne test
identičnega semena. **VLM-r1/r2/r3** ločeno pomeni že shranjene ponovitve prvega
koraka. Vhodni opis se med LLM ponovitvami ne spremeni.

Devet dolgih Florencejevih vhodov uporablja **num_ctx=32768 pri vseh LLM,
obeh promptih in vseh LLM ponovitvah**. Izbor je v
`execution-context-policy.json`: zamrznjeni uporabniški prompt presega 30.000
znakov. Manjši kontekst je pri teh dokumentih omejeval odgovor oziroma krčil
vhod na strežniku. Zamrznjenih vhodov nismo spreminjali ali krajšali.
Pred popravkom je nastalo devet neuspešnih poskusov s kontekstom 16384; vsi so
ohranjeni ločeno v `diagnostics/context-16384/`. Noben od njih ni bil uspešen.
Ponovili so se samo ti neuspešni tehnični poskusi. Glavni obseg ostaja 3096
klicev; devet predhodnih poskusov se prišteje dejanskemu fizičnemu številu.

Najprej je bil opravljen en merjeni pilot za vsak LLM, na istem prvem vhodu.
Piloti imajo isti protokol kot celotni poskus in štejejo kot shranjeni položaji
short/LLM-r1. Dodatnih fizičnih ogrevanj ni. Nadaljevanje preskoči vse že shranjene
zapise, tudi napake; noben uspešen klic se ne izvaja znova. Zgoraj navedena
izrecna izjema za neustrezen kontekst je arhivirana in se ne skriva. Napake sheme ali
manjkajoči razlogi ostanejo dejanski rezultati, brez tihih popravil ali poskusov,
da bi jih nadomestili z drugačnim odgovorom.

## Celotni odzivi in dejanski prompti

V `responses/slika/VLM/vlm-rN/LLM/responses.json` je šest celotnih zapisov tega LLM,
po ključih `short|long` in `r1|r2|r3`. Zapis vsebuje **celoten izvorni Ollama
response**, tudi celoten `response.message.content`, napake, syntactic candidate,
sprejeti parsed rezultat, vse nastavitve zahteve in GPU podatke. Lokalni izvirni
zapisi izvajalnika v `results/stage2-presentation/` ostanejo nespremenjeni;
njihove poti in SHA256 so pri vsakem objavljenem zapisu.

Da isti dolgi vhod ni večkrat podvojen v vsaki datoteki, so odstranjene samo
**preverjene podvojene kopije** teksta: zgornji `content` je isti kot
`response.message.content`, request.messages se rekonstruira iz referenc na
celotni shranjeni prompt. Nobeno besedilo modelovega odgovora se ne krajša.
Dejanski uporabniški prompti so v `user-prompts/`, sistemski v `prompts/`.
Datotečni SHA256 in SHA256 dejanskega sistemskega sporočila sta ločena, ker
obstoječi izvajalnik sistemski prompt ob branju strip-a. Referenca vsebuje ta
izrecni korak; uporabniški prompt se uporabi bajtno nespremenjen. Izvoz preveri
enakost z dejansko poslano zahtevo. V zahtevi ni slik ali referenčnih oznak.

Čas je dejanski celotni HTTP klic, vključno z morebitnim hladnim nalaganjem.
Ollama `load_duration`, `prompt_eval_duration`, `eval_duration`, števila tokenov
in total_duration ostanejo v celoti ohranjeni, zato je nalaganje mogoče ločiti.
`gpu.size_vram` je trenutna procesna alokacija modela po klicu, **ni izmerjeni
peak VRAM**. Vzorci celotnega deljenega GPU so ločen podatek in ne pomenijo
porabe samo tega modela. Druga obremenitev GPU vpliva na čase; niso izoliran
arhitekturni benchmark. Modeli se izvajajo zaporedno.

Po začetnih serijskih klicih je bil uporabljen ločen Ollama vsebnik **iste
lokalne slike in različice 0.40.4**, z istim modelnim volumnom samo za branje.
Prvotni strežnik in uteži ostanejo ohranjeni. Štirje odjemalski delavci obdelujejo
ločene mape prompt/LLM-ponovitev istega modela; 16k in 32k vhodi so razporejeni
v ločeni fazi, da se velikost konteksta ne menja med hkratnimi zahtevami.
Odloženi položaji ostanejo manjkajoči, ne postanejo lažni preskoki.

`OLLAMA_NUM_PARALLEL=4` je **zahtevana nastavitev**, ne zagotovilo dejanske
sočasnosti. Strežnik za arhitekturo qwen35 izrecno opozori, da hkratnih zahtev
ne podpira, in uporabi `n_seq_max=1`. V tem primeru se odjemalci razvrstijo v
čakalno vrsto. `parallel_slots_requested` in `parallel_slots_actual` sta zato
ločena; dokazi so v `provenance/runtime-parallel.json` in zajetem dnevniku.
Pri izvirnih serijskih klicih je dejansko mesto eno. Pri drugem pogonu se
dejanska sočasnost zabeleži iz njegovega dnevnika, ne ugiba iz nastavitve.
Za Gemma je potrjeno `n_seq_max=4` pri obeh kontekstih: skupni kontekst
65536 oziroma 131072 ter štiri mesta s po 16384 oziroma 32768 tokenov.

Celotni HTTP čas poznejših klicev lahko vključuje tudi čakanje za drugimi
zahtevami. Za ločitev dekodiranja uporabite `response.eval_duration` in
`mean_decode_seconds`; seštevek prekrivajočih se HTTP časov ni trajanje celotnega
poskusa. Spreminjanje načina streženja in deljen GPU sta omejitvi časovne
primerjave. Besedilni vhodi, prompti, semena in dekodirni parametri ostanejo isti.

## Primerjave in omejitve

`summary.csv/json` vsebuje statuse, dejanske čase in razdelitev odločitev za vse
skupine. `comparison-summary.csv/json` povzame primerjave; celotne spremenjene
odločitve in razlogi ostanejo v `comparisons/*.jsonl`, kompaktne tabele v CSV:

- `llm-repeats`: trije pari LLM ponovitev, isti zamrznjeni vhod/LLM/prompt;
- `prompts`: kratki proti dolgemu promptu, isti vhod/LLM/LLM-seme;
- `llms`: trije pari LLM, isti vhod/prompt/LLM-seme;
- `vlm-repeats`: različni VLM izhodi istega VLM na isti sliki;
- `vlms`: različni VLM na isti sliki in istem nominalnem indeksu VLM ponovitve.

Prve tri primerjave uporabljajo ID samo zato, ker je **SHA256 celotnega
zamrznjenega vhoda enak**; preverja se tudi tip ID. Neveljavni odzivi in preskoki
ostanejo v imenovalcu števila položajev/parov, niso pa prisilno veljavne
klasifikacije. Agreement ima izrecen imenovalec primerljivih objektnih zapisov.

Pri različnih VLM in njihovih ponovitvah se lokalni ID **ne uporablja za
ujemanje**. Dopustni so samo kandidati z istim normaliziranim imenom, ki se
pojavi natanko enkrat na vsaki strani; normalizacija sta mala začetnica in
presledki. Vsak tak par je **UNCONFIRMED**, ne potrjena identiteta fizičnega
objekta. Ponavljajoča imena ostanejo dvoumna in neujemajoča. Sinonimov, množin,
skupin in delno ujemajočih opisov se ne združuje. Podani so tudi neujemajoči in
dvoumni izrazi; nič kandidatov pomeni nedoločeno strinjanje, ne 100 %.

Razlike odločitev pri različnih VLM lahko izvirajo iz drugačnih opisov,
granularnosti, pokritosti, halucinacij in navideznega ujemanja imen. Florence
lahko isto stvar opiše v več regijah; štetje odločitev zato ni štetje potrjenih
fizičnih objektov. Razporeditev fixed/movable/uncertain ni objektivna natančnost.
**Strinjanje ni pravilnost.** Referenčnih razredov stabilnosti nimamo; ta paket
ne izračunava accuracy. `manual-review-empty.csv` je prazen obrazec za poznejši
človekov pregled, ločen od modelovih odločitev.

## Ponovitev in nadaljevanje

Na obstoječem laboratorijskem okolju, iz korena repozitorija:

```bash
bash scripts/thud_stage2_presentation.sh freeze  # idempotentno, brez slik/GPU
bash scripts/thud_stage2_presentation.sh pilot   # največ en nov klic za vsak LLM
bash scripts/thud_stage2_presentation.sh run     # samo manjkajoči položaji
bash scripts/thud_stage2_presentation.sh export  # samo skriptna analiza/HTML
bash scripts/thud_stage2_presentation.sh verify
python3 -m unittest tests.test_thud_stage2_presentation tests.test_thud_repetitions tests.test_thud_stage1_presentation
```

Izolirani način na istem že nameščenem okolju:

```bash
bash scripts/thud_stage2_parallel.sh prepare
bash scripts/thud_stage2_parallel.sh run
bash scripts/thud_stage2_parallel.sh stop
```

Ta način uporablja iste konfiguracije kot logično storitev, dejanski naslov
izvajanja pa eksplicitno shrani v vsak novi odziv in modelne metapodatke.

`THUD_OLLAMA_URL` je po potrebi mogoče podati eksplicitno. Konfiguracije in
zamrznjeni vhodi so nespremenljivi; spremenjen prompt ali SHA zavrne nadaljevanje.
Izvoz uporabi ohranjene lokalne izvirnike obstoječega stage2 izvajalnika;
če teh v svežem checkoutu ni, uporabi preverjene objavljene skupne zapise.
Objavljeni odzivi in skupni prompti vsebujejo vse podatke za poznejšo analizo;
za pregled rezultatov inference ali modeli niso potrebni. Uteži, Docker okolja
in model cache ostanejo lokalno. Predhodni rezultati prvega koraka se ohranijo.

Za poznejše poročilo se lahko objavljeni skupni zapisi naložijo tudi brez
lokalnih map izvajalnika: `load()` v primerjalnem modulu v svežem Git checkoutu
rekonstruira dejansko zahtevo iz preverjenih celotnih promptov in ohranjenih
odzivov. Za preverjanje popolnosti modeli niso potrebni. Vsaka primerjava še
vedno uporablja zamrznjeni SHA256. Če manjka en ID ali utemeljitev, je celoten
odziv po uporabljeni shemi neveljaven; njegove delne napovedi ostanejo v surovem
odgovoru oziroma kandidatu, niso pa tiho vključene v glavne mere strinjanja.
20 slik so povezani pogledi prostorov, ne 3240 neodvisnih opazovanj.

`responses-index.json` povezuje vsak skupni zapis s SHA256 in številom shranjenih
položajev. Ob zaključku je 540 takih datotek po šest položajev. Pri pripravi
commita se preveri tudi, da so dejansko vključene v Git; zato je javna mapa
poimenovana `responses/`, saj projektov splošni `results/` ostaja lokalno ignoriran.

Obstoječa shema preverja prisotnost in tip utemeljitve, ne njene besedne dolžine.
Zato agregati ločeno štejejo utemeljitve nad navodilom 12 besed (preprosto štetje
po presledkih); tudi taka meritev ni pravilnost klasifikacije. Celotne razloge
vedno ohranimo. Strežnik uporablja tudi predpomnjenje predpone KV: na primer pri
istem dolgem dokumentu poroča 18.477 vhodnih tokenov, obdelava predpone pa je pri
naslednjem klicu krajša. To je ponovno uporabljena predpona, ne kopiran odgovor;
vsaka generacija ima svoj izvorni čas odgovora.

Prazen obrazec pred izpolnjevanjem kopirajte v svojo datoteko. Če med začasnim
pregledom vseeno vnesete presoje neposredno v `manual-review-empty.csv`, jih
naslednji izvoz pred obnovo prazne predloge ohrani lokalno v
`results/stage2-presentation/manual-review-preserved/manual-review-saved-SHA.csv`.
Tak uporabnikov vmesni pregled se ne objavlja samodejno namesto prazne predloge.
Nadrejeni `verification.json` je nespremenjeni posnetek prvega koraka; za ta
poskus velja `drugi del/verification.json`.

`calls.csv` ima ob zaključku 3249 vrstic: 3240 glavnih položajev in devet
izrecno ločenih kontekstnih poskusov. Predvidenih je 3105 dejanskih API poskusov
z diagnostiko; 144 glavnih vrstic
je preskok brez klica. `attempt_role` loči glavni poskus od diagnostike.
`classifications.csv` razširi veljavne glavne odzive po objektnih ID-jih in
za napako oziroma preskok ohrani statusno vrstico. Tip ID je naveden posebej,
zapis `object_id_json` pa ohrani razliko med številom in nizom. To niso nove
neodvisne slike ali referenčne ocene pravilnosti.

API poskus in dokončan odgovor sta ločena podatka. `response_received` v
`calls.csv`, `completed_response_records` v agregatih in zaključni števec
`main_attempts_without_full_response` pokažejo tudi prekinitve strežnika.
Pri Qwen Q4 / kratki prompt / LLM-r1 / vhod 61 je strežnik vrnil HTTP 500:
`prediction aborted, token repeat limit reached`. Celotna napaka je ohranjena;
odgovora in njegovih tokenov ali GPU alokacije nismo izmislili. Tudi ta napaka
je shranjen položaj in se ne ponavlja. Zaključna števila so v preverjanju.
