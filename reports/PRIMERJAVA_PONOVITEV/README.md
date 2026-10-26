# Tri ponovitve slikovnih opisov in besedilnega razvrščanja

Uporabili smo istih 20 THUD slik (10 trgovina + 10 agtea) kot v prejšnjem poskusu.
To so povezani pogledi dveh prostorov, ne 20 neodvisnih prizorov.
Cilj je pogledati, kaj se spremeni med semeni, in ločeno preveriti vpliv kratkega/dolgega
navodila ter BF16/Q4 besedilnega modela na isti shranjeni opis. To je paket meritev
in primerov, iz katerega lahko pripravimo poročilo; novega dolgega PDF ne izdelujemo.

Odpri **[index.html](index.html)** neposredno v brskalniku po prenosu repozitorija.
Vse slike, opisi in podatki desetih primerov so relativno povezani in delujejo brez strežnika.
Na GitHubu je HTML izvorna datoteka; za prikaz prenesi repozitorij ali mapo s primeri.
Deset primerov (5 + 5) izberejo kazalniki razhajanja, napak in gostote, vključno z dvema
stabilnima Florence primeroma. Lastna vizualna presoja je v `manual-review.json` in vsaki
mapi primera; modelove trditve ostanejo ločene od te presoje.

## Dejanski obseg

- VLM: **420 svežih GPU generacij / 420 položajev**.
- LLM: **240 svežih GPU generacij / 240 položajev**, **0 preskokov** zaradi neuporabnega Qwen BF16 r1 vira.
- Diagnostični NVILA poskusi: 22 dodatnih merjenih odgovorov in 2 zaključenih izključenih ogrevanj, ohranjeni posebej; ne štejejo v glavno primerjavo.
- Glavni izbor ni prepis starega cache: fizične generacije imajo enolične ID oziroma strežniške čase. Pilot istega protokola ostane del r1.
- Predhodnih **1971** shranjenih odgovorov je preverjeno nespremenjenih. Ogrevanja in nalaganja so ločena v `warmups.json` in `manifest.json`.

| Model / naloga | r1 OK/vsi | r2 OK/vsi | r3 OK/vsi |
| --- | ---: | ---: | ---: |
| florence / caption | 20/20 | 20/20 | 20/20 |
| florence / dense | 20/20 | 20/20 | 20/20 |
| florence / od | 20/20 | 20/20 | 20/20 |
| gemma3-q4 / description | 18/20 | 16/20 | 16/20 |
| nvila / description | 0/20 | 0/20 | 0/20 |
| qwen35-bf16 / description | 20/20 | 20/20 | 20/20 |
| qwen35-q4 / description | 20/20 | 20/20 | 20/20 |

OK pomeni skladnost z ustreznim izhodnim formatom, **ne vsebinske pravilnosti**.
NVILA ima dobesedno razčlenjen neprazen seznam objektov v **52/60** odgovorih,
tudi kadar celoten zapis ni veljaven po shemi. Te vsebine ne zavržemo.
NVILA več JSON dokumentov/odvečno besedilo ostane napaka celotnega odgovora. Za prikaz
skripta izlušči samo dobesedne objekte iz strogo prebranih JSON dokumentov; ničesar ne
popravlja v veljaven opis. Preostalo besedilo in celotni original ostaneta dostopna.
Neobstoječ razčlenjen seznam ni dokaz, da na sliki ni objektov.

## Protokol in meje primerjave

Qwen BF16/Q4 in Gemma Q4 uporabljajo isti novi `description-v1.txt`, T=0.3,
semena 42/43/44 za r1/r2/r3, kontekst 16384, največ 4096 novih tokenov,
top-p=0.8, top-k=20, presence penalty=1.5. JSON transport je obstoječa Ollama.
NVILA uporablja vsebinsko primerljivo, a izrecneje strukturirano navodilo
`nvila-description-v2.txt`; prvi novi pilot je vrnil prosti tekst in je ohranjen
v `diagnostic-nvila.jsonl`. NVILA je FP16, SDPA, T=0.3, ista semena,
4096 novih tokenov, repetition penalty=1.05; presence penalty ni podprt.
Zaradi različnih promptov/formatov/preprocesiranja razlik ne pripisujemo samo modelu.
Pri NVILA je inicializacija dodatnih tokenov vedno seed42; le generiranje uporablja
42/43/44. Med revizijo odkriti neprimerljivi r2 (in prekinjeno nalaganje r3) sta
ohranjena v diagnostiki, r2/r3 sta izvedena s pravilno ločenima semenoma. R1 je že
uporabil init seed42. Popravljena nalaganja hranijo SHA256 zadnjih 64 vrstic vhodnih
in izhodnih embeddingov; `nvila_initialization_correction` dokumentira popravek.
Pri ponovnem r3 je proces po ogrevanju odpovedal s SIGSEGV. Napaka in neznani čas
nedokončanega klica sta zabeležena; nadaljevanje ni popravilo ali ponovilo shranjenega odgovora.
Vzrok sesutja ni potrjen. Native navodilo se po VILA API pošlje v uporabniškem
sporočilu; dodatna polja `native_*` ohranijo dejanski chat format in njegov privzeti
sistem, raw komponenti `system_prompt/user_prompt` pa ostaneta nespremenjeni.

Florence je izvirni `microsoft/Florence-2-large`, FP16, beam=3,
do_sample=false, max_new_tokens=1024. T=0.3 zanj ni uporabljen in seme pri tem
režimu ni vir vzorčenja; tri dejanske generacije so zahtevane tudi ob enakem odgovoru.
Prejme samo native naloge caption/OD/dense, ne Qwenove JSON sheme.
Ima **180/180** parov z dobesedno enakim odgovorom.

Drugi korak prejme samo **isti zamrznjeni Qwen BF16 r1 opis**, brez slike in referenc.
Datoteke `frozen-inputs/` so objavljene bajt za bajtom, s preverljivim SHA256.
BF16 in Q4, kratki in dolgi prompt brez primerov ter semena 42/43/44 uporabljajo
isti vhod. Napake in preskoki ostanejo v imenovalcih; ne nadomeščamo jih z drugim virom.

- `repeat`: 198/240 veljavnih parov klicev; 153/1127 različnih razredov na primerjanih objektih.
- `prompt_name`: 96/120 veljavnih parov klicev; 152/547 različnih razredov na primerjanih objektih.
- `model_name`: 95/120 veljavnih parov klicev; 113/542 različnih razredov na primerjanih objektih.

To so razlike med napovedmi po istem tipiziranem ID, ne accuracy stabilnosti.
Referenc stabilnosti ni. Imena primerjamo po malih črkah in presledkih; sinonimov,
množin ali skupinskih regij ne enačimo samodejno. Spremembe lastnosti so spremembe
celotnih dobesednih zapisov v skupini istega imena, ne potrjeno iste fizične instance.
Odnosi v besedilu ostanejo v opazovanju; odsotnost strukturiranega polja ni odsotnost odnosa.
Florence okvirji imajo enolično IoU ujemanje. Reference in že določen slovar se uporabljajo
samo pri vrednotenju: `reference-matching.json` hrani vse pare, `reference-summary.*` agregate.
Pokritost anotiranih regij ni celoten precision/recall/mAP; neujemanje ni nujno halucinacija.

GPU je med meritvami v souporabi. Časi vključujejo dejanski obseg posameznega izvajalnika;
Florence combined meri vse tri naloge skupaj brez izrisa. CUDA allocated/reserved in
Ollama size_vram sta različni meritvi. Iz teh časov ne sklepamo o čistem vplivu arhitekture.

## Ponovitev in podatki

Uporabljena so obstoječa Docker okolja in lokalni modeli, brez ponovne namestitve:

```bash
bash scripts/thud_ponovitve.sh prepare
bash scripts/thud_ponovitve.sh all
# Ločeno: pilot/describe --model florence|nvila|qwen35-bf16|qwen35-q4|gemma3-q4
# Describe omogoča tudi --repeat r1|r2|r3; pilot se ohrani kot r1.
bash scripts/thud_ponovitve.sh freeze
bash scripts/thud_ponovitve.sh reason
bash scripts/thud_ponovitve.sh export
bash scripts/thud_ponovitve.sh verify
python3 -m unittest tests.test_separate_stages tests.test_thud_extended tests.test_thud_repetitions tests.test_stage_scripts tests.test_thud_selections -v
```

Zaganjalnik nadaljuje samo manjkajoče klice, ohrani napake in zavrne spremenjene
konfiguracije ali vhode. Vsako novo nalaganje fizično ogreje. GPU modeli tečejo zaporedno.
Veliki lokalni izhodi so `results/ponovitve/`; stare meritve ostanejo v svojih mapah.
Konfiguracije/prompti so `configs/ponovitve/`; točne poslane prompte hrani tudi `prompts/`.

Za **vseh 20** so objavljeni `stage1-responses.jsonl`, `stage2-responses.jsonl`,
zamrznjeni vhodi, manifest, summary in consistency. Deset map `cases/` vsebuje original,
Florence izrise z ID/oznakami za OD in dense vsake ponovitve, ločen referenčni izris,
celotne odgovore/podatke obeh korakov in ročno opombo. Drugi VLM nimajo lastnih okvirjev;
prikazujejo se original in oštevilčeni dobesedni seznami. Dodatno grounding ni izveden.

Iz kompaktnih Git kopij so odstranjena transportna slikovna polja/base64, gostiteljski
argumenti diagnostičnih Docker ukazov (ohranjen SHA256 izvirnika) in Python
tracebacki, laboratorijske poti v strukturnih poljih so relativne. **Besedilo odgovora,
promptov, opisov, objektov in utemeljitev ni skrajšano.** Originalni lokalni zapisi ostanejo
nespremenjeni; `original_record_sha256` jih poveže s kopijami. Nobene uteži ali vendor
knjižnice niso del tega paketa. `verification.json` dokumentira končne preverbe.
