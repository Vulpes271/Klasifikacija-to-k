# Navodila za Codex: THUD, opis slike in stabilnost objektov

To datoteko preberi kot delovno nalogo. **Najprej izvedi 1. del do poročila;
2. del je nadaljevanje, ko uporabnik izrecno naroči njegov zagon.** Ne ostani
pri načrtu: na ciljnem Linux računalniku pripravi manjkajočo kodo, zaženi
poskuse, preveri rezultate in izdelaj poročilo. Piši preprosto, komentirano
kodo; razloži namen modulov, vhodov in izhodov. Ne dodajaj nepotrebnih ogrodij.

## Izhodišče in cilj

- Vmesna predstavitev je **21. 10. 2026**. Prednost imajo preverljivi dejanski
  rezultati in razumljivi primeri; širitev modelov ne sme zadržati poročila.
- Repozitorij: `Vulpes271/Klasifikacija-to-k`, izhodiščna veja
  `feature/separate-vision-stages`. Najprej preglej trenutno vejo, lokalne
  spremembe in morebitna `AGENTS.md`; ohrani uporabnikovo delo.
- Okolje: `~/jakob/Klasifikacija-to-k`, Linux, RTX 5090 32 GB, Docker Compose
  in Ollama. Uporabi `compose.yaml:compose.gpu.yaml`. Predhodno je BF16 že
  deloval na GPU, vendar uporabo preveri pri novih poskusih.
- Dolgoročni cilj: **slika → seznam objektov in podroben opis z odnosi →
  besedilni LLM → fixed / movable / uncertain z utemeljitvijo** za robotsko karto.
  Trenutna negibnost na eni sliki še ne pomeni dolgoročne stabilnosti.
- Dva vključena nabora: `data/thud_dense_test50/objects.jsonl` (trgovina,
  50 slik, 678 regij) in `data/thud_agtea_test50/objects.jsonl` (agtea,
  50 slik, 1021 regij). Preveri štetja, poti in SHA256. Slike so različne,
  a povezani pogledi istih prostorov, ne 100 neodvisnih prizorov.
- Oznake vsebujejo vrste in pravokotnike, **ne referenc stabilnosti**.
  `Products` je lahko skupinska regija; ID velja le znotraj slike.

## Kaj že obstaja

| Zaporedje | Koda in nastavitve | Namen |
| --- | --- | --- |
| 0 | `image_dataset.py`, `data/*/objects.jsonl` | izbor in preverjanje slik |
| 1 | `stage1_describe.py`, `description_prompts.py`, `configs/stage1/`, `scripts/stage1_describe.sh` | opisovanje s slikovnim modelom |
| 2 | `stage2_reason.py`, `configs/stage2/`, `configs/prompts/`, `scripts/stage2_reason.sh` | razvrščanje shranjenih opisov brez slike |
| skupno | `vision_backends.py`, `experiment.py`, `description_report.py` | API, sled poskusa, HTML |

Preberi `README.md`, `docs/WORKFLOW.md` in teste. Obstoječi prvi korak ob vhodu
`objects.jsonl` **modelu pošlje referenčne oznake in okvirje**: to je opisovanje
znanih regij, ne samostojna detekcija. Podpira tudi vhod brez `objects`.
Obstoječi opisovalni prompt nima vsebinskih primerov; ne poročaj o njihovi
odstranitvi, če jih sploh ni bilo. Primeri razredov so v promptu drugega koraka.
NVILA konfiguracija je samo predloga za zunanji API, ne nameščen model.
Stare poskuse NYU ohrani ločeno kot zgodovino, ne mešaj jih z novimi THUD rezultati.

## 1. del — poskusi in osnutek poročila za predstavitev

### A. Priprava in organizacija

1. Preveri podatke, Docker, modele, GPU in obstoječe rezultate. Zaženi obstoječe
   teste in najprej preizkusi po dve sliki iz vsakega nabora. Ne dovoli tihega
   CPU nadomestila. Zabeleži različice, dejansko natančnost, VRAM in Git commit.
2. Dopolni obstoječo ločitev, brez podvajanja cevovoda. V `scripts/`, `configs/`
   in dokumentaciji jasno označi vrstni red: izbor → opis/detekcija → reasoning
   → vrednotenje → poročilo. Dodaj en preprost zaganjalnik za celoten poskus,
   dnevnik napredka in možnost nadaljevanja brez prepisovanja rezultatov.
3. Pred polnim zagonom izmeri pilot in zapiši oceno trajanja/števila klicev.
   Najprej zagotovi osnovno primerjavo na vseh 100 slikah, nato dodatne teste.
   Če čas ne dopušča vsega, poročaj o točnem dokončanem obsegu in napakah;
   ne predstavljaj načrtovanih meritev kot izvedenih.

### B. Modeli in poštena primerjava prvega koraka

Obvezna osnovna primerjava, modeli zaporedno na istem GPU:

| Model | Namen |
| --- | --- |
| `qwen3.5:4b-bf16` | nekvantizirana referenca |
| `qwen3.5:4b-q4_K_M` | vpliv kvantizacije istega modela |
| `gemma3:4b-it-q4_K_M` | druga družina slikovnega modela pri podobni velikosti |
| `Efficient-Large-Model/NVILA-8B` | dodatno, če uspe preverjen zagon zunanjega strežnika |

Pred prenosom preveri točne tage in podporo slikam. NVILA pripravljaj v ločenem
okolju; za začetno diagnostiko porabi največ približno 30 minut, nato ob oviri
shrani razlog in nadaljuj osnovni poskus. Ne preimenuj 8B v 15B zaradi API omejitev.
Viri: [Gemma](https://ollama.com/library/gemma3:4b-it-q4_K_M),
[NVILA uteži](https://huggingface.co/Efficient-Large-Model/NVILA-8B),
[uradna koda](https://github.com/NVlabs/VILA). Splošna samodejna navodila za
streženje niso dokaz, da slikovni API deluje; preveri dejansko zahtevo s sliko.

Izvedi dve **ločeno poimenovani** meritvi na obeh naborih po 50 slik:

- **Samostojno zaznavanje:** model prejme samo sliko in splošno navodilo;
  ne prejme referenčnih imen, ID-jev, okvirjev ali anotacijskih poti. Naredi
  ločen vhodni manifest, reference ohrani samo za vrednotenje. Zahtevaj seznam
  vidnih objektov, podroben opis scene, lokacije in vidne medsebojne odnose.
- **Opisovanje danih regij:** obstoječi anotirani vhod. To primerja kakovost
  opisov istih regij; kopiranje podanega imena ni uspešna detekcija.

Osnova: enak podroben prompt, temperatura 0, seed 42 kjer podprt, enaki slikovni
bajti in primerljiv kontekst/izhodni limit (izhodišče THUD: 16384/8192).
Zabeleži nepodprte parametre in različna notranja predprocesiranja. Primerjave
družin pri različnih natančnostih ne pripisuj samo arhitekturi.

### C. Prompti, temperatura in razvrščanje

Vnaprej fiksiraj 10 slik iz vsakega nabora, porazdeljenih čez seznam; isti izbor
uporabi pri vseh dodatnih testih. Shrani izbor, izvirne prompte in razlike med njimi.

- **Prvi korak:** na Qwen BF16 primerjaj kratek in podroben prompt brez
  vsebinskih primerov, z isto izhodno shemo in istimi obveznimi podatki.
  Nato na izbranem promptu primerjaj temperature 0, 0.3, 0.7 pri seed 42.
  Za oceno variabilnosti na tem podnaboru dodaj seed 43 pri temperaturi 0.3.
  Nastavitve spreminjaj ločeno, ne naredi nepotrebnega kartezičnega produkta.
- **Drugi korak:** zamrzni isti nabor opisov iz ene osnovne izvedbe Qwen BF16.
  Primerjaj `long-examples`, `long-no-examples`, `short-no-examples` na BF16
  pri T=0.3, seed 42. Prva dva se razlikujeta **samo** po bloku vsebinskih
  primerov; definicije razredov in JSON shema ostanejo enake. Kratka različica
  ohrani isti pomen. Shrani vse tri prompt datoteke v Git.
- Nato na izbrani različici drugega koraka preizkusi T=0, 0.3, 0.7 in še
  seed 43 pri T=0.3. Q4 ponovitev je dodatna, če ostane čas. Drugi korak
  nikoli ne prejme slike ali referenčnega razreda stabilnosti.
- Pokaži spremembe `fixed/movable/uncertain` in kratke utemeljitve, posebej
  pri vratih, slikah, policah in pohištvu. Brez primerov je prednost **ob
  primerljivi kakovosti**, ne vnaprej določeni sklep. Strinjanje ni pravilnost.

### D. Meritve, preverjanje in izhodi

- Hitrost: čas na sliko (mediana, p90, povprečje), napake/timeouti, število
  izhodnih tokenov in tokens/s, kjer API omogoča. Loči nalaganje modela,
  ogrevalni klic in merjene klice; ogrevanje naj ne šteje v kakovostni vzorec.
  Zabeleži tudi end-to-end čas in omejitve primerjave različno dolgih odgovorov.
- Kakovost: izpuščeni, napačni in izmišljeni objekti, pravilne lokacije/odnosi,
  vidni dokazi in veljavnost JSON. Pripravi preglednico za ročni pregled na
  istem podnaboru 20 slik, s sliko in referenčnimi oznakami. Loči samodejne
  primerjave imen od ročno potrjenih zadetkov; skupinske/nepopolne oznake
  niso zanesljiva celotna resnica. Ročno nepreverjeno označi kot osnutek.
- Precision/recall/F1 le z dokumentiranim ujemanjem in ustreznimi referencami.
  Brez napovedanih okvirjev ne računaj detekcijskega mAP. Brez ročnih razredov
  stabilnosti ne navajaj accuracy razvrščanja. Ne izmišljaj referenc ali meritev.
- Vsak odgovor shrani surov in razčlenjen z modelom, nastavitvami, promptom,
  SHA256 slike/opisa in časom. Napake in podvojene JSON ključe ohrani in zavrni;
  ne popravljaj odgovora tiho ali izločaj neuspehov iz poročanega imenovalca.
- Razširi smiselne teste: brez uhajanja anotacij v samostojno zaznavanje,
  stage2 brez slike, enaki vhodi med različicami, shranjene napake, pravilni
  izračuni meritev. Zaženi teste in dejanski GPU preizkus na tem računalniku.

Ustvari `reports/THUD_VMESNO/porocilo.tex` in prevedeni `porocilo.pdf`
(slovenščina, približno 4–6 strani), podatkovne tabele CSV/JSON in slike.
Dodaj ponovljiv ukaz oziroma skripto za generiranje/prevajanje, brez ročnega
prepisovanja številk. PDF vizualno preveri. Če prevajalnik manjka, uporabi
izolirano orodje/kontejner in dokončaj PDF; morebitno dejansko oviro jasno navedi.

Poročilo naj vsebuje cilj, shemo postopka, podatke, modele/natančnosti, nastavitve,
primerjavo hitrosti in kakovosti, vpliv promptov ter omejitve. Izberi 3–4 zanimive
dejanske slike iz obeh naborov: uspešen primer, gost/zahteven prizor in napaka
ali nestrinjanje. Ob vsaki pokaži fotografijo, primerjalno tabelo modelov in
kratke dejanske opise/utemeljitve, tudi napake. Razloči modelov besedni razlog
od lastne presoje; ne zahtevaj skritega notranjega razmišljanja modela.
Zaključi z utemeljeno izbiro naslednjega poskusa. To je **osnutek za poznejši
pregled in dopolnitev**, ne dokončno potrjeno znanstveno vrednotenje.

V Git dodaj kodo, prompte, konfiguracije, dokumentacijo in majhen samostojen
paket poročila (PDF, LaTeX, uporabljene slike in agregate). Velike surove izhode
ohrani lokalno z jasnimi potmi. Ne briši rezultatov ali stare kode, ki jih
potrebujejo pretekli poskusi; očitno neuporabljeno podvajanje odstrani šele po
preverbi sklicev/testov. Naredi commit in push na delovno vejo; ob težavi z
avtentikacijo ohrani lokalni commit in povej, kaj manjka. Na koncu podaj dejanske
ukaze, poti do poročil, dokončane teste in odprte omejitve.

## 2. del — nadaljevanje: bogatejši samostojni prvi korak

Ta del izvedi po uporabnikovem naročilu, na podlagi ugotovitev 1. dela.
Nadgradi prvi korak in primerjaj iste tri osnovne modele ter NVILA 8B, če je
delujoč. Izhodišče naj bo najboljši prompt brez primerov, če meritve to podpirajo.

Želeni strukturirani izhod: podroben `scene_description`, `objects` z lokalnim
ID, vrsto, videzom, lokacijo, zakritostjo, vidnimi podporami/stiki in dokazi
namestitve ter ločen seznam `relations` med ID-ji (npr. na, pod, ob, pred,
pritrjeno na — samo če vidno). Nedoločljivo izrecno označi; ne sklepaj o
skritih vijakih, teži ali trajnosti. Ne zamenjuj leve/desne strani slike s
svetovnimi koordinatami. Ohranjen naj bo tudi izvirni surovi odgovor.

Če dodamo napovedane `bbox_xyxy`, zahtevaj koordinatni sistem in preveri okvirje
znotraj slike; izriši označene slike. VLM opis lokacije ni sam po sebi meritev
natančne lokalizacije. Napovedane regije ovrednoti z IoU in dokumentiranim
ujemanim parom napoved/referenca; skupinske oznake obravnavaj posebej. Šele če
meritve pokažejo potrebo, predlagaj ločen detektor, brez nepotrebne nove odvisnosti.

Drugi korak mora dobiti ta shranjeni opis, objekte in odnose, ne slike. Posodobi
validatorje in kompatibilnost, teste, HTML s slikami/oznakami ter meritve. Ohrani
ločitev dokazov iz slike od prihodnje presoje stabilnosti in dejstvo, da lokalni
ID-ji niso sledenje skozi čas. Poročilo dopolni z dejanskimi novimi rezultati,
primerjavo izboljšav in stroškom v času/VRAM. Ohranitev razumljive kode in
ponovljivih poskusov ima prednost pred širino funkcionalnosti.
