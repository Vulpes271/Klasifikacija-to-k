# Industrijski dodatek za vmesno poročilo: LOCO

## Cilj in obseg
Obdelaj že pripravljene slike `data/loco_test30` na tem Linux računalniku z GPU.
Primerjaj opisovanje in zaznavanje z VLM ter nato razvrščanje fixed/movable/uncertain
z LLM. Končni paket mora omogočiti ročen pregled in dopolnitev obstoječega poročila.
Delaj samostojno do preverjenega push na trenutno vejo. Ne ponavljaj starih THUD
testov, ne briši starih rezultatov in ne prenašaj celotnega LOCO arhiva ali novih modelov,
če so zahtevane uteži že lokalno. Ne spreminjaj obstoječih rezultatov.

## 1. Najprej uporabi obstoječo kodo
Na kratko preberi ta navodila, `data/loco_test30/README.md` in ustrezne dele:
`llm_object_classifier/thud_extended_common.py`, `thud_florence.py`, `thud_nvila.py`,
`thud_qwen3_vl.py`, `thud_stage1_presentation.py`, `thud_stage2_presentation.py`.
Preveri dejanska imena modelov, shranjene konfiguracije in zagonske ukaze v obstoječem
paketu v `analiza rezultatov vmesna predstavitev/`. Ne nalagaj vseh surovih odzivov v
kontekst: povzetke in preverjanja izračunaj s skripto; ročno preberi le izbrane primere.

Ponovno uporabi runnerje, parsanje, merjenje časa in izvoz. Dodaj samo potrebne
argumente oziroma tanek adapter za LOCO vhod in izhodno pot. Qwen runner in izvozi
imajo trenutno nekatere poti vgrajene; parametriziraj jih z obstoječimi privzetimi
vrednostmi, da stari ukazi delujejo. Ne kopiraj celotnih modulov v nove različice.
Nova povezovalna koda naj bo kratka, komentirana in ločena od starih eksperimentov.

Uporabi `load_images('data/loco_test30/discovery.jsonl', limit=30)`; privzeti limit je
20, zato izrecno zahtevaj vseh 30. Pretvori v `native_selection` shemo: dataset,
index, frame_id, absolutni image_path, image_sha256, image_size, subset20=false.
Shrani manifest v `results/loco/inputs.json`. Referenčnih oznak iz `objects.jsonl`
ali COCO ne podajaj modelom, niti v opis ali označeno vhodno sliko.

## 2. Prvi del: tri VLM, trikrat
Uporabi iste različice, uteži in obdelavo kot pri obstoječi primerjavi:
- microsoft/Florence-2-large;
- Efficient-Large-Model/NVILA-8B;
- Qwen/Qwen3-VL-8B-Instruct.

Za vseh 30 slik naredi tri dejanske ponovitve vsakega modela (r1/r2/r3).
Pri vzorčenju uporabi temperature 0.3, seeds 42/43/44 ter druge obstoječe nastavitve.
Florence ohrani svoje izvorne naloge opis/OD/dense in deterministično generiranje;
trije enaki odzivi so veljaven rezultat. Ne dodajaj nepodprtih parametrov.
Za NVILA/Qwen uporabi isti obstoječi podrobni opisni prompt. Zahtevaj celoten opis,
seznam objektov, položaje, odnose in vidne dokaze o pritrditvi ali podprtju. Opazovanje
naj ostane ločeno od domneve. Florence naloge in izhode predstavi takšne, kot so;
ne izmišljaj enakega opisnega formata ali drugega LLM za dopolnjevanje odziva.

Najprej preveri GPU in po eno sliko na model, te klice nato vključi med redne rezultate,
če imajo enake nastavitve. Beleži model/revision, prompt, parametre, seed, SHA slike,
celoten surovi odziv, parsing status, čas (ločeno nalaganje/warmup in inferenca), tokene,
GPU/VRAM in napake. Uporabi obstoječe izolirane vsebnike/okolja in modele zaporedno,
da ne zmanjka VRAM. Ne spreminjaj uteži ali njihove inicializacije med ponovitvami.

Pri NVILA je bila že težava z obliko odziva. Ohranjen surovi odgovor je rezultat tudi
ob neveljavnem JSON. Ne izvajaj tihega popravljanja z drugim modelom, ne skrivaj napak,
ne ponavljaj klicev do uspeha. Razlikuj neveljavno shemo od vsebinsko uporabnega opisa.
Označene slike riši iz napovedanih okvirjev samo, kadar jih model dejansko vrne.
Kjer jih ni, napiši »model ni vrnil okvirjev«; referenčno označeno sliko lahko dodaš
ločeno z jasno oznako »referenca LOCO«. Florence OD in dense lahko opisujeta isti
objekt večkrat; število izhodnih regij ni nujno število fizičnih objektov.

## 3. Drugi del: LLM na odzivih vseh VLM
Uporabi že nameščene in v prejšnjem drugem delu preverjene modele:
qwen3.5:4b-bf16, qwen3.5:4b-q4_K_M in obstoječo različico Gemma 3 (točno oznako
preberi iz prejšnje konfiguracije; ne ugibaj in ne dodajaj nove kvantizacije).

Za vsak odziv VLM posebej zamrzni besedilni vhod za drugi del: opis + objekti +
položaji/odnosi, brez referenčnih oznak. Florence naloge sestavi v en sledljiv vhod
na sliko/ponovitev; ne trikrat istega sestavljenega vhoda. Ohrani izvorni tekst,
ID vhodnega odziva in SHA. Ne zamenjuj treh VLM ponovitev s tremi LLM ponovitvami.

Na vsakem uporabnem vhodu poženi vse tri LLM, kratek in daljši sistemski prompt,
za vsakega tri ponovitve (seed 42/43/44, temperature 0.3, ostalo enako).
Ponovno uporabi obstoječa kratki/dolgi prompt in ocenjevalni kriterij, če obstajata.
Oba naj bosta brez razdelka primerov; če obstoječa nista, shrani izrecno novo različico
brez primerov. Oba enako definirata fixed kot nameščen/strukturen objekt, movable kot
objekt, ki ga je mogoče prestaviti, uncertain kot premalo dokazov; gibanje vrat pri
odpiranju ni prestavitev celih vrat. Daljši naj zahteva podrobnejše dokaze, ne drugačnih
razredov. Shrani natančen celoten sistemski in uporabniški prompt pri vsakem rezultatu.

Pričakovana mreža je 30 slik × 3 VLM × 3 VLM ponovitve × 3 LLM × 2 prompta ×
3 LLM ponovitve = 4.860 položajev. Če vhod ni uporaben, evidentiraj skipped z razlogom
na vseh pripadajočih položajih; ne pošiljaj praznega ali izmišljenega vhoda. Dejanske
uspehe, napake, preskoke in število API klicev seštej posebej. Neuspelih klicev ne
prikrivaj s ponovnim generiranjem. Nadaljevanje mora preskočiti že zaključene enake
konfiguracije in preprečiti mešanje spremenjenih promptov z obstoječimi rezultati.

## 4. Vrednotenje in primeri
- Primerjaj hitrost, uporabnost in podrobnost opisov, pokritost petih označenih razredov,
  napake in ponovljivost treh odzivov na isti sliki. Veljaven JSON ni merilo kakovosti.
- Za napovedi z okvirji lahko izračunaš ujemanje z referenco pri IoU >= 0.5 in
  enoličnem parjenju znotraj razreda. Preslikavo imen razredov shrani vnaprej;
  prostega opisa brez okvirjev ne ocenjuj kot detektor z navidezno enakim merilom.
- Referenca vsebuje samo pet logističnih razredov in ni popoln seznam scene. Neoznačen
  regal/stena/človek zato ni avtomatično halucinacija. Baza ne vsebuje ground truth za
  fixed/movable/uncertain: poročaj skladnost in razlike, ne izmišljene točnosti.
- Primerjaj LLM po istem zamrznjenem vhodu; vpliv VLM pokaži ločeno. Zapiši konkretne
  spremembe razreda in celotno kratko utemeljitev. Ujemanje ni dokaz pravilnosti.
- Izberi 10 zanimivih slik: različna okolja, dober/slab primer, razlike med modeli in
  ponovitvami. Ne izberi samo uspehov. Dodaj prazen obrazec CSV za moj ročni pregled,
  ročnih ocen ne izpolnjuj v mojem imenu.

## 5. Oddaja in push
Velike delovne izhode shrani v `results/loco/`, ključen samostojen paket pa v:
`analiza rezultatov vmesna predstavitev/industrija LOCO/`

Predlagana razdelitev: `01_vlm`, `02_llm`, `03_primeri`, `04_povzetki`.
Objavi VSE celotne besedilne odzive vseh 30 slik in ponovitev, napake/preskoke,
prompte, parametre, zamrznjene vhode, čase in strojno berljive primerjave, ne samo
10 lepih primerov. Slike lahko paket uporablja iz `data/loco_test30`, kjer so že v Gitu;
za izbranih 10 dodaj napovedane in ločeno referenčne okvirje z legendo.
Pripravi preprost lokalni HTML za pregled vseh odzivov in kratek `Rezultati_LOCO.pdf`
ter njegov urejevalni vir z 10 primeri (slika, model, celoten opis/seznam, primerjava
LLM z utemeljitvami, preproste črno-bele tabele). Uporabi obstoječe izvozno orodje;
če so odgovori dolgi, jih daj v prilogo in v HTML. Ne izpuščaj jih iz objavljenega paketa.
Ne prepisuj celotnega prejšnjega poročila in ne uporabljaj okrasnega oblikovanja.

Preveri število slik/odzivov, SHA povezave, berljivost slik, poti HTML/PDF in odsotnost
referenc v vhodih. Po spremembah zaženi ustrezne obstoječe teste in manjši preverjalnik.
Dodaj kratek README z ukazi za ponovitev, omejitvami in dejanskim obsegom izvedbe.
Commit/push na trenutno vejo: koda, konfiguracije, izbrani podatki in vsi ključni
rezultati. Brez uteži, virtualnih okolij, skrivnosti ali celotnega arhiva. Na koncu
izpiši commit SHA, število uspehov/napak/preskokov in poti do HTML/PDF. Če je del
blokiran, objavi tudi dosežene rezultate in natančen razlog; ne označi ga kot izvedenega.
