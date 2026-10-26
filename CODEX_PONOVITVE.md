# Codex: slikovni modeli, tri ponovitve in paket desetih primerov

To je aktualna naloga in ima prednost pred starim načrtom predstavitve v
`CODEX_NALOGA.md`. Izvedi oba dela zaporedno na laboratorijskem Linuxu z RTX 5090.
Uporabi pripravljeno kodo in lokalne modele; rezultat mora biti izveden poskus
in objavljen paket podatkov, iz katerega lahko pozneje napišemo poročilo.

## Najprej uporabi, kar že deluje

Izhodišče je veja `feature/separate-vision-stages`, pregledana do `3ec5355`.
Ta commit že objavlja zaključeno primerjavo NVILA in Florence. NVILA je izvedel
100 GPU odgovorov, a 0/100 po strogi rich-v2 shemi; to ni manjkajoča namestitev.
Florence ima caption 100/100, OD 100/100, dense 99/100. Ne ponavljaj namestitev
ali starih poskusov zgolj zato, da bi znova preverjal že potrjeno delovanje.

Preberi ta dokument, morebitni AGENTS.md ter samo potrebne dele spodnjih datotek:

| Uporabi obstoječe | Namen |
| --- | --- |
| `docs/THUD_EXTENDED.md`, `scripts/thud_extended.sh` | pripravljena okolja in zagon |
| `thud_florence.py`, `thud_nvila.py`, `thud_extended_common.py` | native GPU, naloge, okvirji, surovi odgovori |
| `stage1_describe.py`, `vision_backends.py`, `description_prompts.py` | slikovni Qwen/Gemma |
| `stage2_reason.py`, `thud_extended_reason.py` | razvrščanje samo iz shranjenega besedila |
| `thud_box_metrics.py`, `configs/thud-florence-matching.json` | že določen slovar oznak in ujemanje regij |
| `thud_extended_report.py`, `thud_part2_report.py` | pregled slik in obdelava meritev |
| `reports/THUD_VMESNO/data-audit.json` | isti predhodno določeni izbor 10 + 10 slik |

Imena modulov v tabeli so v `llm_object_classifier/`. Dodaj le manjkajoče
argumente za izbor, izhodno mapo, ponovitev in parametre ter tanek zaganjalnik.
Ne kopiraj celih izvajalnikov in ne preoblikuj celotnega projekta.

## Skupni protokol

- Uporabi isti že določeni podnabor **20 slik: 10 trgovine in 10 agtea** iz
  obeh vključenih izborov po 50. Shrani enoten manifest s SHA256, izvorom in
  potmi. Lokalni ID slike/objekta ni skupna identiteta med različnimi izhodi.
- Najprej preveri po eno sliko iz vsakega nabora, nato dokončaj 20 slik.
  Kratki pilot lahko šteje v ponovitev 1 le ob povsem enakem protokolu.
- Vsak model in vsaka naloga morata imeti **tri dejanske generacije** na isti
  sliki: `r1`, `r2`, `r3`. Shranjenega odgovora ne kopiraj kot novo ponovitev.
  Nadaljevanje izvede samo manjkajoče klice; napake ostanejo zabeležene.
- Pri vzorčenju uporabi T=0.3 ter semena 42/43/44 za r1/r2/r3; vsi drugi
  parametri in prompt so znotraj modela enaki. To meri variabilnost vzorčenja
  med semeni, ne ponovljivosti pri fiksnem semenu. Florence ohrani svoj
  delujoči deterministični režim (beam=3, do_sample=false) in tudi zanj izvedi
  tri sveže generacije. Enaki odgovori so veljaven izid. Nepodprte parametre
  izrecno označi; ne spreminjaj dekodiranja modela samo za umetne razlike.
- Pred zagonom shrani konfiguracije in število pričakovanih klicev. Modeli
  tečejo zaporedno, na GPU. Loči nalaganje, fizično ogrevanje, inferenco in
  pripravo prikazov. Beleži čas na sliko/nalogo, tokene, VRAM in napake.
- Obstoječi izhod je lahko r1 le ob preverjeno enakih slikovnih bajtih, promptu,
  nastavitvah in reviziji modela. Poveži izvirnik in zabeleži ponovno uporabo.
  Stari časi z drugačnimi pogoji niso nove primerljive meritve.

## 1. del: slika -> podroben opis in seznam objektov

Obvezno primerjaj **Florence-2-large**, **NVILA-8B** in **Qwen3.5 4B BF16**.
Ker sta že pripravljena, vključi še Qwen3.5 4B Q4 in Gemma 3 4B Q4. To je
primerjava slikovnih modelov/VLM; Qwen/Gemma s slikovnim vhodom sta prav tako
VLM. Besedilnega LLM brez slikovnega vhoda ne predstavljaj kot opisovalnika.

Model dobi samo fotografijo in generični vhod (**discovery**), brez referenčnih
imen, ID-jev, okvirjev ali anotacij. Reference so samo za poznejše vrednotenje
in posebej označen referenčni izris. Načina supplied ne mešaj v detekcijo.

Želimo celoten opis scene, seznam vidnih objektov/posameznih instanc, njihov
videz, položaj, zakritost in vidne odnose oziroma stike, kjer jih model zna
podati. Negotovost naj ostane vidna. Ne zahtevaj presoje stabilnosti v tem delu.

- Florence: uporabi `<MORE_DETAILED_CAPTION>`, `<OD>` in
  `<DENSE_REGION_CAPTION>`; vse tri naloge trikrat. Ohrani njegove native izhode,
  meri vsako nalogo in celoten postopek. Ne vsiljuj mu Qwenove JSON sheme.
- NVILA/Qwen/Gemma: primerljivo vsebinsko navodilo brez primerov objektov;
  uporabi najpreprostejšo delujočo strukturo opisa + objektov. NVILA ne zavrzi
  samo zato, ker ne izpolni rich-v2. Loči surovi tekst, sintaktično razčlenitev,
  skladnost s shemo in vsebinsko presojo. Če spremeniš prompt, novo različico
  shrani in je ne primerjaj kot izolirano spremembo samo modela ali semena.
- Shrani **celoten surovi odgovor**, celotni sistemski/uporabniški prompt in
  razčlenjena polja, tudi pri zavrnitvah. Brez skrajševanja s tremi pikami in
  brez dopolnjevanja izmišljene vsebine pri pretvorbi v skupni prikaz.
- Florence že vrača napovedane okvirje: nariši ID in oznako na sliko, OD in
  dense ločeno. Za model brez lastnih okvirjev pokaži original + oštevilčen
  seznam; ne riši referenčnih okvirjev kot njegovih napovedi. Za dodatno
  lokalizacijo njegovih poimenovanj lahko s tanko razširitvijo uporabiš
  Florence `<CAPTION_TO_PHRASE_GROUNDING>`, vendar tak izris jasno označi kot
  **modelov opis + Florence lokalizacija**, meri ga posebej in ne pripisuj
  okvirjev prvemu modelu. Nepovezane objekte označi kot nelokalizirane.

Za tri ponovitve primerjaj število objektov, dodana/izpuščena imena, spremembe
lastnosti/lokacij, odnose in enakost odgovorov. Izračunaj parno prekrivanje
normaliziranih imen; sinonime in skupine označi, ne enači jih samodejno.
Okvirje primerjaj po dokumentiranem enoličnem ujemanju in IoU, kjer obstajajo.
Strinjanje in veljaven JSON nista pravilnost. Za reference uporabi obstoječo
metodologijo; brez ustreznih ocen ne poročaj mAP ali accuracy stabilnosti.

## 2. del: isti shranjeni opis -> besedilni LLM

Najprej zamrzni opise in objekte Qwen BF16 iz r1 za istih 20 slik, s SHA256
vsakega celotnega vhoda. Če posamezen opis ni uporaben, zabeleži preskok;
ne zamenjaj ga tiho z drugim modelom ali drugo ponovitvijo.

Primerjaj Qwen3.5 4B BF16 in Q4 kot besedilna razvrščevalnika. Vsak prejme
**isti zamrznjeni vhod**, brez slike in referenc stabilnosti. Uporabi obstoječa
kratki in dolgi prompt brez primerov (`thud-short-no-examples.txt`,
`thud-long-no-examples.txt` v `configs/prompts/`), z enakimi definicijami in
izhodno shemo. Vsak prompt izvedi trikrat: T=0.3, semena 42/43/44.
Osnovni obseg je 20 slik x 2 LLM x 2 prompta x 3 = 240 položajev.

Vrni fixed/movable/uncertain in celotno kratko besedno utemeljitev za vsak
podani objekt. Primerjaj razrede po istem tipiziranem ID na istem SHA256 vhoda:
spremembe med tremi ponovitvami, med kratkim/dolgim promptom in med modeloma.
Poročaj veljavne primerljive pare in vse napake/preskoke. Celoten cilj je
preveriti vpliv prompta/LLM ločeno od variabilnosti slikovnega opisa.

## Pregledna struktura in deset zanimivih primerov za Git

Nova koda naj ostane v obstoječih modulih. Nov zaganjalnik in kratek README
naj pokažeta zaporedje: izbor -> VLM x3 -> zamrznjeni opisi -> LLM x3 -> izvoz.
Novi veliki surovi izhodi: `results/ponovitve/`; ničesar starega ne prepisuj.
Konfiguracije/prompti: `configs/ponovitve/`.

V `reports/PRIMERJAVA_PONOVITEV/` objavi:

- `README.md`: kratek študentski opis, točni ukazi, dejansko izveden obseg,
  omejitve in navodilo, kako odpreti primere. Brez dolgih dnevnikov v povzetku.
- `manifest.json`, `summary.csv/json`, `consistency.csv/json`: izbor vseh 20,
  nastavitve, model/revizija, SHA256, hitrost, veljavnost in primerjave ponovitev.
- `prompts/`: celotni dejansko poslani prompti obeh korakov in native naloge.
- `cases/01-...` do `10-...`: izvirna slika (ali lokalna povezava na sliko v
  Gitu), označene slike po modelu/nalogi/ponovitvi, celoten opis scene, celoten
  seznam objektov, vse tri surove odgovore in razčlenjene rezultate vseh modelov,
  kratki/dolgi LLM prompt ter vse tri klasifikacije z razlogi in časi.
- `index.html`: enostaven pregled brez strežnika, fotografija poleg rezultatov,
  jasni zavihki/stolpci model -> r1/r2/r3; opisi naj bodo v celoti dostopni.
  Napovedi, reference in dodatno Florence lokalizacijo vidno razlikuj.

Deset primerov izberi iz teh 20, po možnosti 5 + 5: vključuj razhajanja,
zakritost/gost prizor, napačen opis, napako sheme ter vsaj dva stabilna uspešna
primera. Za vsakega zapiši razlog izbire in razliko med modelovo trditvijo in
lastno vizualno presojo. Ne izbiraj samo lepih odgovorov; skupne številke ostanejo
za vseh 20 slik. Enak ID objekta med modeli ne pomeni istega fizičnega predmeta.

## Varčna izvedba s Codexom in zaključek

Analizo JSON, štetja, izbor kandidatov, izrise in HTML izvedi s skriptami,
ne s prebiranjem vseh surovih odgovorov v kontekst Codexa. V terminal izpisuj
napredek, povzetke in kratke napake, ne base64 slik, vseh tokenov ali stotine
opisov. Ročno podrobno poglej končnih deset primerov. Ne zaganjaj starih celotnih
benchmarkov; za spremembe preveri samo prizadete teste in kratek GPU pilot.
Ohrani že nameščene modele, okolja in uspešne rezultate. GPU lahko računa dlje;
ne ustavljaj dela pri načrtu ali zaradi neobvezne optimizacije.

Pred zaključkom preveri, da ima vsaka zahtevana kombinacija 3 neodvisne
rezultate ali izrecno napako/preskok. Ponovitve ne smejo ostati 3x isti cache.
Preveri, da je vseh 10 primerov berljivih iz Gita in ne uporablja absolutnih
laboratorijskih poti. Ohraniti moraš izvorne surove odgovore; v Git izvozi
kompaktne kopije brez slikovnih base64, skrivnosti, uteži in vendorskih knjižnic.
Če takšno polje odstraniš, to zabeleži; besedila odgovora in prompta ne krajšaj.

Naredi commit in push na delovno vejo: koda, konfiguracije, prompti, agregati in
celoten paket desetih primerov. Za vseh 20 objavi tudi besedilne odgovore in
odločitve brez podvojenih slikovnih bajtov, če ostanejo razumne velikosti; sicer
shrani stisnjen arhiv kot GitHub Release prilogo in v README navedi preverjeno
povezavo ter SHA256. Lokalni podatki sami niso dovolj za poznejši oddaljeni pregled.
Ob oviri pri objavi navedi konkretno napako in kaj ostaja samo lokalno.
Končni odgovor naj poda commit, povezavo na pregled desetih primerov, dejansko
število klicev in največ tri ključne ugotovitve. Končno poročilo bomo sestavili
iz teh objavljenih podatkov; zdaj ni treba znova generirati dolgega PDF.
