# Primerjava samo prvega koraka

Odprite [index.html](index.html) v lokalnem brskalniku. V GitHubovem prikazu HTML
prenesite repozitorij in datoteko odprite lokalno. Vse slike in povezave so lokalne;
ni zunanjih skript ali storitev. Posamezna slika ima svojo stran s tremi stolpci
modelov in vsemi tremi ponovitvami.

Paket obsega isti predhodno določeni izbor **20 slik: 10 trgovine in 10 agtea**,
tri modele in tri ponovitve: **180 položajev**. Florence ima v vsakem položaju
tri ločene naloge, zato je skupaj **300 izvornih odgovorov**: 180 Florence,
60 NVILA in 60 Qwen3-VL. Slike so povezani pogledi istih prostorov, ne 20
neodvisnih prizorov. Noben izvajalnik ni dobil referenčnih imen, okvirjev ali
objektnih anotacij. Drugi korak in izdelava PDF v tej nalogi nista izvedena.

Zaključeno: Florence ima 180/180 uspešno razčlenjenih native odgovorov, Qwen3-VL
60/60 po uporabljeni shemi, NVILA pa 0/60 po strogi shemi celotnega odgovora.
Pri NVILA je iz 52/60 odgovorov mogoče dobesedno izluščiti neprazen seznam
objektov za prikaz. To so podatki o zapisu, **ne meritve zaznavne pravilnosti**.
Florence ima enake odgovore v vseh 180 parih ponovitev; NVILA in Qwen imata
različne surove odgovore v vseh svojih 60 parih.

## Kaj je bilo dejansko izvedeno in kaj ponovno uporabljeno

- **Florence-2-large:** ponovno uporabljene tri sveže GPU ponovitve iz poskusa
  26. 10. 2026. Vsaka ima lastni izvorni `call_id`, čas in zapis. Enaki odgovori
  ne pomenijo kopiranih generacij.
- **NVILA-8B:** ponovno uporabljenih 60 izvornih GPU odgovorov iz istega poskusa,
  vključno s celotnimi neveljavnimi JSON in napakami. Uporabljena je popravljena
  serija, kjer je inicializacijsko seme vseh nalaganj 42, vzorčenje pa 42/43/44.
  Nepravilno inicializirani diagnostični poskusi niso uvoženi. R2 in r3 imata
  ujemajoča se izmerjena prstna odtisa dodanih embedding vrstic; r1 je bil naložen
  s semenom 42, vendar takrat prstni odtis ni bil zajet. Dokazi in omejitev so
  v `nvila-initialization-audit.json` in `provenance/`.
- **Qwen3-VL-8B-Instruct:** 60 novih GPU generacij, z enakim vsebinskim navodilom
  kot NVILA. To je točen zahtevani 8B model, ne prej uporabljeni Qwen3.5 4B.
  Uteži na tem računalniku še niso bile prisotne; prenesena je pripeta revizija.
  Ogrevanje in prvi neuspešni API preizkus sta shranjena ločeno in ne štejeta
  med 60 odgovorov. Prvi dve merjeni sliki, po ena iz vsakega nabora, štejeta v r1.

`provenance.json` povezuje vsak objavljeni odgovor z izvirno lokalno datoteko,
njenim SHA256, izvirnim identifikatorjem klica in časom. Datoteke `*.raw.json`
so **bajtno enake izvirnikom**, ne nove generacije. Njihove stare absolutne
poti so del izvornega zapisa; prenosljive poti so v `manifest.json` in HTML.
Objava jih ne uporablja za odpiranje slik. Izvirni lokalni rezultati se ohranijo.

## Protokol in pomembne razlike

| Nastavitev | Florence-2-large | NVILA-8B | Qwen3-VL-8B-Instruct |
|---|---|---|---|
| Revizija HF | `21a599d414c4d928c9032694c424fb94458e3594` | `049743cc51bdd8872cdf696a1a58b03fef2c367e` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Dejanska natančnost uteži | FP16 | FP16 | BF16 |
| Dekodiranje | deterministično, beam 3 | vzorčenje, beam 1 | vzorčenje, beam 1 |
| T / semena r1,r2,r3 | T ni aktiven / 42,43,44 | 0.3 / 42,43,44 | 0.3 / 42,43,44 |
| top-p / top-k / repetition penalty | neaktivno | 0.8 / 20 / 1.05 | 0.8 / 20 / 1.05 |
| Izhodni limit | 1024 tokenov na nalogo | 4096 tokenov | 4096 tokenov |
| Kontekst | lasten encoder-decoder | 16384 | vhod + izhod največ 16384 |
| Attention | eager | SDPA | SDPA |
| Naloge | caption + OD + dense, ločeno | opis + objekti z lokacijami in odnosi | isto vsebinsko navodilo kot NVILA |
| Lastni okvirji | OD in dense, xyxy v originalnih pikah | v tem protokolu jih ni | v tem protokolu jih ni |
| Predobdelava | native 768 × 768 | native dynamic-S2 ploščice | privzeti processor; dejanski `image_grid_thw` v odgovorih |
| Okolje | obstoječi thud-florence | obstoječi thud-nvila | tanka nadgradnja thud-florence za API Qwen3-VL |

Vsa okolja uporabljajo RTX 5090 in PyTorch 2.7.1 + CUDA 12.8; zahtevana je
prisotnost vseh parametrov na GPU. Qwen3-VL uporablja Transformers 4.57.6,
stara native izvajalnika 4.46.0. Celotni podatki o verzijah, dejanskih dtype,
napravah, številu parametrov in nalaganju so v `provenance/`.

Florence dobi samo native naloge `<MORE_DETAILED_CAPTION>`, `<OD>` in
`<DENSE_REGION_CAPTION>`, brez vsiljene JSON sheme za opisovalnika. Njegove
slike prikazujejo **njegove napovedi**, OD in dense posebej. ID je lokalno
oštevilčenje rezultatov posamezne naloge, ne identiteta objekta med modeli.
Referenčnih okvirjev paket ne prikazuje. NVILA in Qwen prikazujeta original
ter svoj celoten seznam, brez pripisovanja Florencejevih ali referenčnih okvirjev.

NVILA in Qwen uporabljata isto shranjeno navodilo `nvila-description-v2.txt`.
To navodilo že vsebuje omejitev približno 120 besed za sceno in 60 za posamezno
opazovanje. **Pri izvozu se odgovorov ne krajša**; ohranjen je tudi vsak odgovor,
ki te meje prekorači. Native sistemsko sporočilo je »You are a helpful assistant«,
vsebinsko navodilo in generični vhod pa sta uporabniško sporočilo s sliko.
NVILA-jevi starejši komponenti `system_prompt`/`user_prompt` ostaneta v surovem
zapisu; dejanska native razporeditev je pojasnjena v `*.result.json` pod `prompts`.
Florencejev razširjeni procesorski prompt je izpeljan iz pripete kode procesorja
in označen kot rekonstrukcija. Qwenov native chat prompt je zajet med izvajanjem.
`presence_penalty` se ne uporablja; native `repetition_penalty` ni njegov ekvivalent.
Znotraj vsakega modela so druge nastavitve in prompt enaki v vseh ponovitvah.

Razlike med modeli vključujejo natančnost uteži, naloge, slikovno predobdelavo,
programsko okolje in čas merjenja. Rezultat zato ni izolirana primerjava arhitektur.
GPU je deljen; procesov drugih uporabnikov nismo ustavljali. Stari časi Florence
in NVILA ter novi Qwenovi časi so opazovalne meritve iz različnih terminov.

## Datoteke in interpretacija

Vsaka mapa `slika/model/r1|r2|r3/` vsebuje:

- `caption|od|dense|description.raw.json`: celoten izvorni odgovor, tokeni,
  čas, VRAM in morebitna napaka, vključno z izvorno sledjo napake;
- `*.result.json`: celoten opis in seznam objektov v `view`, dejanski prompti
  v `prompts`, parametri, revizija, izvor ter meritve v `metadata`. Izluščenje
  je dobesedno in ne popravlja trditev; celotna okolja so v `provenance/`;
- samo pri Florence: `od.predictions.jpg`, `dense.predictions.jpg` in
  `combined.raw.json` z ločeno meritvijo celotnega postopka treh nalog.

`summary.csv/json` primerja čas in pomnilnik po modelu/nalogi.
`response-metrics.csv` vsebuje vse posamezne meritve. Čas posameznega odgovora
je CUDA-sinhroniziran čas od predobdelave do razčlenitve; ni zgolj čas dekodiranja.
`tokens_per_end_to_end_second` zato vključuje tudi predobdelavo in parser.
Florencejev skupni čas je izmerjen neposredno in vključuje nalaganje slike,
tri naloge in zapis rezultatov, izključuje pa risanje okvirjev. Ločena pomanjkanja
ali napake niso nadomeščena z ocenjenimi uspešnimi rezultati. VRAM allocated in
reserved sta vrhova PyTorchovega procesa; ne pomenita celotne zasedenosti GPU.
Ogrevanje, nalaganje uteži in priprava pregledovalnika so ločeni od inferenc.

`repeat-differences.csv/json` vsebuje vse tri pare ponovitev za vsako sliko in
nalogo: enakost celotnega surovega odgovora in scene, število zapisov objektov,
dobesedno dodana/izpuščena normalizirana imena ter spremembe vseh lastnosti
znotraj skupin istih imen. Normalizacija imen je samo mala začetnica in presledki;
sinonimov se ne združuje. Namigi o možnih sinonimih niso ujemanje objektov.
Identična imena ne zagotavljajo iste fizične instance, množinski zapis pa ni
štetje vseh instanc. Spremembe lokacij in odnosov so del celotnih `observation`
polj; brez izmišljene semantične razčlenitve. Če ni strukturiranih odnosov,
`relations_equal` ostane nedoločen.

Pri Florence OD/dense se okvirji ujemajo **med ponovitvama**, po istem dobesednem
normaliziranem imenu, ena-proti-ena, pri IoU ≥ 0.5. Uporabljen je obstoječi
algoritem največjega števila ujemanj, ne nujno največje vsote IoU. To je meritev
doslednosti napovedi, **ne zaznavna natančnost proti referencam**, tudi če se
notranja polja pomočnika imenujejo precision/recall. `box_matching` za druga dva
modela ne obstaja. Pri nedostopnem objektu zaradi parserja je Jaccard prazen,
ne pa navidezna meritev prazne scene. Caption ne predstavlja detektorjevega
seznama objektov. Različni nalogi OD in dense se ne seštevata v število instanc.

NVILA pogosto vrne več zaporednih JSON dokumentov ali nepravilno narekovano
besedilo. Stroga veljavnost celotnega odgovora ostane neuspešna. Prikaz lahko
izlušči samo cele, strogo berljive zaporedne JSON dokumente in to označi kot
nevalidiran prikaz. Preostanek besedila je vedno v surovem odgovoru. Kadar
izluščenje ni mogoče, `available=false` pomeni nedostopno razčlenitev seznama,
ne dokaz nič zaznanih objektov. Tudi strukturno veljaven odgovor ni dokaz
pravilnega zaznavanja ali pravilnih odnosov. Ta paket ne podaja samodejne
»accuracy«, ker opisom in odnosom nimamo potrjenih ustreznih referenc.

`manual-review-empty.csv` je **prazen obrazec za vaš pregled**, po ena vrstica
na sliko/model/ponovitev. Imena in povezave so izpolnjena, vse presoje so prazne.
`verification.json` zabeleži preverjanje vseh 180 položajev in 300 odgovorov,
enakost izvirnih bajtov in celotnih izvozov ter delovanje lokalnih povezav.
HTML je statičen; avtomatsko preverjanje povezav ne pomeni vizualnega pregleda
v brskalniku. Modelov tekst se v HTML vedno ubeži, nikoli izvrši kot HTML.

## Ponovitev na obstoječem laboratorijskem okolju

Iz korena repozitorija:

```bash
# Samo če manjka novo Qwenovo okolje/uteži; ohrani obstoječi PyTorch/CUDA.
bash scripts/thud_stage1_presentation.sh prepare
# Samo manjkajoči Qwen klici; že shranjenih odgovorov ne generira ponovno.
bash scripts/thud_stage1_presentation.sh qwen
# Samo skriptna obdelava; brez GPU, drugega koraka in PDF.
bash scripts/thud_stage1_presentation.sh export
python3 -m unittest tests.test_thud_stage1_presentation tests.test_thud_repetitions tests.test_thud_extended
```

Izvoz zahteva ohranjene izvirnike v `results/ponovitve/stage1/{florence,nvila}`
in `results/stage1-presentation/qwen3-vl`. Neposredna native izvajalnika za
Florence/NVILA ostaneta nespremenjena. Za nov, neodvisen poskus izberite novo
lokalno izhodno mapo in ga jasno poimenujte; izvirnikov ne brišite za prisiljeno
ponavljanje. Tu se uvažajo samo protokolarno preverjene prejšnje generacije.
Uteži, HF cache in Docker okolja ostanejo lokalno in niso del Git objave.

Viri za API in modele:
[Qwen model in uradni način nalaganja](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Florence](https://huggingface.co/microsoft/Florence-2-large),
[NVILA](https://huggingface.co/Efficient-Large-Model/NVILA-8B),
[pripeta VILA koda](https://github.com/NVlabs/VILA/tree/57ecbfb904b99b563db69ef7916f6390eecea516).
