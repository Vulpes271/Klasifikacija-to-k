# 50 slik za teste: THUD, agtea

Ta prvotni izbor je nadomeščen z [gostejšim trgovinskim naborom](../thud_dense_test50/README.md).
Za teste, pri katerih je ključno malo praznega prostora, uporabi novi nabor.

Iz ene serije `data/THUD_Robot/Real_Scenes/agtea/dynamic/Capture_1` je izbranih
50 izvirnih RGB slik velikosti 960 × 540. Izvorne datoteke niso spremenjene.

| Lastnost | Vrednost |
| --- | ---: |
| Pregledanih slik v izbrani seriji | 365 |
| Kandidatov po filtriranju | 241 |
| Izbranih slik | 50 |
| Označenih objektov skupaj | 1021 |
| Objektov na sliko | 14–37 |
| Povprečno objektov na sliko | 20,42 |
| Različnih vrst na posamezni sliki | 6–11 |
| Različnih vrst v celotnem izboru | 21 |

Za vizualni pregled odpri [galerijo](gallery.html) ali pregledni sliki
[1–25](contact_sheet_01.jpg) in [26–50](contact_sheet_02.jpg).
V galeriji klik na sličico odpre izvirno sliko.

## Zakaj ta serija

Pri začetnem pregledu je bilo vzorčenih do 12 enakomerno razporejenih oznak
iz vsake od 16 resničnih serij. Primerjava je v `series_comparison.csv`;
to je vzorčna primerjava, ne popolno štetje objektov v vseh serijah.
Serija agtea združuje številne objekte, raznolike vrste predmetov in različne
poglede istega prostora. Trgovina ima veliko vizualnih elementov, vendar so
številni izdelki označeni skupinsko; izbrana serija je primernejša za primerjavo
posameznih predmetov, kot so stoli, mize, police, torbe in prenosniki.

## Merila izbora

- Najmanj 14 dovolj velikih označenih objektov in 6 različnih vrst na sliko.
  Uporaben okvir pokriva najmanj 0,1 % slike in ima obe stranici dolgi najmanj
  10 pik. Izbrane slike ohranijo vse izvorne oznake.
- Izločena je desetina slik z najnižjo varianco robov. To je relativna ocena
  ostrine znotraj te serije, ne zagotovilo, da je vsaka slika povsem ostra.
- Med izbranima številkama sličic je razlika najmanj 4. To ni čas v sekundah.
- Izbor združuje število objektov, raznolikost vrst, ostrino in vizualno
  razdaljo, izračunano iz pomanjšanih barvnih slik in razlikovnih zgoščenk.
- Vseh 50 slik ima različne SHA-256 zgoščene vrednosti. Povezani pogledi istega
  prostora ostajajo; to niso neodvisni prostori ali statistično neodvisni vzorci.

Obe pregledni sliki sta bili vizualno pregledani. Nekateri posnetki vsebujejo
bleščanje oken, neenakomerno osvetlitev, delno zakrite predmete ali zameglitev
zaradi premikanja. Primerni so tudi za preverjanje obnašanja v zahtevnejših
razmerah; izbor ni omejen na idealno osvetljene prizore.

## Datoteke

- `images/`: 50 izvirnih PNG slik; brez spreminjanja ali ponovnega kodiranja.
- `annotations/`: izvorne pravokotne oznake LabelMe. Polje `imageData` je
  nastavljeno na `null`, ker je slika že shranjena ločeno;
  `imagePath` kaže na kopijo v `images/`. Vsi `shapes` ostanejo enaki izvoru.
- `objects.jsonl`: 50 zapisov v obliki, ki jo bere projektni klasifikator,
  z ločenimi objekti iste vrste, okvirji `bbox_xyxy`, velikostjo slike in izvorom.
  Poti `image_path` in `annotation_path` so relativne glede na ta imenik.
- `selection.csv`: meritve za vseh 365 slik in stolpec `selected`.
- `summary.json`: statistika, pragovi in zastopanost posameznih vrst.
- `SHA256SUMS.txt`: kontrolne vrednosti 50 izbranih RGB datotek.
- `gallery.html`, `preview/`, `contact_sheet_*.jpg`: vizualni pregled izbora.

ID objekta je zaporedna številka izvorne oznake znotraj slike, ne identiteta
predmeta skozi čas. Število objektov pomeni število izvornih pravokotnih oznak;
ne gre za novo detekcijo ali ročno popravljeno anotacijo. Isti fizični predmet
je lahko označen na več slikah.

Novi ločeni prvi korak (`stage1_describe`) neposredno podpira te okvirje
`bbox_xyxy`, lokalne ID-je in oznake. Za zagon glej [data/README.md](../README.md).
Starejši združeni `vision_pipeline` je še vedno namenjen predvsem maskam NYUv2.
Referenčnih razredov `fixed/movable/uncertain` ni; za accuracy so potrebne ročne reference.

Referenčni razredi `fixed/movable/uncertain` niso podani. Pred merjenjem
pravilnosti jih je treba ročno označiti; obstoječe oznake podajajo vrsto objekta.
Ker so slike iz istega zajema, naj celoten zajem pri delitvi za učenje in test
ostane na isti strani, da podobni pogledi ne prehajajo med obema množicama.

## Ponovitev izbora

Iz korena projekta, s Pillow in Python 3.9 ali novejšim:

```powershell
python scripts/select_thud_test_images.py --mode diverse --source data/THUD_Robot/Real_Scenes/agtea/dynamic/Capture_1 --output data/thud_agtea_test50_repeat
```

Skripta zavrne obstoječo izhodno mapo. Zgornji ukaz uporabi isto serijo in izbere
50 slik. Za ponovljive primerjave modelov uporabi že pripravljen `objects.jsonl`
in enake slike, ne novega naključnega izbora pri vsakem testu.

Preverjeno: točno 50 berljivih različnih slik, ujemanje kontrolnih vrednosti
z izvirniki, ujemanje vseh izvornih oznak, veljavnost projektne sheme, vseh
1021 okvirjev znotraj slik ter upoštevanje minimalnega števila objektov,
vrst in razmika sličic. Modelski testi še niso bili izvedeni.
