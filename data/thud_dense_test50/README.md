# 50 slik s polnimi prizori za teste

Aktualni izbor po zahtevi, da naj bo kader poln objektov in naj vsebuje malo
praznega prostora. Ta nabor nadomešča prvotni izbor `thud_agtea_test50`.

Vseh 50 slik prihaja iz ene resnične serije:
`THUD_Robot/Real_Scenes/store/dynamic/Capture_1`.
Na slikah prevladujejo napolnjene trgovinske police, izdelki, hladilniki,
steklenice, košare in druga oprema. Slike niso izrezane ali drugače spremenjene.

Odpri [galerijo](gallery.html) ali pregledni sliki [1–25](contact_sheet_01.jpg)
in [26–50](contact_sheet_02.jpg). Klik na sličico v galeriji odpre izvirni PNG.

## Kako je bil izbor popravljen

Glavno merilo je zapolnjenost kadra s predmeti. Za predizbor je uporabljen delež
slike, ki ga pokriva unija pravokotnih oznak predmetov. Prekrivajoče oznake se
štejejo samo enkrat. Okna, vrata, stene, tla, strop, stopnice in dvigala so iz
te meritve izključeni, da velike prazne površine ne zvišujejo ocene.

Vsaka izbrana slika ima vsaj 75 % pokritost z okvirji, povprečje je 90,0 %.
**To je približek iz pravokotnih oznak, ne segmentacija in ne natančna meritev
praznega prostora.** Okvirji lahko vsebujejo tudi ozadje in prostor med predmeti.
Obe pregledni sliki z vseh 50 posnetki sta bili dodatno vizualno pregledani.
Pogledi na prazne prehode, vhod ali večje prazne površine niso prednostni.
Nekaj stropa, tal in odprtin ostaja; gre za izvirne poglede iz zajema.

Pomožna merila so ostrina, število oznak, raznolikost pogledov in najmanjši
razmik 3 med številkama sličic. Izločena je najmanj ostra desetina serije po
relativni varianci robov. Nekaj gibalne zameglitve in bleščanja ostaja.

| Lastnost | Vrednost |
| --- | ---: |
| Pregledanih slik iz izbrane serije | 360 |
| Kandidatov po filtriranju | 140 |
| Izbranih slik | 50 |
| Ločenih pravokotnih oznak skupaj | 678 |
| Oznak na sliko | 8–25 |
| Različnih imen oznak skupaj | 21 |
| Velikost slik | 960 × 540 |

Izdelki so pogosto označeni skupinsko kot `Products`, zato 8 oznak ne pomeni
samo 8 vidnih predmetov. Police vsebujejo veliko posameznih izdelkov, ki nimajo
vsak svojega izvornega okvirja. Število oznak zato ni glavno merilo tega izbora.

## Uporaba

- `images/`: 50 nespremenjenih izvirnih PNG slik.
- `objects.jsonl`: isti nabor z izvirnimi imeni oznak in okvirji `bbox_xyxy`.
- `annotations/`: izvorne oznake LabelMe, z nespremenjenimi `shapes`;
  `imageData` je nastavljen na `null`, `imagePath` kaže na kopijo v `images/`.
- `selection.csv`: meritve in odločitev za vseh 360 slik iz izbrane serije.
- `summary.json`: natančni pragovi in statistika izbora.
- `SHA256SUMS.txt`: kontrolne vrednosti izbranih slik.
- `series_coverage_comparison.json`: vzorčna primerjava 16 resničnih serij,
  do 20 enakomerno razporejenih oznak na serijo; ne popolna analiza vseh slik.

Poti do slik v JSONL so relativne glede na ta imenik. ID objekta pomeni indeks
oznake v posamezni sliki, ne sledenja istemu predmetu skozi čas. Vseh 50 slik
ima različne kontrolne vrednosti, vendar ostajajo povezani pogledi iste trgovine.
Nabor je namenjen skupni primerjavi testov, ne delitvi sosednjih pogledov med
učenje in testiranje.

Novi ločeni prvi korak (`stage1_describe`) neposredno podpira te okvirje
`bbox_xyxy`, lokalne ID-je in oznake. Za zagon glej [data/README.md](../README.md).
Starejši združeni `vision_pipeline` je še vedno namenjen predvsem maskam NYUv2.
Referenčnih razredov `fixed/movable/uncertain` ni; za accuracy so potrebne ročne reference.

Ponovitev iz korena projekta v novo mapo:

```powershell
python scripts/select_thud_test_images.py --output data/thud_dense_test50_repeat
```

Preverjeno: točno 50 različnih berljivih slik, enakih izvirnikom, ujemanje vseh
izvornih oznak, veljavnost projektne sheme in okvirjev ter upoštevanje pragov
pokritosti in razmika. Modelski testi še niso bili zagnani.
