# Dnevnik tega paketa — 27. 10. 2026

- Prebrana aktualna zahteva ter predhodna naloga, dnevnik in uporabljeni deli
  izvajalnikov. Ohranjena veja `feature/separate-vision-stages`, izhodišče
  `f2eb213243eded670393ed973c9313c718d57a83`; uporabnikova datoteka
  `vision-launch.log` se ne spreminja in ne vključuje v objavo.
- Preverjenih vseh 20 slik z izvirnimi SHA256 in izborom 10 + 10.
- Protokol Florence/NVILA preverjen po vseh šestih manifestih in posameznih
  zapisih: revizije, slike, semena, dekodiranje, prompti in izvorni klici.
  Uporabljena popravljena NVILA inicializacija 42; neustrezni diagnostični
  poskusi izključeni. Ponovljenih starih inferenc ni.
- Točnega Qwen3-VL-8B-Instruct ni bilo med lokalnimi modeli. Dodan samo novi
  native izvajalnik z obstoječimi THUD pomožnimi funkcijami. Prenesenih 16
  pripetih HF datotek, od tega 17.534.339.512 bajtov uteži. Uteži ostanejo lokalno.
- Obstoječa Docker osnova PyTorch 2.7.1/CUDA 12.8 uporabljena s tanko nadgradnjo
  na Transformers 4.57.6 in Accelerate 1.12.0. Vseh 750 parametrskih tenzorjev
  Qwen3-VL na CUDA:0, dejanski BF16, 8.767.123.696 parametrov skupaj z vidnim delom.
- Prvi fizični API preizkus je končal pred generiranjem zaradi nizovne vsebine
  sistemske vloge (`string indices must be integers, not 'str'`). Popravljeno na
  seznam besedilnih vsebin, kot zahteva večmodalni processor. Izvorna napaka in
  njena sled ostaneta v `provenance/qwen3-vl/warmups/`. Vsebina navodila je enaka.
- 20:16:51 CET: uspešno GPU ogrevanje Qwen3-VL, ločeno od glavnega vzorca.
  20:17:06 in 20:17:22: prva merjena slika vsakega nabora uspešna; štejeta v r1.
- Pripravljeni celotni izvorni zapisi Florence/NVILA in njihovi lastni izrisi
  OD/dense za vseh 20 slik. Skriptni izvoz ne popravlja neveljavnih odgovorov.
- Opravljenih 15 ciljnih testov za ohranitev surovih odgovorov, zavrnitev odmika
  protokola, strogo razčlenitev in obstoječe native pomožne funkcije.
- Lokalni `git push --dry-run` zazna manjkajočo HTTPS prijavo. Preverjena povezava
  GitHub ima pravico push; vsebinski Git objekti se pripravljajo prek te povezave.
  Veja ostane nespremenjena do zaključnega preverjanja in commita.

- 20:32:11 CET: zaključenih vseh 60 Qwen3-VL odgovorov; 60/60 strukturno
  veljavnih, 0 merjenih napak. NVILA ima vseh 60 celotnih odgovorov ohranjenih,
  0/60 po strogi shemi in 52/60 dobesedno dostopnih nepraznih seznamov objektov.
  Florence vseh 180 native odgovorov uspešno razčlenjenih. To niso ocene pravilnosti.
- Zaključni izvoz preveri vseh 180 položajev, 300 različnih izvornih klicev,
  300 bajtno enakih surovih zapisov, 120 native Florencejevih izrisov in 987
  lokalnih HTML povezav. Vseh 180 vrstic obrazca ima prazne presoje. Nobenih
  pripisanih tujih okvirjev; brez drugega koraka ali PDF.
- Prvi zagon preverjevalnika je odkril krožno odvisnost: index je že povezoval
  `verification.json`, preden bi bil ta prvič zapisan. Popravljeno z začetnim
  stanjem `complete=false`, ki postane true šele po vseh preverjanjih. Dodan
  regresijski preizkus, da neuspel ponovni izvoz ne ohrani stare trditve o popolnosti.
- Končnih 16 ciljnih testov opravljenih. Preverjeni Python sintaksa, shell
  sintaksa, nespremenjenost 300 starih zapisov (vključno z meritvami combined)
  in SHA256 dejansko izvedenega novega Qwenovega izvajalnika proti objavljeni kodi.
- GPU proces je zaključen in lastno vzorčenje GPU ustavljeno. Zaključna objava
  po GitHub povezavi uporablja običajen nov commit na izhodiščni veji; lokalni
  commit je ohranjen na arhivski veji. Commit in povezava sta v zaključnem sporočilu.

Strojno preverjene invariance so v `verification.json`. Vse meritve so dejanske;
veljaven zapis, enakost odgovorov in vzorčno ujemanje imen ne dokazujejo zaznavne
pravilnosti. Ročna presoja je namenoma prepuščena praznemu obrazcu.
