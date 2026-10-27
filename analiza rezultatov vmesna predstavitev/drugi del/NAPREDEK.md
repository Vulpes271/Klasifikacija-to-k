# Dnevnik drugega koraka

- 27. 10. 2026: izhodišče `d76deb9419f93d79f8df9f8ed64779547a4db071`.
  Prvi korak ni ponovno izveden. Za vseh 888 njegovih objavljenih datotek so
  shranjeni kontrolni SHA256 za zaključno preverjanje nespremenjenosti.
- Pred inferenco zamrznjenih 180 vhodov: isti izbor 20 slik × 3 VLM × 3 njihove
  ponovitve. Vsak vhod ima indeks, izvorne SHA256 in stabilne lokalne ID-je.
  Enoten dokument se brez sprememb uporabi pri vseh 18 LLM kombinacijah.
- 172 vhodov je uporabnih. Osem NVILA odgovorov nima zanesljivega dobesednega
  seznama; vseh 144 pripadajočih položajev je vnaprej določenih kot preskok.
  Načrt je 3096 glavnih klicev in skupaj 3240 položajev.
- Florencejev obstoječi pretvornik ohrani caption in vse OD/dense regije, njihove
  izvorne naloge in lokalne ID-je regij. Prekrivanje ni potrjena različnost
  fizičnih objektov. NVILA-jevi delni vhodi ohranijo celotno neveljavno besedilo,
  dobesedni seznam in opozorila o transportnih zastavicah. Qwenovi opisi ostanejo
  nespremenjeni. Nobenega dodajanja vidnih dejstev ali nadomestitve ponovitev.
- 21:18 CET: uspešno po en pilot za Qwen BF16, Qwen Q4 in Gemma Q4. Vsi na GPU,
  vsi po isti konfiguraciji kot poznejši kratki prompt / LLM-r1. Piloti štejejo
  med glavne klice, dodatnih fizičnih ogrevanj ni.
- Začetek celotnega poskusa v obstoječem okolju. GPU je deljen z drugim Python
  procesom; časi zato niso izoliran benchmark. Modeli tečejo zaporedno.
- Pripravljena sta skriptna analiza in začasni HTML pregled za vse slike in
  izvore. Dokler vsi položaji niso shranjeni, `verification.json` izrecno
  označuje `complete=false`. Razlike ne temeljijo na prepisovanju ali krajšanju
  surovih odgovorov. Neveljavni odgovori ostanejo napake.
- 21:32 CET: odkrito, da devet dolgih Florencejevih vhodov pri kontekstu 16384
  ne pusti dovolj prostora za celoten odgovor oziroma strežnik vhod skrči.
  Lastni proces je ustavljen po atomarnem zapisu rezultata in summary, preverjeno
  brez nedokončanega klica. Vsi uspešni klici so ohranjeni.
- Zamrznjeni vhodi ostanejo nespremenjeni. Ločena nespremenljiva politika določi
  kontekst 32768 samo za devet vhodov s promptom nad 30.000 znakov, pri vseh LLM,
  obeh promptih in vseh ponovitvah. Drugi vhodi ostanejo pri 16384. Arhiviranih
  je vseh devet neuspešnih tehničnih poskusov s prvotnimi nastavitvami in SHA256;
  ponovljeni so samo ti neuspešni položaji. Število fizičnih klicev vključuje
  tudi teh devet poskusov, pričakovano 3105, ne samo 3096 glavnih.
- 21:33:59 CET: nadaljevanje samo manjkajočih položajev. Prvi dolgi vhod zdaj
  obdelan v celoti: 18.477 vhodnih tokenov, 58 regij, veljaven celoten odgovor.
- Opravljenih 14 ciljnih testov in 14 obstoječih testov ločenih stopenj.
  Preizkušeni so ohranitev neveljavnega teksta, izvori prekrivajočih regij,
  zavračanje ujemanja različnih VLM samo po ID in kontekstna izjema ob
  nespremenjenem zamrznjenem opisu, semenu in temperaturi.
- Nadaljevanje v ločenem Ollama vsebniku iste lokalne slike 0.40.4 in z istim
  modelnim volumnom samo za branje. Prvotna storitev ostane nedotaknjena.
  Zahtevana štiri strežniška mesta niso dejanska sočasnost: arhitektura qwen35
  poroča `n_seq_max=1`; štirje odjemalski delavci pri njej čakajo v vrsti.
  Kontekstni fazi 16384/32768 sta razporejeni ločeno. Ta sprememba streženja,
  čakanje in druge obremenitve GPU so dokumentirane omejitve meritev časa.
- 22:45 CET: shranjenih več kot 980 glavnih klicev. BF16 zaključuje krajše
  vhode; sledijo vsi njegovi daljši vhodi ter preostala modela. Začasni izvoz
  ostaja izrecno nepopoln. Popolni šestčlenski skupni zapisi so pripravljeni
  za objavo brez krajšanja odgovorov; kazalec Git veje še ni premaknjen.
- 23:11 CET: BF16 zaključen: 1080 položajev, od tega 936 veljavnih odzivov,
  96 napak in 48 vnaprej določenih preskokov. To so statusi zapisa, ne ocena
  pravilnosti klasifikacije. Začela se je celotna izvedba Qwen Q4; njegovi
  odgovori se shranjujejo v ločene mape in ne nadomeščajo BF16 rezultatov.
- 23:33 CET: Qwen Q4 je pri vhodu 61, kratkem promptu in LLM-r1 vrnil HTTP 500
  `prediction aborted, token repeat limit reached`. Zapisano kot dejanska
  napaka strežnika, brez ponovnega klica ali izmišljanja odgovora. Agregati in
  preverjanje dodatno ločijo število API poskusov od dokončanih odzivov.
- 28. 10. 2026, 00:04 CET: Qwen Q4 je dopolnil vseh šest skupin z vhodi za
  kontekst 16384 in obdeluje še devet daljših dokumentov v vsaki skupini.
  Strežniški dnevnik potrjuje eno dejansko mesto tudi za njegov kontekst
  32768. Skupaj shranjenih več kot 2100 položajev; poskus še ni zaključen.
- 00:20 CET: Qwen Q4 zaključen: 955 veljavnih odzivov, 77 napak, 48 preskokov.
  Od 1032 API poskusov ima 1031 celoten odgovor; enega je strežnik prekinil.
  Začela se je Gemma. Dnevnik zanjo izrecno potrjuje `n_seq_max=4`, skupni
  kontekst 65536 in štiri mesta s po 16384. Njena dejanska sočasnost je zato
  različna od Qwenove; časi HTTP niso neposredno primerljiv izoliran benchmark.

- 00:44 CET: celotna inferenca zaključena z izhodno kodo 0. Gemma ima 916
  veljavnih odzivov, 116 napak in 48 preskokov. Skupaj: 2807 veljavnih zapisov,
  289 napak in 144 preskokov, vseh 3240 položajev. To so statusi sheme,
  ne pravilnost zaznavanja ali razvrstitve. Opravljenih 3096 glavnih API poskusov
  in devet arhiviranih kontekstnih poskusov, skupaj 3105; prejetih 3104 odzivov,
  en HTTP 500. Uspešnih klicev nismo ponavljali.
- Dnevnik potrjuje štiri Gemma mesta tudi pri kontekstu 32768 (skupno 131072).
  Poskusni strežnik ustavljen; prvotna Ollama storitev ostaja vključena.
  Uteži in okolja ostanejo lokalno. Surovi zapisi izvajalnika so ohranjeni.
- Končni skriptni izvoz in `verification.json`: complete=true. Preverjenih je
  540 skupnih datotek po šest položajev, vseh 180 zamrznjenih SHA256, točna
  sporočila in nastavitve, enolični odzivi, 10.688 HTML povezav in 3240 praznih
  vrstic ročnega obrazca. Vseh 888 datotek prvega koraka je nespremenjenih.
- Dodatno preverjena obnovitev vseh 3240 zapisov samo iz objavljenih datotek:
  zahteve, odzivi, statusi, napake in rezultati razčlenjevanja so enaki lokalnim.
  Modeli in ignorirane mape izvajalnika za pregled ali analizo niso potrebni.
- Prestalo je 14 ciljnih testov in 14 testov obstoječih ločenih stopenj;
  preverjeni sta tudi Python sintaksa in sintaksa obeh shell izvajalnikov.
  Zaključeni paket je pripravljen za commit in objavo na isti Git veji.

Poskus je zaključen in preverjen. Zaključna števila so v `verification.json`,
vse razlike in imenovalci pa v primerjalnih tabelah. PDF in prvi korak v tej
nalogi nista bila ponovno izvedena. Identiteto objavljene različice določa
commit v zgodovini Git; operativno potrdilo objave se shrani tudi lokalno.
