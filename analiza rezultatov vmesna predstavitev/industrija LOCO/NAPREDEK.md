# LOCO: dnevnik napredka

- Prebrana CODEX_LOCO.md in opis podatkov; uporabljenih vseh 30 RGB slik.
- Referenčne oznake niso vhod modelov. Preslikava petih razredov zamrznjena pred inferenco.
- Uporaba lokalnih uteži in obstoječih izoliranih slik; brez THUD inferenc ali prepisovanja rezultatov.
- Piloti so merjeni r1 položaji; brez dodatnih ogrevalnih generacij. Nalaganje je zabeleženo posebej.

- 28. 10. 2026, piloti: Florence 3/3 native naloge, Qwen veljaven opis; NVILA celoten odgovor z napako Extra data. Vsi na GPU; piloti vključeni v redne r1 rezultate.
- Florence zaključena: vseh 270 odgovorov, native parse uspešen. NVILA nadaljuje; neveljavnih JSON ne popravljamo in ne ponavljamo.
- Opravljenih 23 testov združljivosti in 13 testov LOCO/protokola/izvozov (prekrivajoči se testi). Preverjen LaTeX izvoz v ločeno mapo v nameščenem okolju.

- Qwen proces se je po 16 shranjenih opisih končal z izhodom 139 (SIGSEGV), brez Python izjeme. Shranjena je konkretna procesna napaka in SHA vseh 16 zapisov. Čas in točen obseg nezapisanega klica nista obnovljiva; vzroka ne ugibamo. Nadaljevanje bo uporabilo iste nastavitve, preskočilo vseh 16 zapisov in vključilo faulthandler za nadaljnjo diagnostiko. To ni prikrito ponavljanje napak formata.

- 28. 10. 2026, 16:20 CET: zaključeno vseh 450 native zapisov / 270 izvornih položajev, 450 različnih call_id, vsi s tokeni in GPU dokazom. Florence 270 in Qwen 90 veljavnih zapisov; NVILA 90 napak JSON s celotnim besedilom. Qwen nadaljevanje je ohranilo vseh 16 prejšnjih SHA. GPU vsebnik nadaljevanja je končal z 0; naknadna napaka zaganjalnika (2) zaradi urejanja aktivne skripte je ločeno shranjena in popravljena brez ponovnih generacij.
- Opravljenih 28 testov. Zamrznjenih 270 vhodov: Florence 90 uporabnih, Qwen 90 uporabnih, NVILA 72 delnih uporabnih in 18 preskokov. Načrt 4536 API klicev + 324 preskokov = 4860 položajev. Kontekstni razredi za uporabne vhode: 237 × 16k, 12 × 32k, 3 × 64k, nespremenjeni pri vseh modelih/promptih/semenih. Začeti GPU piloti treh LLM.

- 28. 10. 2026, 16:39 CET: vsi trije LLM piloti uspešni na GPU, vključeni med redne klice. Celotna mreža teče z enim mestom in enim odjemalcem; napake sheme se ohranijo brez popravnih generacij. Izbranih in slikovno/besedilno pregledanih deset primerov (dva na okolje); izbor v configs/loco-examples.json, opombe pomočnika ločene od praznih uporabnikovih ocen. Pregledane tudi native OD/dense regije, skupinske in prekrivajoče napovedi. Metrični agregati še vedno uporabljajo vseh 30 slik.

- 28. 10. 2026, 17:17 CET: preverjenih 29 testov, nespremenjenih vseh 450 native raw SHA in prvih 16 Qwen zapisov. Priloga s 150 celotnimi izbranimi native odzivi je prevedena v 189 strani PDF brez LaTeX napak/prelivanja. BF16 je obdelal šest skupin osnovnega konteksta; dolgi vhodi nadaljujejo v vnaprej določenih 32k/64k skupinah. Med že izvedenimi vnosi so primeri nad 25.000 tokenov. Doslej ni strežniških opozoril o krajšanju vhodov; inicializacijska opozorila GPU so shranjena posebej. Končna popolnost in poročilo še nista potrjena.

- 28. 10. 2026, 17:39 CET: Qwen BF16 zaključen: 1620 položajev, 1512 generacijskih klicev, statusi {'ok': 1336, 'error': 176, 'skipped': 108}. Preverjeni dejanski 16k/32k/64k konteksti po ločenem hashu uteži. Qwen Q4 nadaljuje, Gemma pilot ostaja v redni mreži. Objavni predogled osvežen iz shranjenih podatkov, brez ponovitev.

- 28. 10. 2026, 19:51 CET: Qwen Q4 zaključen: 1620 položajev, 1388 veljavnih, 124 napak, 108 preskokov (1512 generacijskih klicev). Gemma se je 18:11 CET ustavila po šestih GPU-nepreverjenih odgovorih: gostitelj vidi GPU, vsebnik vrne Failed to initialize NVML: Unknown Error. Osnovni vzrok ni potrjen. Celoten izsek dnevnika in SHA vseh 3247 takratnih zapisov sta ohranjena. Ponovni zagon samo loco-ollama obnovi GPU; nadaljevanje ob 19:50 CET je preskočilo vse shranjene uspehe/napake, prvi novi rezultati so GPU-preverjeni. Nobena od šestih napak ni ponovno generirana. Časi vključujejo prekinitve/pogoje deljenega GPU; to ni neprekinjen izoliran benchmark.

- 28. 10. 2026, 20:04 CET: Gemma po obnovitvi nadaljuje na GPU; doseženih več kot 3900 skupnih položajev. Objavljeni predogled obnovi vseh 1620 BF16 položajev brez izgube raw response, request, promptov, parsanja, GPU in napak. Oba Qwen modela imata dejansko potrjene 16k/32k/64k kontekste. Celoten paket ostaja označen kot nedokončan do zadnjih Gemma skupin in končnih preverjanj.

- 28. 10. 2026, 20:19 CET: prekinitev pogovora je ustavila odjemalca po 4290 shranjenih položajih. Za naslednji Gemma long/r2 vhod 36 je strežnik zapisal HTTP 500/cancel task: položaj je zabeležen kot napaka brez odgovora, samo zahtevek je deterministično rekonstruiran iz zamrznjenih vhodov. Ta klic ni ponovljen. Checkpoint ohrani SHA vseh 4290 obstoječih zapisov. Vsebnik je ponovno izgubil NVML; odjemalec je bil premorjen pri vhodu 69, LOCO vsebnik obnovljen in ista Gemma/16k prednaložena brez prompta (done_reason=load, prazen tekst). Izvirni odgovor 69 je ohranjen z dejanskim odjemalskim časom 117,890 s, ki vključuje ta premor. Prednaložitev ni besedilna generacija; strežniški čas preklicanega klica ni izmišljen kot neznani odjemalski čas. Nadaljevanje preskoči vse shranjene uspehe in napake.

- 28. 10. 2026, 20:21 CET: opravljenih vseh 30 relevantnih testov. Dodan preverjen pripomoček loco_interruption za evidenco dokazanega preklica brez ponovitve/izmišljanja odziva in z zaščito pred prepisom obstoječega rezultata. Izvoz je brez izgube obnovil zahtevek preklicanega položaja, ne da bi dodal odgovor ali neznani odjemalski čas. Povprečja časov uporabljajo samo dejansko časovno izmerjene klice, ločeno od števila vseh poskusov.

- 28. 10. 2026, 20:40 CET: uporabnik je omejil zaključek na že začete generacije, shranjevanje vseh odgovorov/slik, preverjanje obstoja datotek ter commit/push. Samodejni zaključni testi in izdelava novih poročil so ustavljeni. Obstoječa priloga PDF/LaTeX in dosedanji izračuni so ohranjeni; novih testov, ponovitev, obsežnih analiz ali poročil ne izvajamo.

- 28. 10. 2026, 20:43 CET: že začete generacije zaključene, shranjenih vseh 4860 položajev v 810 svežnjih. Ohranjenih 450 native raw datotek, 30 originalnih slik in 180 napovednih prikazov. Preverjen samo obstoj teh datotek; obstoječi pregledovalnik osvežen brez novih analiz, testov ali PDF. Sledi commit/push vseh ključnih rezultatov.
