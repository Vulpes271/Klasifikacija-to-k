# Napredek 1. dela

- 2026-10-20: prebrana CODEX_NALOGA.md, README in WORKFLOW; veja feature/separate-vision-stages, izhodišče 624caef. Obstoječi vision-launch.log ohranjen.
- Začetnih 46 testov uspešnih zunaj peskovnika (results/thud-vmesno/tests-initial.log). Peskovnik preprečuje lokalne HTTP vtičnice in dostop do GPU/Docker.
- Potrjena RTX 5090, 32607 MiB, gonilnik 610.57.04. Drug proces uporablja GPU; ni ustavljen. Projektna Ollama zdrava; BF16 in Q4 že nameščena.
- Uradni katalog potrjuje vse tri točne tage in slikovno podporo. Razlika BF16/Q4 vključuje tudi različno natančnost slikovnega projektorja; metapodatki se ohranijo.
- Drugi korak cevovoda spada v 1. del naloge. 2. del nadgradnje se ne izvaja.

- 2026-10-20 21:36:49: Podatki preverjeni: 50/678 in 50/1021; zamrznjen razpršen izbor 10+10 slik ter ločeni vhodi brez anotacij.

- 2026-10-20 21:37:01: Podatki preverjeni: 50/678 in 50/1021; zamrznjen razpršen izbor 10+10 slik ter ločeni vhodi brez anotacij.

- 2026-10-20 21:37:01: Zagon pilot-qwen35-bf16-discovery-thud_dense_test50

- 2026-10-20 21:37:44: Podatki preverjeni: 50/678 in 50/1021; zamrznjen razpršen izbor 10+10 slik ter ločeni vhodi brez anotacij.

- 2026-10-20 21:37:44: Zagon pilot-qwen35-bf16-discovery-thud_dense_test50

- 2026-10-20 21:37:45: Končano pilot-qwen35-bf16-discovery-thud_dense_test50: izhod 2, 0.1 s

- 2026-10-20 21:37:45: Zagon pilot-qwen35-bf16-discovery-thud_agtea_test50

- 2026-10-20 21:39:16: Končano pilot-qwen35-bf16-discovery-thud_agtea_test50: izhod 0, 91.5 s

- 2026-10-20 21:39:16: Zagon pilot-qwen35-bf16-supplied-thud_dense_test50

- 2026-10-20 21:40:21: Končano pilot-qwen35-bf16-supplied-thud_dense_test50: izhod 0, 65.2 s

- 2026-10-20 21:40:21: Zagon pilot-qwen35-bf16-supplied-thud_agtea_test50

- 2026-10-20 21:41:02: Končano pilot-qwen35-bf16-supplied-thud_agtea_test50: izhod 0, 40.9 s

- 2026-10-20 21:41:02: Zagon pilot-qwen35-q4-discovery-thud_dense_test50

- 2026-10-20 21:41:49: Končano pilot-qwen35-q4-discovery-thud_dense_test50: izhod 0, 47.0 s

- 2026-10-20 21:41:49: Zagon pilot-qwen35-q4-discovery-thud_agtea_test50

- 2026-10-20 21:42:23: Končano pilot-qwen35-q4-discovery-thud_agtea_test50: izhod 0, 34.2 s

- 2026-10-20 21:42:23: Zagon pilot-qwen35-q4-supplied-thud_dense_test50

- 2026-10-20 21:42:35: Podatki preverjeni: 50/678 in 50/1021; zamrznjen razpršen izbor 10+10 slik ter ločeni vhodi brez anotacij.

- 2026-10-20 21:43:23: Končano pilot-qwen35-q4-supplied-thud_dense_test50: izhod 0, 59.6 s

- 2026-10-20 21:43:23: Zagon pilot-qwen35-q4-supplied-thud_agtea_test50

- 2026-10-20 21:44:26: Končano pilot-qwen35-q4-supplied-thud_agtea_test50: izhod 0, 62.5 s

- 2026-10-20 21:44:26: Zagon pilot-gemma3-q4-discovery-thud_dense_test50

- 2026-10-20 21:44:47: Končano pilot-gemma3-q4-discovery-thud_dense_test50: izhod 0, 21.3 s

- 2026-10-20 21:44:47: Zagon pilot-gemma3-q4-discovery-thud_agtea_test50

- 2026-10-20 21:44:57: Končano pilot-gemma3-q4-discovery-thud_agtea_test50: izhod 0, 10.3 s

- 2026-10-20 21:44:57: Zagon pilot-gemma3-q4-supplied-thud_dense_test50

- 2026-10-20 21:45:15: Podatki preverjeni: 50/678 in 50/1021; zamrznjen razpršen izbor 10+10 slik ter ločeni vhodi brez anotacij.

- Dopolnjeno: shranjevanje celotnih teles HTTP napak in časov neuspelih zahtev; zavrnitev delnega CPU offloada; ločeni ogrevalni klici. Testi za ohranitev HTTP telesa in prompte dodani.
- Začetna CPU diagnostika (dva neuspešna merjena klica ob prekinitvi kontejnerja) ohranjena v `results/thud-vmesno/diagnostic-cpu-initial-pilot/`; ne šteje v obnovljeni GPU pilot. Podatki/utegi niso bili izbrisani.

- 2026-10-20 21:45:58: Končano pilot-gemma3-q4-supplied-thud_dense_test50: izhod 0, 60.7 s

- 2026-10-20 21:45:58: Zagon pilot-gemma3-q4-supplied-thud_agtea_test50

- Pred polnim zagonom: vseh 53 testov uspešnih (`results/thud-vmesno/tests-pre-run.log`). Koda poskusa in prompti lokalno potrjeni v commitu `3f48caf`; Git nima uporabniške identitete, zato commit uporablja avtorja `Codex <codex@localhost>` samo za ta ukaz, brez globalnih sprememb.
- Za meritve se ločeno beležijo API generiranje/tokeni, nalaganje, ogrevanje in end-to-end čas; skupni GPU se dodatno vzorči v `results/thud-vmesno/gpu-samples.jsonl`.

- 2026-10-20 21:48:22: Končano pilot-gemma3-q4-supplied-thud_agtea_test50: izhod 2, 144.0 s

- 2026-10-20 21:48:22: Pilot in ocena shranjena: {'pilot_calls': 22, 'baseline_measured_calls': 600, 'baseline_warmups': 12, 'additional_measured_calls': 200, 'additional_warmups_stage1': 8, 'estimated_baseline_seconds': 9647.948387777433, 'note': 'Linear pilot estimate; variable output length and external GPU contention limit prediction.'}

- 2026-10-20 21:48:22: Zahtevana faza zaključena; surovi odgovori, napake in konfiguracije ohranjeni.

- 2026-10-20 21:48:38: Podatki preverjeni: 50/678 in 50/1021; zamrznjen razpršen izbor 10+10 slik ter ločeni vhodi brez anotacij.

- 2026-10-20 21:48:38: Zagon pilot-qwen35-bf16-discovery-thud_dense_test50

- 2026-10-20 21:48:54: Končano pilot-qwen35-bf16-discovery-thud_dense_test50: izhod 0, 16.0 s

- 2026-10-20 21:48:54: Pilot in ocena shranjena: {'pilot_calls': 24, 'baseline_measured_calls': 600, 'baseline_warmups': 12, 'additional_measured_calls': 200, 'additional_warmups_stage1': 8, 'additional_warmups_stage2': 12, 'estimated_baseline_seconds': 9067.483272089157, 'note': 'Linear pilot estimate; variable output length and external GPU contention limit prediction.'}

- 2026-10-20 21:48:54: Zagon baseline-qwen35-bf16-discovery-thud_dense_test50

- GPU pilot: 24 odgovorov, 23 ustreza shemi, 1 napaka manjkajočega ID (Gemma/regije/agtea). Linearna ocena osnovnih 600 klicev: 9067 s oziroma približno 2,5 h, brez dodatnih testov in ogrevanj. Osnovni poskus se je začel okrog 21:49.
- Dodatna metodološka omejitev: BF16 metapodatki poročajo 4.326.350.848 parametrov, 33 blokov in nextn_predict_layers=1; Q4 4.205.751.296 parametrov in 32 blokov. Slikovna projektorja F16/BF16. Tagov ne predstavljamo kot popolnoma izolirane spremembe kvantizacije iste pretvorbe; vpliv dodatne napovedne plasti ni posebej preverjen.
- Aktivna koda osnovnega zagona: commit `399d6ea`; SHA256 vseh izvornih modulov/promptov/nastavitev je v `results/thud-vmesno/source-snapshots/all-*.json`.

- 2026-10-20 21:57:22: Končano baseline-qwen35-bf16-discovery-thud_dense_test50: izhod 0, 508.0 s

- 2026-10-20 21:57:22: Zagon baseline-qwen35-bf16-discovery-thud_agtea_test50

- 2026-10-20 22:01:12: Končano baseline-qwen35-bf16-discovery-thud_agtea_test50: izhod 0, 230.4 s

- 2026-10-20 22:01:12: Zagon baseline-qwen35-bf16-supplied-thud_dense_test50

- 2026-10-20 22:09:14: Končano baseline-qwen35-bf16-supplied-thud_dense_test50: izhod 2, 481.9 s

- 2026-10-20 22:09:14: Zagon baseline-qwen35-bf16-supplied-thud_agtea_test50

- 2026-10-20 22:29:06: Končano baseline-qwen35-bf16-supplied-thud_agtea_test50: izhod 2, 1192.1 s

- 2026-10-20 22:29:06: Zagon baseline-qwen35-q4-discovery-thud_dense_test50

- 2026-10-20 22:31:49: Končano baseline-qwen35-q4-discovery-thud_dense_test50: izhod 0, 163.2 s

- 2026-10-20 22:31:49: Zagon baseline-qwen35-q4-discovery-thud_agtea_test50

- 2026-10-20 22:34:27: Končano baseline-qwen35-q4-discovery-thud_agtea_test50: izhod 0, 158.0 s

- 2026-10-20 22:34:27: Zagon baseline-qwen35-q4-supplied-thud_dense_test50

- 22:35: BF16 zaključil vseh 200 osnovnih klicev: samostojni način 100/100 po shemi, regije 94/100; vseh šest zavrnitev je preimenovalo podane oznake. Q4 zaključil samostojni način 100/100 in nadaljuje regije.
- Revizija do zdaj dokončanih skupin ni našla uhajanja anotacij, različnih slikovnih bajtov ali CPU offloada. Preglednik podpira prikaz izvornih okvirjev; vsebinski komentarji za štiri slike so v `manual-notes.json`, sistematična ročna polja ostajajo `pending`.
- Sintaktično berljivi zavrnjeni kandidati so ločeno razčlenjeni v lokalnem `rejected-candidates.jsonl`, brez spreminjanja izvornih odgovorov ali statusa. Generator in revizija sta v commitu `ad9998a`; inferenca še vedno uporablja zamrznjeni izvor `399d6ea`.

- 2026-10-20 22:39:30: Končano baseline-qwen35-q4-supplied-thud_dense_test50: izhod 0, 303.2 s

- 2026-10-20 22:39:30: Zagon baseline-qwen35-q4-supplied-thud_agtea_test50

- 2026-10-20 22:46:27: Končano baseline-qwen35-q4-supplied-thud_agtea_test50: izhod 2, 416.8 s

- 2026-10-20 22:46:27: Zagon baseline-gemma3-q4-discovery-thud_dense_test50

- 2026-10-20 22:49:32: Končano baseline-gemma3-q4-discovery-thud_dense_test50: izhod 0, 184.9 s

- 2026-10-20 22:49:32: Zagon baseline-gemma3-q4-discovery-thud_agtea_test50

- 2026-10-20 22:52:51: Končano baseline-gemma3-q4-discovery-thud_agtea_test50: izhod 0, 199.2 s

- 2026-10-20 22:52:51: Zagon baseline-gemma3-q4-supplied-thud_dense_test50

- 2026-10-20 22:58:46: Končano baseline-gemma3-q4-supplied-thud_dense_test50: izhod 2, 354.3 s

- 2026-10-20 22:58:46: Zagon baseline-gemma3-q4-supplied-thud_agtea_test50

- 2026-10-20 23:06:16: Končano baseline-gemma3-q4-supplied-thud_agtea_test50: izhod 2, 450.7 s

- 2026-10-20 23:06:16: Zagon extra-short-t0-thud_dense_test50

- 2026-10-20 23:07:00: Končano extra-short-t0-thud_dense_test50: izhod 0, 43.4 s

- 2026-10-20 23:07:00: Zagon extra-long-t03-thud_dense_test50

- 23:06: končana celotna osnovna primerjava 600 GPU odgovorov. Samostojni način 300/300 po shemi; dane regije 276/300. BF16 6 zavrnitev (preimenovanje), Q4 3, Gemma 15. Vse zavrnitve so ohranjene. End-to-end vsota osnovnih skupin: približno 77,4 min; pilotna ocena je bila višja zaradi variabilnosti in drugih GPU bremen.
- Začeti dodatni testi: BF16 kratek/podroben prompt in T/seed na istem zamrznjenem izboru 20 slik, nato besedilni drugi korak brez slike. Širitev 2. dela se ne izvaja.

- 2026-10-20 23:08:13: Končano extra-long-t03-thud_dense_test50: izhod 0, 73.5 s

- 2026-10-20 23:08:13: Zagon extra-long-t07-thud_dense_test50

- 2026-10-20 23:09:34: Končano extra-long-t07-thud_dense_test50: izhod 0, 80.4 s

- 2026-10-20 23:09:34: Zagon extra-long-t03-s43-thud_dense_test50

- 2026-10-20 23:10:45: Končano extra-long-t03-s43-thud_dense_test50: izhod 0, 70.9 s

- 2026-10-20 23:10:45: Zagon reason-long-examples-t0.3-s42-thud_dense_test50

- 2026-10-20 23:10:59: Končano reason-long-examples-t0.3-s42-thud_dense_test50: izhod 0, 14.4 s

- 2026-10-20 23:10:59: Zagon reason-long-no-examples-t0.3-s42-thud_dense_test50

- 2026-10-20 23:11:15: Končano reason-long-no-examples-t0.3-s42-thud_dense_test50: izhod 2, 15.9 s

- 2026-10-20 23:11:15: Zagon reason-short-no-examples-t0.3-s42-thud_dense_test50

- 2026-10-20 23:11:29: Končano reason-short-no-examples-t0.3-s42-thud_dense_test50: izhod 0, 14.3 s

- 2026-10-20 23:11:29: Zagon reason-long-no-examples-t0-s42-thud_dense_test50

- 2026-10-20 23:12:07: Končano reason-long-no-examples-t0-s42-thud_dense_test50: izhod 0, 37.9 s

- 2026-10-20 23:12:07: Zagon reason-long-no-examples-t0.7-s42-thud_dense_test50

- 2026-10-20 23:12:29: Končano reason-long-no-examples-t0.7-s42-thud_dense_test50: izhod 0, 21.9 s

- 2026-10-20 23:12:29: Zagon reason-long-no-examples-t0.3-s43-thud_dense_test50

- 2026-10-20 23:12:54: Končano reason-long-no-examples-t0.3-s43-thud_dense_test50: izhod 0, 24.7 s

- 2026-10-20 23:12:54: Zagon extra-short-t0-thud_agtea_test50

- 2026-10-20 23:13:47: Končano extra-short-t0-thud_agtea_test50: izhod 0, 52.9 s

- 2026-10-20 23:13:47: Zagon extra-long-t03-thud_agtea_test50

- 2026-10-20 23:14:45: Končano extra-long-t03-thud_agtea_test50: izhod 0, 58.0 s

- 2026-10-20 23:14:45: Zagon extra-long-t07-thud_agtea_test50

- 2026-10-20 23:15:34: Končano extra-long-t07-thud_agtea_test50: izhod 0, 49.1 s

- 2026-10-20 23:15:34: Zagon extra-long-t03-s43-thud_agtea_test50

- 2026-10-20 23:17:03: Končano extra-long-t03-s43-thud_agtea_test50: izhod 0, 89.5 s

- 2026-10-20 23:17:03: Zagon reason-long-examples-t0.3-s42-thud_agtea_test50

- 2026-10-20 23:17:34: Končano reason-long-examples-t0.3-s42-thud_agtea_test50: izhod 0, 30.6 s

- 2026-10-20 23:17:34: Zagon reason-long-no-examples-t0.3-s42-thud_agtea_test50

- 2026-10-20 23:18:05: Končano reason-long-no-examples-t0.3-s42-thud_agtea_test50: izhod 2, 31.6 s

- 2026-10-20 23:18:05: Zagon reason-short-no-examples-t0.3-s42-thud_agtea_test50

- 2026-10-20 23:18:31: Končano reason-short-no-examples-t0.3-s42-thud_agtea_test50: izhod 0, 25.5 s

- 2026-10-20 23:18:31: Zagon reason-long-no-examples-t0-s42-thud_agtea_test50

- 2026-10-20 23:18:53: Končano reason-long-no-examples-t0-s42-thud_agtea_test50: izhod 0, 22.3 s

- 2026-10-20 23:18:53: Zagon reason-long-no-examples-t0.7-s42-thud_agtea_test50

- 2026-10-20 23:19:12: Končano reason-long-no-examples-t0.7-s42-thud_agtea_test50: izhod 2, 18.9 s

- 2026-10-20 23:19:12: Zagon reason-long-no-examples-t0.3-s43-thud_agtea_test50

- 2026-10-20 23:19:32: Končano reason-long-no-examples-t0.3-s43-thud_agtea_test50: izhod 0, 19.8 s

- 2026-10-20 23:19:32: Zahtevana faza zaključena; surovi odgovori, napake in konfiguracije ohranjeni.

- 2026-10-20 23:21:43: Zagon reason-q4-long-no-examples-t0.3-s42-thud_dense_test50

- 2026-10-20 23:21:54: Končano reason-q4-long-no-examples-t0.3-s42-thud_dense_test50: izhod 2, 10.2 s

- 2026-10-20 23:21:54: Zagon reason-q4-long-no-examples-t0.3-s42-thud_agtea_test50

- 2026-10-20 23:22:12: Končano reason-q4-long-no-examples-t0.3-s42-thud_agtea_test50: izhod 2, 18.3 s

- 2026-10-20 23:22:12: Dodatna besedilna Q4 ponovitev zaključena na istih 20 zamrznjenih opisih.

- 23:22: zaključeni vsi dodatni testi: 80/80 opisovalnih odgovorov ustreza shemi; BF16 reasoning 115/120, dodatni Q4 reasoning 17/20. Razčlenjevalnik je v dejanskih odgovorih zavrnil tudi podvojene JSON ključe; ni izbral zadnje vrednosti.
- Končni obseg brez pilota: 820 odgovorov, vseh 820 s potrjeno polno GPU dodelitvijo; 788 ustreza shemi, 32 zavrnjenih, brez preskokov. Pilot: 24 GPU odgovorov. Ogrevalni klici: 46, vsi ločeno ohranjeni in izključeni iz kakovostnih imenovalcev. Skupaj 844 merjenih GPU klicev.
- Revizija 34/34 skupin uspešna: enaki slikovni bajti, nespremenjeni opisi, brez anotacij v discovery zahtevah, brez slik v stage2, brez CPU offloada. 54/54 testov uspešnih; glej tests.json in verification.json.
- PDF: 6 strani; vizualno pregledanih vseh šest, povečano preverjeni tabeli promptov in utemeljitev ter fotografije. Popravljena tipografska puščica in izbor dejanskega pohištva v tabeli primerov. Brez LaTeX napak ali prelivanja.
- Predlog naslednjega pregleda je kratki prompt brez primerov (20/20 po shemi, enako kot dolgi s primeri; dolgi brez primerov 16/20). To je kandidat za ročni pregled, ne potrjena pravilnost. Vsa sistematična ročna polja ostanejo pending. NVILA ostaja neizveden; GPU je bil deljen in pretvorbi Qwen nista strogo izoliran poseg v kvantizacijo. 2. del ni izveden.

- Končni PDF uporablja vektorske pisave Latin Modern in Unicode preslikavo: šest strani, pravilno kopirljivi šumniki, brez Type 3 pisav, LaTeX napak ali prelivanja. Dokazila so v `pdf-validation.json`; končni predogledi vseh strani so lokalno v `results/thud-vmesno/pdf-preview/`.
- Samostojen paket poročila je pripravljen za Git; surovi izhodi (približno 806 MB) ostajajo lokalno. Obstoječi uporabnikov `vision-launch.log` je ohranjen. Vsa koda, prompti, nastavitve, CSV/JSON, slike, LaTeX in PDF so pripravljeni za shranitev na delovno vejo.

- Lokalni zaključni commit s kodo in celotnim paketom poročila: `06f42d3`. Push je prvotno zavrnil samodejni varnostni pregled zaradi nepotrjenega oddaljenega cilja. Preverjeno je natančno ujemanje `origin` z repozitorijem in izrecnim navodilom za push v CODEX_NALOGA.md; ponovni pregled je ukaz dovolil.
- Dejanski `git push -u origin feature/separate-vision-stages` je nato odpovedal z izhodom 128: `fatal: could not read Username for 'https://github.com': terminal prompts disabled`. Za objavo manjkajo GitHub poverilnice s pravico pisanja. Lokalni commiti, poročilo in vsi rezultati so ohranjeni; noben podatek ni bil potisnjen na oddaljeno vejo.
