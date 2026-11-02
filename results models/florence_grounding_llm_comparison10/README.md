# Qwen3.5 27B / 4B na desetih shranjenih Florence vhodih

Odprite `index.html`. Oba modela sta Q4_K_M. En klic na model in vhod;
T=0.3, seme 42, kontekst 8192, največ 2048 tokenov, think=false.
Obstoječa prompta: configs/prompts/reasoning-system.txt in reasoning.txt.
`manifest.json` vsebuje točno besedilo in SHA256 desetih nespremenjenih vhodov;
`responses/` celotne zahteve, odzive, čase in GPU alokacijo; `models/` metapodatke.
Označene slike so kopije obstoječih Florence napovedi. Florence ni ponovno zagnan.

Ponovitev/nadaljevanje: `bash scripts/florence_grounding_llm.sh` na obstoječem
Ollama strežniku z nameščenima modeloma. Shranjeni poskusi se ne ponavljajo.
Za samo ponovni izvoz uporabite argument `export`.

Ločeni `preflight.json`: kratek izhod 16 tokenov na najdaljšem besedilnem vhodu,
pri celotnem kontekstu 8192. Ni del primerjave. Pred njim je bil prekinjen en
CPU poskus zaradi izgubljenega NVML dostopa dolgo delujočega vsebnika;
napaka in stanje sta ohranjena v `setup/`. Ponovni zagon istega vsebnika je
obnovil GPU dostop. `gpu-preflight-ollama-ps.txt` potrjuje 100% GPU.

Čas v HTML je stenski čas celotnega API klica in lahko vključuje nalaganje.
Ni analize pravilnosti, primerjav razredov, ponovitev ali PDF. Uteži ostanejo lokalno.

## Dodatek: Gemma 3 27B

`bash scripts/florence_grounding_llm.sh run --gemma-only` doda deset klicev
gemma3:27b-it-q4_K_M in en ločen kratek GPU preizkus pri kontekstu 8192.
Za sam izvoz: `bash scripts/florence_grounding_llm.sh export --gemma-only`.
`gemma3-manifest.json` hrani iste zamrznjene vhode, prompt in nastavitve ter razlike:
Gemma nima načina thinking; izvorna pogovorna predloga in zaključni tokeni so drugačni.
HTML ima tretji stolpec. Qwenovi odzivi in Florence so ponovno uporabljeni.
