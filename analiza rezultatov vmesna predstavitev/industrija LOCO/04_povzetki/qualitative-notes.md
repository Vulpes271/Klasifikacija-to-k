# Opombe ob pregledu desetih primerov

Opombe pomočnika: niso uporabnikove ročne ocene ali ground truth. Pregled izvirnih slik, r1 opisov/izluščenih seznamov ter r1 OD/dense napovedi; razlike drugih ponovitev so izračunane skriptno. Vsi uporabnikovi obrazci ostanejo prazni.

## loco-s1-33078

Oranžen transportni stroj v ospredju in skladiščni regali so prepoznavni pri vseh treh modelih.

Florence postavi moder zaboj na levo, na sliki je izrazit moder zaboj desno. NVILA trdi, da so regali pritrjeni na steno, brez navedenega vidnega dokaza. Opisi stationary/moving iz ene fotografije niso potrjen časovni podatek.

Florence OD r1 vrne 28 box in 9 tag regij, brez okvirja za osrednji viličar, čeprav ga caption poimenuje. Dense je podoben; regije se med nalogama prekrivajo.

## loco-s1-33243

Vsi trije opisi prepoznajo mrežast zaboj v ospredju in delno zakrite regale.

Imena metal cage-like structure, metal cage pallet in metal cage niso potrjena identiteta med modeli. Qwen navede leseno paletno osnovo; ta na pregledu ni jasno potrjena. Oznaka visible_installation za regal zahteva ločen pregled dejanskega sidranja.

Florence OD veliko prednjo regijo poimenuje box, dense isto območje opiše kot galvanized welded wire mesh cooling tower in warehouse. Funkcija hladilnega stolpa s slike ni potrjena; geometrijska lokalizacija sama ne potrdi pomena.

## loco-s2-34608

Temen dolg regalni hodnik, veliki leseni zaboji in svetila so dobro razpoznavni v opisih.

Qwen pri blue plastic bins vrne visible_installation, čeprav opazovanje opisuje le postavitev na polici. Pri rack hkrati omenja bolted to floor in appears_freestanding. NVILA r1 ima le en izluščen objekt, čeprav opis scene omenja več kategorij; to je delni vhod, ne popoln popis.

Florence OD vrne 13 box regij, dense pa prazno regijsko napoved; prazen seznam dense ni trditev, da je scena prazna.

## loco-s2-34489

Vsi modeli zaznajo zamegljenost in paletno/regalno okolje; Qwen je pri rdečem stroju previden glede tipa.

Qwen pri cardboard boxes uporabi visible_installation, opazovanje pa govori o podpori police. Florence konkretno poimenuje fire extinguisher; zaradi zamegljenosti je treba to podrobnost dodatno preveriti. Blur ni dovoljenje za dopolnjevanje podrobnosti.

Florence OD vrne en okvir shelf čez velik del zamegljenega regala, dense je prazen. To ni zaznava posameznih palet.

## loco-s3-36694

Vozilo z odprtim prtljažnikom, človek in rumen mrežast voziček so vidni ter opisani pri vseh treh modelih.

Nalaganje/razlaganje je razumna, a iz ene fotografije nepotrjena razlaga aktivnosti. Florence rumeni element poimenuje pallet truck, ostala opisa cage/cart. Ne gre za potrjeno ujemanje imen ali fizičnih ID-jev. Odprt prtljažnik ne spremeni definicije prestavljivosti celotnega vozila.

Florence OD okvirji vključujejo car in person; dense navede Hyundai i20. Vrsta/model vozila nista potrjena z referenco LOCO. Napovedan okvir in natančno ime sta ločeni trditvi.

## loco-s3-36540

Delno odrezan oranžen transportni stroj desno in rdeči regali so opisani pri vseh modelih.

Qwen za moder sod navede, da očitno vsebuje tekočino; na pregledani sliki vsebina ni vidna. Pri skladovnicah razlikuje zaboje in palete, vendar en opis hkrati govori o neposrednem stiku s tlemi in podpori palete. Ti odnosi zahtevajo pregled.

Florence OD ima dva široka shelf okvirja, dense en celosten opis regala; to sta skupinski regiji, ne popis vsakega zaboja.

## loco-s4-31835

Beli kartoni z napisom POTT, lesene palete in regalna struktura so jasno razpoznavni.

Florence napiše, da so črke POTT bele; na sliki je temen napis na svetli embalaži. NVILA r1 zaradi neveljavnega zapisa nima zanesljivega izluščenega seznama, čeprav celotni raw vsebuje opis in objektnim podobne odseke; preskok je pravilen protokolarni izid. Qwen pritrditve označi kot vidne, a v tekstu priznava, da sidra niso jasno vidna.

Florence OD vrne le en box okvir ob spodnjem levem robu, dense nima regij. Dolg caption zato ne pomeni primerljivega obsega detekcij.

## loco-s4-32693

Širok prehod, visoki regali, rumena vozila in rumene zaščite so prepoznani.

Florence in NVILA sklepata o gibanju oddaljenega vozila, kar ena fotografija ne potrjuje. Qwen yellow safety bollard označi appears_freestanding; videz sam ne potrdi, ali je zaščita sidrana. To je primer razlike med opaženim stikom in dokazom namestitve.

Florence OD vrne le delno odrezano levo land vehicle regijo, dense je prazen. Napovedi ne preimenujemo v referenčni forklift brez ustrezne vnaprejšnje preslikave.

## loco-s5-35448

Oseba z odsevnim jopičem, oranžna transportna vozila in regali so dejansko vidni; odsotnost človeka med petimi referenčnimi razredi ni halucinacija.

Qwen napiše visible from waist down, čeprav je viden tudi trup z jopičem. NVILA r1 izlušči samo Worker; drugih objektov v opisu scene to ne izniči. Qwen red safety block razglasi za freestanding brez jasnega dokaza o odsotnosti pritrditve.

Florence vrne prekrivajoče se regije person, trousers in footwear; dense dodatno opiše ista oblačila/osebo. To niso štirje potrjeno različni ljudje.

## loco-s5-35671

Zamegljen mimoidoči, skladiščni regali in rdeči zaboji so vidni. Qwen izrecno pove, da stik osebe z objektom ni jasen.

Florence trdi, da oseba nosi velik rdeč zaboj; na zamegljeni sliki to ni dovolj jasno potrjeno. NVILA r1 vrne zaporedna neveljavna JSON odseka in nima zanesljivega seznama, zato je vhod preskočen; celotni tekst ostane objavljen. Gibanje osebe ni dokaz o premičnosti nameščenih znakov ali regalov.

Florence OD lokalizira dve person regiji in številne box/crate regije. Dense opis oddaljene osebe navaja red carpeted floor, kar pri pregledu ni vidno potrjeno; tla delujejo pretežno betonska.
