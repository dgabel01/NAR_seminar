Zadatak:

- Primijeniti Mean Filter 7x7 filter za sliku "Togir.jpg"

Mean filter 7x7 vrsta je linearnog filtra koji se koristi u obradi slike za izglađivanje slika i smanjenje šuma. 
Mean filter radi tako što izračunava prosjek vrijednosti piksela unutar lokalnog susjedstva definiranog veličinom filtra.
U ovom slučau, susjedstvo je mreža 7x7 centrirana oko svakog piksela na slici.


Princip rada:

- Veličina jezgre: Filtar koristi jezgru 7x7, što znači da uzima u obzir kvadrat 7x7 piksela oko svakog piksela na slici.
- Usrednjavanje: Za svaki piksel na slici, filtar izračunava prosječnu vrijednost piksela i njegovih 48 susjeda (budući da je 7x7 = 49).
- Zamijeni vrijednost piksela: središnja vrijednost piksela se zatim zamjenjuje ovom prosječnom vrijednošću.
- Pomicanje po slici: filtar se pomiče po slici, ponavljajući ovaj postupak za svaki piksel (osim za granične piksele, gdje je potrebno posebno rukovanje).
