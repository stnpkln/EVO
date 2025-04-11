# PROJEKT EVO

## Fitness funkce
- MSE pro filtr
- Klasifikator - pocet chybne klasifikovanych

## Funkční sada
- pro zanechání původních hodnot filtru se nejspíš hodí podmíněné přiřazení a identita
- např.: y if (x > 127) else x conditional assignment
- Celkově bych použil primárně funkce z článku ze zadání

## Typy šumu
- Strukturované šumy: Diagonální čáry, Vertikální čáry s periodou (Něco ve stylu - - - - -), Nějaký obecný periodický? (třeba == == == ).
- salt and pepper (na porovnání)

## Implementace
- Python a HAL CGP

## Testování
- nejspíš na notebooku (obrázky 128x128), ale kdyby to nešlo tak zažádat metacentrum

## Dotazy
- není mi úplně jasné jak specifický má být filtr? Například pokud budu mít nějaký ten periodický šum, měl by jeden filtr fungovat na různé periody, nebo ho mám naučit pouze na jednu?
- Jak přesně mám porovnávat se salt and pepper?
- jak velká by měla být testovací sada?

- trenovat zvlast klasifikator a filtr
- zkouset ruzna okoli
- Univerzalni i dedikovane filtry
- klidne nespojite jadra (treba hlavni pixel, pak o pet vedle apod) - muze se hodit na ty periodicke