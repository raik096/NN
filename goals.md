- [x] regolarizzazione
    - [ ] tykhonov L2

- [x] minibatch

- [x] plot_utils
    - [ ] plot finale accuracy, precision, recall per problemi di classificazione e plot finale caso regressione 
    - [ ] plot grid search su ciò che ci interessa sui parametri in esplorazione
    - [ ] plot k - fold
    - [ ] plot test set 

- [/] gestione test set (merge necessario)

- [x] grid search k-fold (merge necessario)
 
- [ ] nested grid search 

- [ ] random grid search 

- [ ] implementare altre strategie di eta variabile

- [x] capire funzione di attivazione output layer

- [x] merge
    - [ ] resa stabile branch IOROBOT
    - [ ] forward chiamabile dall'esterno (test set)
    - [ ] grid search flag in trainer, cambia comportamento in base a se è l'ultimo train
    - [ ] prendere mee da branch versioneLeo
    - [ ] opzionale, fissare parametri automaticamento in grid search in base ad altri parametri esclusi, tipo decay_fault implica decay factor non da esplorare


- [ ] da vedere
    - bootstrap
    - cascay correlation


- [domani]

    - denormalizzazione dati plot cup
    - migliorare il layout dati plot cup top 3
    - aggiungere mee 
    - tabella grid search tr vl ts
    - runnare il final model dei top 3 con plot della learning curve denormalizzati di tr vl ts