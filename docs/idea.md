Idea 3 (estesa): Negoziazione con asimmetria informativa + Social Learning testuale
Contesto di base
Scenario:

Agente Seller (Venditore di un software).

Sa che il prodotto ha un bug critico.

Vuole massimizzare il prezzo e ha istruzioni a non rivelare spontaneamente il bug.

Agente Buyer (Compratore).

Ha un budget massimo (es. 10.000 €).

Il suo prompt dice che deve cercare di scoprire eventuali problemi e abbassare il prezzo se emergono.

Compito: farli negoziare sul prezzo tramite dialogo naturale, con asimmetria informativa (solo il venditore conosce il bug) e obiettivi contrastanti (massimo profitto vs prezzo minimo/sicurezza).

Obiettivo del progetto:

Analizzare strategie emergenti (persuasione, bluff, omissione, concessioni)

Valutare se queste strategie sembrano frutto di ragionamento genuino, di adattamento pragmatico al contesto, o semplice ripetizione di pattern visti nei prompt (scripted imitation).

Il tutto in solo testo (nessuna multimodalità, niente ambienti 3D), quindi compatibile con i tuoi vincoli computazionali.

Fase 1 – Baseline: Zero-shot Negotiation
Imposti il setup minimo:

Definisci chiaramente prompt di sistema per Seller e Buyer (ruolo, obiettivi, conoscenze private).

Fai negoziare 10–20 partite Buyer–Seller con lo stesso modello (es. GPT-4o-mini vs GPT-4o-mini) e con modelli diversi (es. GPT vs LLaMA) per vedere differenze di stile.

Logghi:

L’intero dialogo

(Opzionale ma fortissimo) Un Chain-of-Thought privato per ogni mossa, che non viene inviato all’altro agente, ma solo salvato a scopo di analisi offline.

Da qui ottieni:

Distribuzione dei prezzi finali

Frequenza con cui il bug viene rivelato / scoperto

Lunghezza e tono delle negoziazioni

Questa Fase 1 ti serve sia come baseline numerica, sia come “materiale grezzo” da cui far emergere norme/tattiche nelle fasi successive.

Fase 2 – Socialized Learning (testuale) + Behavioral Cloning in-context
Qui importi, in forma leggera, le idee di Social Learning e Behavioral Cloning dai paper multi‑agente, ma usando solo testo e prompt.

2.1. Generazione di una “coorte”
Lanci K negoziazioni in parallelo (es. 20 partite Seller–Buyer).

Introduci un terzo LLM come Observer/Arbitro:

Legge ogni partita.

Assegna uno score a Buyer e Seller (es. “utile per Buyer”, “utile per Seller”, “equilibrato”, “fallimento totale”).

2.2. Selezione delle partite “di successo”
Per ciascun ruolo:

Buyer di successo: chi riesce a scoprire il bug e/o ottenere un forte sconto.

Seller di successo: chi riesce a:

chiudere a prezzo alto, oppure

ammettere il bug in modo strategico ma tenendo un prezzo comunque alto,

evitando rottura della trattativa.

L’Observer seleziona, per ogni ruolo, le 1–2 partite migliori, e ne estrae:

Una descrizione astratta della tattica (in linguaggio naturale: “prima builda fiducia, poi fa domande mirate, poi usa il bug per ottenere sconto”).

Alcuni snippet di dialogo come esempi.

2.3. Behavioral Cloning come Prompting
Invece di aggiornare i pesi, fai Behavioral Cloning in-context:

Aggiorni il system prompt del Buyer successivo con qualcosa tipo:

“You are a buyer. Previous successful buyers learned the following strategy:

Start by building rapport and asking open-ended questions about reliability and security.

Then ask highly specific questions about past incidents and vulnerabilities.

If the seller hesitates or is vague, push for a discount citing potential hidden risks.
Here is an example of a particularly successful negotiation:
[SNIPPET DI DIALOGO]
Now negotiate according to this strategy while still adapting to the specific context.”

Fai lo stesso per il Seller, ma con tattiche di gestione della disclosure, framing positivo, spostamento del focus su feature e supporto.

Questo è esattamente l’uso “soft” di social learning + cloning:

gli agenti futuri apprendono socialmente dai “pari di successo” (descrizione tattica)

e fanno cloning comportamentale in-context (via few-shot + regole in prompt).

Quando e come valutare
“Genuine reasoning” vs “Pragmatic adaptation” vs “Scripted imitation”
Questa è la parte chiave per il tuo Research Question. Puoi strutturarla così.

1. Scripted Imitation (copione rigido)
Cosa cerchi: il modello sta solo ripetendo pattern dal prompt/esempi, senza vera flessibilità.

Come misurarlo/prenderlo evidenza:

Analisi di similarità:

Misuri quanto le nuove negoziazioni, dopo il cloning, sono vicine (es. tramite cosine similarity di embedding sentence-level) ai dialoghi di esempio forniti nel prompt.

Se molte nuove partite sono quasi copie parola-per-parola, o cambiano solo numeri e nomi, è segno forte di imitazione scriptata.

Test Out-of-Distribution:

Cambi leggermente il contesto (software diverso, bug diverso, range di prezzi) ma mantieni lo stesso prompt di tattica.

Se l’agente continua a usare formulazioni quasi identiche e non adatta la strategia ai nuovi numeri/contesto, stai vedendo più copione che ragionamento.

2. Pragmatic Adaptation (adattamento strategico al contesto)
Cosa cerchi: il modello applica la tattica, ma la modula in base allo stato della trattativa.

Segnali e metriche:

Dipendenza dal contesto:

Stessa tattica di base, ma:

se il Seller appare molto cooperativo, il Buyer chiude prima,

se il Seller è difensivo, il Buyer intensifica domande o minacce velate.

Qui puoi usare un LLM‑judge che, partita per partita, annota se le mosse sono “appropriate given the dialogue so far”.

Variazione controllata di strategia:

Introduci perturbazioni (es. “il Buyer ha pochissimo tempo”, “budget più basso”, “il Seller offre inizialmente un prezzo molto buono”).

Valuti se l’agente cambia tattica in modo coerente con il nuovo contesto o segue il copione invariato (il che indicherebbe scripted imitation).

3. Genuine Reasoning (ragionamento “genuino”)
Qui sei onesto: non puoi provare in assoluto che il modello “pensi”, ma puoi cercare indizi strutturali di ragionamento.

Evidenze possibili:

Catene di pensiero coerenti ma non triviali (dal CoT privato):

Nel CoT il Buyer valuta trade-off:

“Se spingo troppo sul bug, rischiamo breakdown → meglio prima ottenere riconoscimento implicito, poi usare come leva per sconto moderato.”

Vedi uso esplicito di condizioni del tipo “Se… allora… altrimenti…”, e riferimenti a stati precedenti della conversazione (ragionamento condizionale dipendente dallo storico).

Generazione di tattiche nuove rispetto al prompt:

Dopo 2–3 cicli di Social Learning, chiedi a un LLM‑judge:

“Descrivi la strategia usata in questa partita in 3 punti. È diversa dalle strategie fornite negli esempi?”

Se emergono pattern strategici non esplicitamente descritti nel prompt originario (es. uso di minacce credibili, escalation/de‑escalation, tattiche di time pressure), è un segno che il modello sta combinando informazioni e non solo copiando.

Coerenza cross‑episodio:

Confronti partite diverse con condizioni simili: se il modello mantiene una linea strategica coerente ma con variazioni contestuali sensate, hai un pattern che ricorda policy + planning più che mero recall verbatim, in linea con quanto osservato in piattaforme tipo NegotiationArena.

Nel report puoi esplicitamente dedicare una sottosezione tipo:

“Distinguishing Reasoning from Imitation: Operational Criteria”
– e lì elencare i test sopra come operationalization di quelle tre categorie.


