# Dolphin 3.0 Llama 3.2 3B — rapport de conversion

**L’export Core ML stateful est généré et ses poids compressés sont vérifiés. La compilation Apple, l’inférence Core ML et l’exécution sur l’ANE restent à valider sur un appareil Apple.**

Rapport produit le 2026-10-13T00:08:23.144158+00:00.

## Livrable

Source : [dphn/Dolphin3.0-Llama3.2-3B](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B/tree/392a6f57223e7ccfe6ef4ebdb2ff101a42d57364).

Révision figée : `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.

Le pipeline contient un embedding, sept blocs de quatre couches et une tête de vocabulaire. Les neuf fichiers sont de vrais packages ML Program. Les blocs possèdent des états KV FP16 persistants. La configuration Swift utilise `.cpuAndNeuralEngine`, qui permet encore le repli de certaines opérations sur le CPU.

| Propriété | Valeur |
| --- | --- |
| Minimum système | iOS 18 / macOS 15 |
| Contexte fixe | 2,048 tokens consommés |
| Taille d’une requête | 1 token |
| Prefill | Token par token |
| Poids | LUT6, palettes FP16, groupes de 16 canaux de sortie |
| Activations et cache KV | FP16 |
| Vocabulaire | 128 258 tokens |
| Poids et spécifications des neuf packages | 2,718,527,235 octets (2.719 Go décimaux) |
| Allocation des états KV | 234,881,024 octets / 224 MiB |

La taille des fichiers et celle des états ne sont pas une mesure de la mémoire de pointe d’une application iOS. Les buffers de travail, le chargement et la compilation ajoutent leurs propres coûts.

## Inventaire des packages

| Package | Couches, indices de départ inclus et de fin exclus | Taille, Mo décimaux | Matrices LUT6 vérifiées | États KV, MiB |
| --- | --- | ---: | ---: | ---: |
| embedding.mlpackage | — | 296.569 | 1 | 0 |
| block_00.mlpackage | [0, 4) | 303.624 | 28 | 32 |
| block_01.mlpackage | [4, 8) | 303.624 | 28 | 32 |
| block_02.mlpackage | [8, 12) | 303.624 | 28 | 32 |
| block_03.mlpackage | [12, 16) | 303.624 | 28 | 32 |
| block_04.mlpackage | [16, 20) | 303.624 | 28 | 32 |
| block_05.mlpackage | [20, 24) | 303.624 | 28 | 32 |
| block_06.mlpackage | [24, 28) | 303.624 | 28 | 32 |
| head.mlpackage | — | 296.587 | 8 | 0 |

## Vérifications effectuées

- **84 tests Python réussis**, sans échec : comparaison avec l’implémentation Llama indépendante de Transformers, RoPE, caches, découpage, masquage causal, normalisation finale et derniers tokens du vocabulaire.
- Les deux shards du checkpoint et les sept fichiers de métadonnées/tokenizer correspondent aux identités de la révision Hugging Face figée. Les preuves sont dans `toolkit/source-verification.json`.
- Les **205 matrices compressées** sont présentes. Les indices sont réellement de type `uint6`, avec des palettes scalaires de 64 valeurs et des groupes de 16 canaux.
- Les LUTs et les indices ont été relus depuis les packages enregistrés. Chaque matrice reconstruite correspond exactement, par SHA-256 des octets FP16, aux poids utilisés pour la référence numérique CPU.
- Les inventaires, tailles et SHA-256 des fichiers de chaque package correspondent à l’export ; les interfaces et les états du protobuf sont vérifiés directement.
- Les buffers `keyCache` et `valueCache` mesurent `[4,1024,1,2048]` dans chacun des sept blocs : 14 buffers dans sept objets `MLState`, soit 224 MiB au total.
- Les fichiers de référence contiennent les distributions complètes de 128 258 logits et des probes vérifiés pour trois prompts.

## Comparaison numérique avant et après LUT6

Ces résultats proviennent de l’architecture d’export exécutée en PyTorch FP16 sur CPU. La référence LUT6 utilise la reconstruction exacte des poids des packages. Il ne s’agit pas d’une exécution de Core ML.

| Cas | Tokens du prompt | Prochain token FP16 | Prochain token LUT6 | Même token | KL FP16→LUT6, nats | Écart maximal des logits | Top 10 communs |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| greeting_en | 35 | 'Hello' (9906) | 'Hello' (9906) | Oui | 0.003005 | 1.066406 | 10/10 |
| arithmetic | 42 | '4' (19) | '4' (19) | Oui | 0.009955 | 1.906250 | 9/10 |
| greeting_fr | 39 | 'Bonjour' (82681) | 'Bonjour' (82681) | Oui | 0.014317 | 1.239258 | 10/10 |

Prompts examinés :

- **greeting_en** : Say hello in one short sentence.
- **arithmetic** : What is 2 + 2? Answer with the number only.
- **greeting_fr** : Dis bonjour en français, en une courte phrase.

Une KL plus petite indique des distributions plus proches. Le contrôle de livraison exige des logits finis et le même premier token que la référence sur les trois cas. Ce seuil est un contrôle de cohérence limité : ces probes ne mesurent ni la qualité des réponses complètes, ni la perplexité, ni le comportement à la capacité maximale du cache.

La référence FP16 a été calculée pendant la première passe puis réutilisée pour éviter un second calcul identique. Les IDs de tokens, la configuration, le contexte et les empreintes de tous les paramètres source sont contrôlés avant comparaison. Le rapport JSON conserve la provenance des fichiers de référence réutilisés.

## Détails du checkpoint conservés

Les tokens 128256 (`<|im_end|>`) et 128257 (`<|im_start|>`) sont conservés. La tête expose sept sorties de 16 032 logits et une dernière de 16 034. Les lignes de padding nécessaires à la compression sont retirées des logits à l’intérieur du graphe.

Les tables RoPE reproduisent exactement les équations Llama 3 de Transformers 4.47.1 avant leur stockage FP16. Le tokenizer et son template ChatML proviennent du checkpoint. L’embedding et la tête partagent les poids source liés, stockés dans deux packages distincts. La normalisation finale est appliquée une seule fois.

La compression LUT6 utilise le solveur scalaire pondéré `kmeans1d` fourni avec Core ML Tools 9, sur les valeurs FP16 distinctes et leurs fréquences. Chaque groupe reçoit une palette de 64 valeurs FP16. Aucune calibration sur un corpus ni aucun entraînement supplémentaire n’ont été effectués.

## Correction de la première stratégie de compression

Une première stratégie LUT4 utilisant un clustering local initialisé par quantiles a été rejetée : elle prédisait le même mauvais token pour les trois prompts. Les vérifications ont exclu une corruption des indices, un alias de tenseurs et un défaut de réinitialisation des caches. Le diagnostic a montré que quelques poids rares de la projection MLP `down_proj` de la première couche étaient écrasés alors qu’ils multipliaient de fortes activations. La stratégie et la précision ont été changées, puis les comparaisons ont été reprises. Les anciens poids LUT4 ne sont pas inclus dans le livrable.

## Ce qui reste à mesurer sur Apple

| Contrôle | Statut |
| --- | --- |
| Génération des spécifications Core ML et des poids LUT6 | Réussie |
| Intégrité des packages et égalité des poids reconstruits | Réussie |
| Références numériques PyTorch sur les vrais poids | Réussies, contrôle limité aux trois prompts ci-dessus |
| Compilation Swift avec le SDK Apple | Non exécutée |
| Compilation/chargement des modèles sur Apple | Non exécutés |
| Parité numérique Core ML avec les références | Non mesurée |
| Placement ANE dans MLComputePlan | Non mesuré |
| Exécution ANE réelle, latence, mémoire et énergie | Non mesurées |

Le runtime Swift et son validateur sont fournis et relus. Le connecteur Mac était indisponible pendant cette session. Aucun débit en tokens/s sur iPhone, aucun taux de placement ANE et aucune compilation Apple réussie ne sont annoncés.

Le fichier `models/evidencefixtures.json` propose une tolérance absolue de 1.0 sur les logits pour le premier contrôle appareil. Elle est explicitement provisoire et non calibrée. Le validateur distingue la parité numérique et le plan de calcul ; une trace Instruments reste nécessaire pour observer l’exécution réelle.

## Commande de validation Apple

Depuis la racine de l’archive complète décompressée :

```sh
cd toolkit/swift
swift build -c release

.build/release/dolphin-ane-validate \
  --models ../../models \
  --fixtures ../../models/evidencefixtures.json \
  --strict-ane \
  --output device-validation.json
```

Le README explique aussi l’ajout des neuf modèles compilés, du manifeste et des tables RoPE aux ressources d’une application iOS. Le runtime reçoit des IDs de tokens ; l’intégration du tokenizer dans l’application reste à la charge de l’hôte.

## Reproductibilité et sources

Passe de référence CPU : PyTorch 2.5.0+cpu, 4 threads de calcul et 4 workers pour les palettes, durée totale 1048.4 s. Cette durée inclut le chargement, le hashing et la compression ; elle ne représente pas la vitesse de génération. Le calcul FP16 préalable réutilisé n’est pas inclus dans cette durée.

Les logs, rapports par package, tests JUnit, références complètes et empreintes accompagnent les modèles dans `models/evidence/`. La copie GitHub fournit les petites preuves numériques dans `ane-lut6-c2048/evidence/`, sans poids ni logs volumineux. Les versions Python sont figées dans le toolkit. Cette édition prépare la publication du code dans `ales27pm/CoreML-DOLPHIN3.0` et des modèles dans `ales27pm/Dolphin3.0-CoreML`, sous le préfixe `ane-lut6-c2048/`. La disponibilité des téléchargements dépend du commit de publication.

- [Apple — Stateful models, disponibles depuis iOS 18/macOS 15](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Apple — Palettization et groupes de canaux](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
- [Transformers 4.47.1 — RoPE](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/modeling_rope_utils.py)
- [Dolphin — modèle source](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B)

Les poids conservent la licence communautaire Llama 3.2 du modèle source. Le model card, la licence et la politique d’utilisation sont fournis. Built with Llama.
