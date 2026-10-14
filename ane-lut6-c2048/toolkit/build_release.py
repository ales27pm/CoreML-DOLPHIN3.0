#!/usr/bin/env python3
"""Package a verified export and its evidence; never include source weight shards."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

from download_source import assert_pinned_source_snapshot, verify_pinned_source_snapshot

ROOT = Path(__file__).resolve().parent
SMALL = "Dolphin3_ANE_Conversion_Toolkit"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_source_metadata_directory(directory, records):
    """Verify the exact copied metadata set against the pinned identity."""
    directory = Path(directory)
    expected = {item["file"] for item in records}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if len(expected) != len(records) or actual != expected:
        raise RuntimeError("Copied source metadata inventory differs from its verified identity")
    for item in records:
        copied = directory / item["file"]
        if copied.stat().st_size != item["bytes"] or digest(copied) != item["sha256"]:
            raise RuntimeError(f"Copied source metadata differs from its verified identity: {item['file']}")


def verify_archived_source_metadata(archive_path, prefix, records):
    """Re-hash the metadata bytes actually stored in one finished archive."""
    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist()]
        source_prefix = str(Path(prefix) / "source") + "/"
        expected_members = {source_prefix + item["file"] for item in records}
        actual_members = [name for name in names if name.startswith(source_prefix) and not name.endswith("/")]
        if len(actual_members) != len(expected_members) or set(actual_members) != expected_members:
            raise RuntimeError("Archived source metadata inventory differs from its verified identity")
        for item in records:
            member = source_prefix + item["file"]
            info = archive.getinfo(member)
            digest_value = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest_value.update(chunk)
            if info.file_size != item["bytes"] or digest_value.hexdigest() != item["sha256"]:
                raise RuntimeError(f"Archived source metadata differs from its verified identity: {member}")


def verify_archived_release_inputs(archive_path, prefix, verification, toolkit_only=False):
    """Bind the delivered ZIP bytes, not just its valid CRC, to verified inputs.

    Files can change after the pre-packaging check. Re-hash archived payloads
    against that check's inventory before emitting release_ready; never derive
    the expected hashes from the newly written archive itself.
    """
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Archived file inventory contains duplicate members")
        members = set(names)

        def check_files(root, records, exact=False):
            expected = {root + "/" + item["path"] for item in records}
            actual = {name for name in members if name.startswith(root + "/") and not name.endswith("/")}
            if len(expected) != len(records) or (exact and actual != expected):
                raise RuntimeError(f"Archived verified file inventory differs: {root}")
            for item in records:
                member = root + "/" + item["path"]
                if member not in members:
                    raise RuntimeError(f"Archived verified file is missing: {member}")
                info = archive.getinfo(member)
                value = hashlib.sha256()
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        value.update(chunk)
                if (("bytes" in item and info.file_size != item["bytes"])
                        or value.hexdigest() != item["sha256"]):
                    raise RuntimeError(f"Archived file differs from its verified identity: {member}")

        source_record = {"path": "source-verification.json",
                         "sha256": verification["pinned_source_verification"]["sha256"]}
        check_files(prefix if toolkit_only else prefix + "/toolkit", [source_record])
        if toolkit_only:
            return
        model_prefix = prefix + "/models"
        package_names = {part + ".mlpackage" for part in verification["packages"]}
        actual_packages = {name[len(model_prefix) + 1:].split("/", 1)[0] for name in members
                           if name.startswith(model_prefix + "/")
                           and name[len(model_prefix) + 1:].split("/", 1)[0].endswith(".mlpackage")}
        if actual_packages != package_names:
            raise RuntimeError("Archived model package set differs from verified inputs")
        for part, info in verification["packages"].items():
            check_files(model_prefix + "/" + part + ".mlpackage",
                        info["package_files_verified"], exact=True)
        check_files(model_prefix, verification["model_sidecar_files"])
        check_files(model_prefix + "/evidence/reference", verification["reference_files"])
        fixtures = verification["swift_fixtures"]
        check_files(model_prefix, [
            {"path": "manifest.json", "sha256": verification["manifest_sha256"]},
            {"path": fixtures["file"], "sha256": fixtures["sha256"]},
        ])
        check_files(model_prefix + "/evidence", [source_record])
        try:
            archived_verification = json.loads(archive.read(model_prefix + "/evidence/verification.json"))
        except (KeyError, ValueError) as error:
            raise RuntimeError("Archived verification report is missing or invalid") from error
        if archived_verification != verification:
            raise RuntimeError("Archived verification report differs from verified inputs")


def md(value):
    return str(value).replace("|", "\\|").replace("\n", "\\n")


def report(models, reference, verification):
    ref = read(reference / "reference-report.json")
    manifest = read(models / "manifest.json")
    bits = manifest["weights"]["index_bits"]
    context = manifest["context_length"]
    palette_size = 1 << bits
    weight_format = f"LUT{bits}"
    reused_source = bool(ref.get("source_reference_reuse"))
    baseline_note = ("La référence FP16 a été calculée pendant la première passe puis réutilisée pour éviter un second calcul identique. Les IDs de tokens, la configuration, le contexte et les empreintes de tous les paramètres source sont contrôlés avant comparaison. Le rapport JSON conserve la provenance des fichiers de référence réutilisés."
                     if reused_source else
                     "La référence FP16 et la référence compressée ont été calculées pendant cette passe à partir des mêmes poids source et des mêmes prompts.")
    source = read(ROOT / "source-verification.json")
    test_suites = ET.parse(models / "evidence/unit-tests.xml").getroot().findall(".//testsuite")
    tests = sum(int(s.attrib["tests"]) for s in test_suites)
    failures = sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in test_suites)
    if failures:
        raise RuntimeError("Unit-test evidence contains failures")
    parts = verification["packages"]
    lines = [
        "# Dolphin 3.0 Llama 3.2 3B — rapport de conversion", "",
        "**L’export Core ML stateful est généré et ses poids compressés sont vérifiés. La compilation Apple, l’inférence Core ML et l’exécution sur l’ANE restent à valider sur un appareil Apple.**", "",
        f"Rapport produit le {datetime.now(timezone.utc).isoformat()}.", "",
        "## Livrable", "",
        f"Source : [dphn/Dolphin3.0-Llama3.2-3B](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B/tree/{manifest['source_revision']}).", "",
        f"Révision figée : `{manifest['source_revision']}`.", "",
        "Le pipeline contient un embedding, sept blocs de quatre couches et une tête de vocabulaire. Les neuf fichiers sont de vrais packages ML Program. Les blocs possèdent des états KV FP16 persistants. La configuration Swift utilise `.cpuAndNeuralEngine`, qui permet encore le repli de certaines opérations sur le CPU.", "",
        "| Propriété | Valeur |", "| --- | --- |",
        "| Minimum système | iOS 18 / macOS 15 |",
        f"| Contexte fixe | {context:,} tokens consommés |",
        "| Taille d’une requête | 1 token |",
        "| Prefill | Token par token |",
        f"| Poids | {weight_format}, palettes FP16, groupes de 16 canaux de sortie |",
        "| Activations et cache KV | FP16 |",
        "| Vocabulaire | 128 258 tokens |",
        f"| Poids et spécifications des neuf packages | {verification['package_bytes']:,} octets ({verification['package_bytes']/1e9:.3f} Go décimaux) |",
        f"| Allocation des états KV | {verification['total_state_bytes']:,} octets / {verification['total_state_mib']:.0f} MiB |", "",
        "La taille des fichiers et celle des états ne sont pas une mesure de la mémoire de pointe d’une application iOS. Les buffers de travail, le chargement et la compilation ajoutent leurs propres coûts.", "",
        "## Inventaire des packages", "",
        f"| Package | Couches, indices de départ inclus et de fin exclus | Taille, Mo décimaux | Matrices {weight_format} vérifiées | États KV, MiB |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, item in parts.items():
        bounds = "—"
        if name.startswith("block_"):
            start = int(name[-2:]) * 4
            bounds = f"[{start}, {start+4})"
        lines.append(f"| {name}.mlpackage | {bounds} | {item['package_bytes']/1e6:.3f} | {item['compressed_weights_verified']} | {item['state_bytes']/1048576:.0f} |")
    lines += ["", "## Vérifications effectuées", "",
        f"- **{tests} tests Python réussis**, sans échec : comparaison avec l’implémentation Llama indépendante de Transformers, RoPE, caches, découpage, masquage causal, normalisation finale et derniers tokens du vocabulaire.",
        "- Les deux shards du checkpoint et les sept fichiers de métadonnées/tokenizer correspondent aux identités de la révision Hugging Face figée. Les preuves sont dans `toolkit/source-verification.json`.",
        f"- Les **{verification['compressed_tensor_count']} matrices compressées** sont présentes. Les indices sont réellement de type `uint{bits}`, avec des palettes scalaires de {palette_size} valeurs et des groupes de 16 canaux.",
        "- Les LUTs et les indices ont été relus depuis les packages enregistrés. Chaque matrice reconstruite correspond exactement, par SHA-256 des octets FP16, aux poids utilisés pour la référence numérique CPU.",
        "- Les inventaires, tailles et SHA-256 des fichiers de chaque package correspondent à l’export ; les interfaces et les états du protobuf sont vérifiés directement.",
        f"- Les buffers `keyCache` et `valueCache` mesurent `[4,1024,1,{context}]` dans chacun des sept blocs : 14 buffers dans sept objets `MLState`, soit {verification['total_state_mib']:.0f} MiB au total.",
        "- Les fichiers de référence contiennent les distributions complètes de 128 258 logits et des probes vérifiés pour trois prompts.", "",
        f"## Comparaison numérique avant et après {weight_format}", "",
        f"Ces résultats proviennent de l’architecture d’export exécutée en PyTorch FP16 sur CPU. La référence {weight_format} utilise la reconstruction exacte des poids des packages. Il ne s’agit pas d’une exécution de Core ML.", "",
        f"| Cas | Tokens du prompt | Prochain token FP16 | Prochain token {weight_format} | Même token | KL FP16→{weight_format}, nats | Écart maximal des logits | Top 10 communs |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    prompts = {x["id"]: x for x in ref["fixtures"]}
    baseline = {x["fixture_id"]: x for x in ref["results"]["source_fp16"]}
    quant = {x["fixture_id"]: x for x in ref["results"][f"lut{bits}_reference"]}
    for comp in ref[f"source_vs_lut{bits}"]:
        key = comp["fixture_id"]
        b, q = baseline[key], quant[key]
        lines.append(f"| {key} | {prompts[key]['prompt_token_count']} | {md(repr(b['greedy_token_text']))} ({b['greedy_token_id']}) | {md(repr(q['greedy_token_text']))} ({q['greedy_token_id']}) | {'Oui' if comp['greedy_token_equal'] else 'Non'} | {comp[f'kl_source_to_lut{bits}_nats']:.6f} | {comp['logit_max_absolute_difference']:.6f} | {comp['top_10_overlap_count']}/10 |")
    lines += ["", "Prompts examinés :", ""]
    for case in ref["fixtures"]:
        lines.append(f"- **{case['id']}** : {case['messages'][-1]['content']}")
    lines += ["",
        "Une KL plus petite indique des distributions plus proches. Le contrôle de livraison exige des logits finis et le même premier token que la référence sur les trois cas. Ce seuil est un contrôle de cohérence limité : ces probes ne mesurent ni la qualité des réponses complètes, ni la perplexité, ni le comportement à la capacité maximale du cache.", "",
        baseline_note, "",
        "## Détails du checkpoint conservés", "",
        "Les tokens 128256 (`<|im_end|>`) et 128257 (`<|im_start|>`) sont conservés. La tête expose sept sorties de 16 032 logits et une dernière de 16 034. Les lignes de padding nécessaires à la compression sont retirées des logits à l’intérieur du graphe.", "",
        "Les tables RoPE reproduisent exactement les équations Llama 3 de Transformers 4.47.1 avant leur stockage FP16. Le tokenizer et son template ChatML proviennent du checkpoint. L’embedding et la tête partagent les poids source liés, stockés dans deux packages distincts. La normalisation finale est appliquée une seule fois.", "",
        "La compression LUT6 utilise le solveur scalaire pondéré `kmeans1d` fourni avec Core ML Tools 9, sur les valeurs FP16 distinctes et leurs fréquences. Chaque groupe reçoit une palette de 64 valeurs FP16. Aucune calibration sur un corpus ni aucun entraînement supplémentaire n’ont été effectués.", "",
        "## Correction de la première stratégie de compression", "",
        "Une première stratégie LUT4 utilisant un clustering local initialisé par quantiles a été rejetée : elle prédisait le même mauvais token pour les trois prompts. Les vérifications ont exclu une corruption des indices, un alias de tenseurs et un défaut de réinitialisation des caches. Le diagnostic a montré que quelques poids rares de la projection MLP `down_proj` de la première couche étaient écrasés alors qu’ils multipliaient de fortes activations. La stratégie et la précision ont été changées, puis les comparaisons ont été reprises. Les anciens poids LUT4 ne sont pas inclus dans le livrable.", "",
        "## Ce qui reste à mesurer sur Apple", "",
        "| Contrôle | Statut |", "| --- | --- |",
        f"| Génération des spécifications Core ML et des poids {weight_format} | Réussie |",
        "| Intégrité des packages et égalité des poids reconstruits | Réussie |",
        "| Références numériques PyTorch sur les vrais poids | Réussies, contrôle limité aux trois prompts ci-dessus |",
        "| Compilation Swift avec le SDK Apple | Non exécutée |",
        "| Compilation/chargement des modèles sur Apple | Non exécutés |",
        "| Parité numérique Core ML avec les références | Non mesurée |",
        "| Placement ANE dans MLComputePlan | Non mesuré |",
        "| Exécution ANE réelle, latence, mémoire et énergie | Non mesurées |", "",
        "Le runtime Swift et son validateur sont fournis et relus. Le connecteur Mac était indisponible pendant cette session. Aucun débit en tokens/s sur iPhone, aucun taux de placement ANE et aucune compilation Apple réussie ne sont annoncés.", "",
        "Le fichier `models/evidencefixtures.json` propose une tolérance absolue de 1.0 sur les logits pour le premier contrôle appareil. Elle est explicitement provisoire et non calibrée. Le validateur distingue la parité numérique et le plan de calcul ; une trace Instruments reste nécessaire pour observer l’exécution réelle.", "",
        "## Commande de validation Apple", "",
        "Depuis la racine de l’archive complète décompressée :", "", "```sh", "cd toolkit/swift", "swift build -c release", "", 
        ".build/release/dolphin-ane-validate \\", "  --models ../../models \\", "  --fixtures ../../models/evidencefixtures.json \\", "  --strict-ane \\", "  --output device-validation.json", "```", "",
        "Le README explique aussi l’ajout des neuf modèles compilés, du manifeste et des tables RoPE aux ressources d’une application iOS. Le runtime reçoit des IDs de tokens ; l’intégration du tokenizer dans l’application reste à la charge de l’hôte.", "",
        "## Reproductibilité et sources", "",
        f"Passe de référence CPU : PyTorch {ref['torch_version']}, {ref['threads']} threads de calcul et {ref.get('palette_workers', 1)} workers pour les palettes, durée totale {ref['total_seconds']:.1f} s. Cette durée inclut le chargement, le hashing et la compression ; elle ne représente pas la vitesse de génération." + (" Le calcul FP16 préalable réutilisé n’est pas inclus dans cette durée." if reused_source else ""), "",
        "Les logs, rapports par package, tests JUnit, références complètes et empreintes sont conservés dans `models/evidence/`. Les versions Python sont figées dans le toolkit. Aucun modèle n’a été publié sur Hugging Face ou GitHub par cette conversion.", "",
        "- [Apple — Stateful models, disponibles depuis iOS 18/macOS 15](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)",
        "- [Apple — Palettization et groupes de canaux](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)",
        "- [Transformers 4.47.1 — RoPE](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/modeling_rope_utils.py)",
        "- [Dolphin — modèle source](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B)", "",
        "Les poids conservent la licence communautaire Llama 3.2 du modèle source. Le model card, la licence et la politique d’utilisation sont fournis. Built with Llama.", "",
    ]
    return "\n".join(lines)


def verify_current_inputs(models, reference, verification):
    """Bind the archive to the exact files which passed independent verification."""
    if digest(models / "manifest.json") != verification["manifest_sha256"]:
        raise RuntimeError("Manifest changed after verification")
    pinned = verification.get("pinned_source_verification", {})
    if (pinned.get("sha256") != digest(ROOT / "source-verification.json")
            or pinned.get("source_weight_identity") != verification.get("source_weight_identity")):
        raise RuntimeError("Pinned source verification changed after independent verification")

    def check_files(root, records, exact=False):
        expected = {item["path"] for item in records}
        if len(expected) != len(records):
            raise ValueError("Duplicate paths in verified file inventory")
        if exact and expected != {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}:
            raise RuntimeError(f"Verified file inventory changed: {root}")
        for item in records:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise RuntimeError(f"Missing or escaped verified file: {path}")
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise RuntimeError(f"File changed after verification: {path}")

    package_names = {part + ".mlpackage" for part in verification["packages"]}
    if package_names != {path.name for path in models.glob("*.mlpackage")}:
        raise RuntimeError("Model package set changed after verification")
    for part, info in verification["packages"].items():
        check_files(models / (part + ".mlpackage"), info["package_files_verified"], exact=True)
    check_files(reference, verification["reference_files"])
    check_files(models, verification["model_sidecar_files"])
    fixtures = verification["swift_fixtures"]
    if digest(models / fixtures["file"]) != fixtures["sha256"]:
        raise RuntimeError("Swift reference fixtures changed after verification")


def verify_release_source_binding(models, reference, verification, source_dir=None,
                                  source_verification=None):
    """Reject legacy/mixed evidence before any release directory is created."""
    source_dir = (ROOT / "source") if source_dir is None else Path(source_dir)
    source_verification = (ROOT / "source-verification.json" if source_verification is None
                           else Path(source_verification))
    manifest = read(models / "manifest.json")
    ref = read(reference / "reference-report.json")
    ref_hashes = read(reference / "reference-weight-hashes.json")
    if verification.get("schema_version") != 3:
        raise RuntimeError("Unsupported verification schema; expected schema 3 with cryptographic source-input binding")
    if manifest.get("format_version") != 3:
        raise RuntimeError("Unsupported manifest format; expected format 3 with cryptographic source-input binding")
    if ref.get("schema_version") != 3 or ref_hashes.get("schema_version") != 3:
        raise RuntimeError("Unsupported reference evidence schema; expected schema 3 after regeneration or migration")
    source_identity, source_snapshot = verify_pinned_source_snapshot(source_dir, source_verification)
    identities = (manifest.get("source_weight_identity"),
                  ref.get("source_identity", {}).get("source_weight_identity"),
                  ref_hashes.get("source_weight_identity"), verification.get("source_weight_identity"))
    if any(value != source_identity for value in identities):
        raise RuntimeError("Release inputs do not share the freshly re-hashed pinned source identity")
    return manifest, ref, source_snapshot


def add_tree(archive, directory, prefix, exclude=()):
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and not any(relative == Path(item) or relative.is_relative_to(item) for item in exclude):
            archive.write(path, str(Path(prefix) / relative))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    a = p.parse_args()
    models, reference, destination = a.models.resolve(), a.reference.resolve(), a.destination.resolve()
    evidence = read(models / "evidence/verification.json")
    if not evidence.get("artifact_integrity_and_weight_parity_passed") or not evidence.get("serialized_lut_reconstruction_verified"):
        raise RuntimeError("A complete, successful serialized-weight verification is required")
    manifest, ref, source_snapshot = verify_release_source_binding(models, reference, evidence)
    bits = manifest["weights"]["index_bits"]
    comparisons = ref.get(f"source_vs_lut{bits}")
    if bits != 6:
        raise RuntimeError("This release recipe accepts only the corrected LUT6 conversion")
    if not ref.get("numerical_quality_gate_passed") or not evidence.get("numerical_quality_gate_passed"):
        raise RuntimeError("A successful numerical sanity gate is required before packaging")
    if not comparisons or len(comparisons) != 3 or not all(item["greedy_token_equal"] for item in comparisons):
        raise RuntimeError("All three intended probes must retain the source greedy token")
    verify_current_inputs(models, reference, evidence)
    release = f"Dolphin3_ANE_Stateful_LUT{bits}_C{manifest['context_length']}"
    destination.mkdir(parents=True, exist_ok=True)
    staged = destination / "toolkit-staging"
    if staged.exists():
        raise FileExistsError("Refusing to overwrite a prior staging directory")
    staged.mkdir()
    for name in ("dolphin_ane", "tests", "swift", "licenses"):
        shutil.copytree(ROOT / name, staged / name, ignore=shutil.ignore_patterns("__pycache__", ".build", ".DS_Store"))
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, staged / path.name)
    for name in ("README.md", "requirements.txt", "environment-lock.txt", "source-verification.json"):
        shutil.copy2(ROOT / name, staged / name)
    metadata_names = ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md")
    (staged / "source").mkdir()
    for name in metadata_names:
        shutil.copy2(ROOT / "source" / name, staged / "source" / name)
    metadata_records = evidence["source_weight_identity"].get("metadata")
    if (not isinstance(metadata_records, list) or len(metadata_records) != len(metadata_names)
            or {item.get("file") for item in metadata_records} != set(metadata_names)):
        raise RuntimeError("Verified source identity has an invalid metadata inventory")
    verify_source_metadata_directory(staged / "source", metadata_records)
    assert_pinned_source_snapshot(ROOT / "source", source_snapshot, ROOT / "source-verification.json")
    report_path = destination / "Dolphin3_ANE_Conversion_Report.md"
    report_path.write_text(report(models, reference, evidence), encoding="utf-8")
    shutil.copy2(report_path, staged / "CONVERSION_REPORT.md")
    full_zip = destination / (release + ".zip")
    small_zip = destination / (SMALL + ".zip")
    for archive_path in (full_zip, small_zip):
        if archive_path.exists():
            raise FileExistsError(archive_path)
    with zipfile.ZipFile(full_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        # Add reference evidence directly. Packaging never mutates verified input
        # directories, so a failed archive can be rebuilt into a fresh destination.
        add_tree(z, models, release + "/models", exclude=("evidence/reference", "evidence/source-verification.json"))
        add_tree(z, reference, release + "/models/evidence/reference")
        z.write(ROOT / "source-verification.json", release + "/models/evidence/source-verification.json")
        add_tree(z, staged, release + "/toolkit")
        z.write(ROOT / "README.md", release + "/README.md")
        z.write(report_path, release + "/CONVERSION_REPORT.md")
    with zipfile.ZipFile(small_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        add_tree(z, staged, SMALL)
    for archive_path in (full_zip, small_zip):
        with zipfile.ZipFile(archive_path) as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError(f"Archive CRC failure: {archive_path}: {bad}")
        archive_prefix = release + "/toolkit" if archive_path == full_zip else SMALL
        verify_archived_source_metadata(archive_path, archive_prefix, metadata_records)
        verify_archived_release_inputs(archive_path, release if archive_path == full_zip else SMALL,
                                       evidence, toolkit_only=archive_path == small_zip)
        print(json.dumps({"file": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": digest(archive_path)}), flush=True)
    verify_source_metadata_directory(staged / "source", metadata_records)
    assert_pinned_source_snapshot(ROOT / "source", source_snapshot, ROOT / "source-verification.json")
    records = [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in (full_zip, small_zip, report_path)]
    (destination / "release-files.json").write_text(json.dumps(records, indent=2) + "\n")
    (destination / "SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in records))
    print(json.dumps({"release_ready": True, "files": records}), flush=True)


if __name__ == "__main__":
    main()
