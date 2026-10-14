"""Real serialized LUT4/LUT6 inspection must not modify package metadata."""

import json
from pathlib import Path
import shutil

from test_exporter import exported_block
import verify_export


def test_saved_lut_reconstruction_is_read_only_and_matches_exporter(exported_block, tmp_path):
    args, _ = exported_block
    # Other fixture tests legitimately instantiate MLModel. Inspect a private
    # copy so their metadata rewrites cannot influence this read-only contract.
    package = tmp_path / "block.mlpackage"
    shutil.copytree(args.output, package)
    expected = json.loads(Path(str(args.output) + ".json").read_text())["weights"]
    before = verify_export.capture_evidence_inventory(package)

    actual = verify_export.reconstruct_serialized_luts(package, expected_bits=args.bits)

    assert verify_export.capture_evidence_inventory(package) == before
    verify_export.assert_package_inventories(tmp_path, {package.name: before})
    expected_by_name = {item["parameter_name"]: item for item in expected}
    actual_by_name = {item["parameter_name"]: item for item in actual}
    assert len(actual_by_name) == len(actual) == len(expected) == 28
    assert actual_by_name.keys() == expected_by_name.keys()
    for name, record in actual_by_name.items():
        for field in ("shape", "index_dtype", "index_bits", "lut_shape", "dense_fp16_sha256"):
            assert record[field] == expected_by_name[name][field], (name, field)
