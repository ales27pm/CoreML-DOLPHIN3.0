from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILE = ROOT / "Sources" / "App" / "Bench" / "BenchmarkCSVSupport.swift"


def run_swift_script(source: str, *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="swift-script-") as tmpdir:
        script_path = Path(tmpdir) / "script.swift"
        raw_lines = source.strip("\n").splitlines()
        import_lines: list[str] = []
        body_lines: list[str] = []
        body_started = False
        for line in raw_lines:
            stripped = line.strip()
            if not body_started and stripped.startswith("import "):
                import_lines.append(stripped)
                continue
            body_started = True
            body_lines.append(line)

        script_parts: list[str] = []
        if import_lines:
            script_parts.extend(import_lines)
            script_parts.append("")
        script_parts.append("@main")
        script_parts.append("struct Harness {")
        script_parts.append("    static func main() throws {")
        if not body_lines:
            script_parts.append("        return")
        else:
            for line in body_lines:
                if line.strip():
                    script_parts.append(f"        {line}")
                else:
                    script_parts.append("")
        script_parts.append("    }")
        script_parts.append("}")
        script_parts.append("")
        script_path.write_text("\n".join(script_parts), encoding="utf-8")
        binary_path = Path(tmpdir) / "runner"
        compile_cmd = [
            "swiftc",
            str(SUPPORT_FILE),
            str(script_path),
            "-o",
            str(binary_path),
        ]
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        command = [str(binary_path)]
        command.extend(args)
        return subprocess.run(command, check=True, capture_output=True, text=True)


def test_csv_render_is_deterministic() -> None:
    script = textwrap.dedent(
        """
        import Foundation

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        let timestamp = formatter.date(from: "2025-10-26T12:34:56.789Z")!

        let row = BenchmarkCSVRow(
            timestamp: timestamp,
            tokensPerSecond: 42.75,
            initMilliseconds: 120.5,
            decodeMillisecondsPerToken: 23.5,
            embedMilliseconds: 87.0
        )
        let writer = BenchmarkCSVWriter()
        let csv = writer.render(samples: [row])
        print(csv, terminator: "")
        """
    )
    result = run_swift_script(script)
    expected = "\n".join(
        [
            "timestamp,tokens_per_second,init_ms,decode_ms_per_token,embed_ms",
            "2025-10-26T12:34:56.789Z,42.75,120.50,23.50,87.00",
            "",
        ]
    )
    assert result.stdout == expected


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "artifacts" / "bench" / "result.csv"
    script = textwrap.dedent(
        """
        import Foundation

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        let timestamp = formatter.date(from: "2025-10-26T00:00:00Z")!

        let rows = [
            BenchmarkCSVRow(
                timestamp: timestamp,
                tokensPerSecond: 40.0,
                initMilliseconds: 111.0,
                decodeMillisecondsPerToken: 25.0,
                embedMilliseconds: 90.0
            ),
        ]
        let writer = BenchmarkCSVWriter()
        guard let argument = CommandLine.arguments.last else {
            fatalError("Expected output path argument")
        }
        let destination = URL(fileURLWithPath: argument)
        try writer.write(samples: rows, to: destination)
        print(destination.path)
        """
    )
    result = run_swift_script(script, str(output_path))
    assert result.stdout.strip() == str(output_path)
    contents = output_path.read_text(encoding="utf-8")
    assert "timestamp,tokens_per_second" in contents
    assert "40.00" in contents


def test_validation_rejects_low_throughput() -> None:
    script = textwrap.dedent(
        """
        import Foundation

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        let timestamp = formatter.date(from: "2025-10-26T01:02:03Z")!

        let rows = [
            BenchmarkCSVRow(
                timestamp: timestamp,
                tokensPerSecond: 30.0,
                initMilliseconds: 100.0,
                decodeMillisecondsPerToken: 33.3,
                embedMilliseconds: 85.0
            ),
            BenchmarkCSVRow(
                timestamp: timestamp.addingTimeInterval(1.0),
                tokensPerSecond: 35.0,
                initMilliseconds: 100.0,
                decodeMillisecondsPerToken: 28.6,
                embedMilliseconds: 85.0
            ),
        ]

        do {
            _ = try ThroughputRegressor.validate(rows, minimumRate: 33.0)
            print("no-error")
        } catch {
            print(error.localizedDescription)
        }
        """
    )
    result = run_swift_script(script)
    assert result.stdout.strip() == "Iteration 0 produced 30.00 tok/s below the minimum of 33.0."
