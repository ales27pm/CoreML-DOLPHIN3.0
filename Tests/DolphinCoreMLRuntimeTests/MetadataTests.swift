import XCTest
@testable import DolphinCoreMLRuntime

final class MetadataTests: XCTestCase {
    func testDecodeMetadata() throws {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("model.mlpackage", isDirectory: true)
        try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)

        let payload: [String: Any] = [
            "model": [
                "identifier": "repo/model",
                "revision": "main",
                "variant": "8B",
                "family": "llama3",
                "context_length": 8192,
                "vocab_size": 128256,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rope_theta": 10000.0,
                "rope_scaling": [
                    "type": "linear",
                    "factor": 1.0,
                    "original_max_position_embeddings": 8192
                ],
                "embedding_dimension": 4096,
                "parameter_count": 8000000000
            ],
            "quantization": [
                "wbits": 4,
                "group_size": 16,
                "palett_granularity": "per_grouped_channel",
                "variant_index": 0,
                "variant_count": 1
            ],
            "pipeline": [
                "generated_at": "2024-07-01T00:00:00Z",
                "compute_units": "ALL",
                "script": "dolphin2coreml_full.py"
            ],
            "lora": [
                "path": "lora",
                "base_model_name": "base",
                "rank": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "k_proj"],
                "scaling": 0.5
            ],
            "llm2vec": [
                "path": "llm2vec",
                "embedding_dimension": 4096,
                "projection_dim": 4096,
                "pooling": "mean",
                "normalize": true
            ]
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        let metadataURL = packageURL.appendingPathComponent(DolphinModelMetadata.fileName)
        try data.write(to: metadataURL)

        let metadata = try DolphinModelMetadata.load(from: packageURL)
        XCTAssertEqual(metadata.model.variant, "8B")
        XCTAssertEqual(metadata.model.numKeyValueHeads, 8)
        XCTAssertEqual(metadata.quantization?.wbits, 4)
        XCTAssertEqual(metadata.pipeline?.computeUnits, "ALL")
        XCTAssertEqual(metadata.llm2vec?.embeddingDimension, 4096)
    }

    func testLoadMissingMetadataThrows() throws {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("model.mlpackage", isDirectory: true)
        try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)

        XCTAssertThrowsError(try DolphinModelMetadata.load(from: packageURL))
    }
}
