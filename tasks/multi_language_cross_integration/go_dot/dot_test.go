package vectormath

import "testing"

func TestDotComputesProduct(t *testing.T) {
	result, err := Dot([]float64{1, 2, 3}, []float64{4, 5, 6})
	if err != nil {
		t.Fatalf("Dot returned error: %v", err)
	}
	if result != 32 {
		t.Fatalf("unexpected dot product: got %f", result)
	}
}

func TestDotLengthMismatch(t *testing.T) {
	_, err := Dot([]float64{1, 2}, []float64{1})
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
	if got := err.Error(); got != "length mismatch: got 2 and 1" {
		t.Fatalf("unexpected error message: %s", got)
	}
}
