package vectormath

import "fmt"

// Dot multiplies corresponding elements and returns their sum.
func Dot(a []float64, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("length mismatch: got %d and %d", len(a), len(b))
	}
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}
