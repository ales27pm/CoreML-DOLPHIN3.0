// Package vectormath provides vector arithmetic primitives tuned for parity
// with the repository's cross-language dot product implementations.
//
// ### func Dot(a []float64, b []float64) (float64, error)
//
// Dot multiplies corresponding elements in *a* and *b* and returns their sum.
// The slices must be equal length.
//
// **Inputs**
//
// - `a []float64`: First vector.
// - `b []float64`: Second vector.
//
// **Outputs**
//
// - `(float64, error)`: Dot product or error when lengths mismatch.
//
// **Example**
//
//	result, err := Dot([]float64{1, 2}, []float64{3, 4})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(result) // 11
//
// **Complexity**
//
// - Time: O(n)
// - Space: O(1)
package vectormath
