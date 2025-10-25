#ifndef FFI_BRIDGE_H
#define FFI_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Mirrors the Rust `rust_echo` symbol.
char *rust_echo(const unsigned char *ptr, size_t len);

/// Convenience wrapper that accepts a traditional C string.
char *rust_echo_c(const char *ptr);

/// Releases memory allocated by `rust_echo`/`rust_echo_c`.
void rust_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif // FFI_BRIDGE_H
