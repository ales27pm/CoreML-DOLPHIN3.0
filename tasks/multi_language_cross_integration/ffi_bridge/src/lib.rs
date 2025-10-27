//! Swift ↔ Rust bridge helpers.
//!
//! The exported functions provide a minimal yet production-friendly string
//! roundtrip that mirrors the bridge described in `Codex_Master_Task_Results.md`.
//! The Rust side exposes UTF-8 aware entry points together with an allocator-aware
//! deleter so that foreign runtimes (Swift, Python via `ctypes`, etc.) can manage
//! ownership deterministically.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Creates an owned [`CString`] from a byte slice without panicking.
fn utf8_cstring(bytes: &[u8]) -> Option<CString> {
    match std::str::from_utf8(bytes) {
        Ok(valid) => CString::new(valid).ok(),
        Err(err) => {
            log::error!("invalid utf8 payload: {err}");
            None
        }
    }
}

/// Echoes a UTF-8 byte buffer and returns an owned C string pointer.
///
/// The caller takes ownership of the returned pointer and must release it by
/// invoking [`rust_free`]. When invalid UTF-8 bytes are supplied the function
/// returns `NULL` so that the caller can surface an appropriate error.
///
/// # Safety
///
/// `ptr` must either be null (paired with `len == 0`) or reference a valid
/// buffer of at least `len` bytes that remains live for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn rust_echo(ptr: *const u8, len: usize) -> *mut c_char {
    if len == 0 {
        return CString::new("")
            .expect("empty string is always valid")
            .into_raw();
    }

    if ptr.is_null() {
        log::error!("received null pointer with non-zero length");
        return ptr::null_mut();
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    utf8_cstring(bytes)
        .map(CString::into_raw)
        .unwrap_or_else(ptr::null_mut)
}

/// Echoes a null-terminated C string.
///
/// This overload is convenient for languages that naturally produce
/// `char *` payloads. Ownership semantics mirror [`rust_echo`].
///
/// # Safety
///
/// `ptr` must either be null or reference a valid, NUL-terminated byte buffer
/// that remains live for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn rust_echo_c(ptr: *const c_char) -> *mut c_char {
    if ptr.is_null() {
        return CString::new("")
            .expect("empty string is always valid")
            .into_raw();
    }

    let c_string = unsafe { CStr::from_ptr(ptr) };
    let bytes = c_string.to_bytes();
    utf8_cstring(bytes)
        .map(CString::into_raw)
        .unwrap_or_else(ptr::null_mut)
}

/// Releases memory allocated by [`rust_echo`] or [`rust_echo_c`].
///
/// # Safety
///
/// `ptr` must originate from a call to [`rust_echo`] or [`rust_echo_c`] and
/// must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn rust_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }

    // SAFETY: the pointer originated from `CString::into_raw` so converting it
    // back via `from_raw` yields the owning instance which then drops.
    drop(CString::from_raw(ptr));
}

#[cfg(feature = "swift")]
extern "C" {
    fn swift_echo(ptr: *const c_char) -> *mut c_char;
}

/// Calls into the Swift echo implementation.
///
/// This API is compiled only when the `swift` Cargo feature is enabled and a
/// Swift object file providing `swift_echo` is linked. It mirrors the
/// roundtrip used to validate cross-language invocation in production builds.
#[cfg(feature = "swift")]
#[no_mangle]
pub extern "C" fn rust_call_swift(ptr: *const c_char) -> *mut c_char {
    unsafe { swift_echo(ptr) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_from_bytes() {
        let payload = b"hello from rust";
        let pointer = unsafe { rust_echo(payload.as_ptr(), payload.len()) };
        assert!(!pointer.is_null());

        let result = unsafe { CStr::from_ptr(pointer) };
        assert_eq!(result.to_str().unwrap(), "hello from rust");

        unsafe { rust_free(pointer) };
    }

    #[test]
    fn roundtrip_from_c_string() {
        let original = CString::new("swift bridge").unwrap();
        let pointer = unsafe { rust_echo_c(original.as_ptr()) };
        assert!(!pointer.is_null());

        let result = unsafe { CStr::from_ptr(pointer) };
        assert_eq!(result.to_str().unwrap(), "swift bridge");

        unsafe { rust_free(pointer) };
    }

    #[test]
    fn invalid_utf8_returns_null() {
        let bytes = [0xf8u8, 0x28, 0x8c, 0x28];
        let pointer = unsafe { rust_echo(bytes.as_ptr(), bytes.len()) };
        assert!(pointer.is_null());
    }

    #[test]
    fn null_pointer_with_zero_length_returns_empty_string() {
        let pointer = unsafe { rust_echo(ptr::null(), 0) };
        assert!(!pointer.is_null());

        let result = unsafe { CStr::from_ptr(pointer) };
        assert_eq!(result.to_str().unwrap(), "");

        unsafe { rust_free(pointer) };
    }
}
