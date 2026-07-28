//! Exact port of vLLM v0.25.0's prefix-cache block hashing.
//!
//! Mirrors (all paths relative to `vllm-v0.25.0-edited/`):
//!   * `vllm/utils/hashing.py:26` `sha256()`  -> pickle protocol-5 dump + SHA-256
//!   * `vllm/v1/core/kv_cache_utils.py:99`  `init_none_hash()`
//!   * `vllm/v1/core/kv_cache_utils.py:577` `hash_block_tokens()`
//!   * `vllm/v1/core/kv_cache_utils.py:673` `get_request_block_hasher()`
//!   * `vllm/v1/core/kv_cache_utils.py:57`  `make_block_hash_with_group_id()`
//!
//! Only the `sha256` (pickle) hash function is ported. `sha256_cbor` / `xxhash*`
//! return an error at construction so the patch refuses to enable rather than
//! silently producing a different key space (a silent mismatch zeroes the prefix
//! hit rate).
//!
//! ponytail: the pickle memo is write-only (we emit MEMOIZE, never BINGET). CPython
//! emits BINGET when the *same object* appears twice in one dump; with our input shape
//! (`(bytes, tuple[int], extra_keys)`) that needs two identical interned strings inside
//! one block's extra keys, which vLLM never produces. Upgrade path if it ever does:
//! keep an identity-keyed memo in `Pickler` and emit `BINGET`/`LONG_BINGET`.

use sha2::{Digest, Sha256};

pub const HASH_LEN: usize = 32;
pub type Digest32 = [u8; HASH_LEN];

/// Extra-key payload for a block hash (`generate_block_hash_extra_keys`, kv_cache_utils.py:539).
/// LoRA name -> `Str`, MM feature -> `Tuple([Str(identifier), Int(offset)])`,
/// cache salt -> `Str`, prompt-embeds -> `Bytes`.
#[derive(Clone, Debug, PartialEq)]
pub enum Key {
    Str(String),
    Int(i64),
    Bytes(Vec<u8>),
    Tuple(Vec<Key>),
    None,
}

/// Minimal pickle protocol-5 writer covering the object shapes vLLM hashes.
struct Pickler {
    buf: Vec<u8>,
}

impl Pickler {
    fn new(cap: usize) -> Self {
        let mut buf = Vec::with_capacity(cap);
        buf.push(0x80); // PROTO
        buf.push(5);
        buf.extend_from_slice(&[0x95, 0, 0, 0, 0, 0, 0, 0, 0]); // FRAME + u64 placeholder
        Self { buf }
    }

    fn finish(mut self) -> Vec<u8> {
        self.buf.push(b'.'); // STOP
        // The frame length covers everything after the 8-byte frame header.
        let frame_len = (self.buf.len() - 11) as u64;
        self.buf[3..11].copy_from_slice(&frame_len.to_le_bytes());
        self.buf
    }

    fn bytes(&mut self, b: &[u8]) {
        if b.len() < 256 {
            self.buf.push(b'C'); // SHORT_BINBYTES
            self.buf.push(b.len() as u8);
        } else if b.len() <= u32::MAX as usize {
            self.buf.push(b'B'); // BINBYTES
            self.buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        } else {
            self.buf.push(0x8e); // BINBYTES8
            self.buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
        }
        self.buf.extend_from_slice(b);
        self.buf.push(0x94); // MEMOIZE
    }

    fn str(&mut self, s: &str) {
        let b = s.as_bytes();
        if b.len() < 256 {
            self.buf.push(0x8c); // SHORT_BINUNICODE
            self.buf.push(b.len() as u8);
        } else {
            self.buf.push(b'X'); // BINUNICODE
            self.buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        }
        self.buf.extend_from_slice(b);
        self.buf.push(0x94); // MEMOIZE
    }

    /// `save_long` (CPython Modules/_pickle.c). Ints are never memoized.
    fn int(&mut self, v: i64) {
        if (0..256).contains(&v) {
            self.buf.push(b'K'); // BININT1
            self.buf.push(v as u8);
        } else if (256..65536).contains(&v) {
            self.buf.push(b'M'); // BININT2
            self.buf.extend_from_slice(&(v as u16).to_le_bytes());
        } else if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
            self.buf.push(b'J'); // BININT
            self.buf.extend_from_slice(&(v as i32).to_le_bytes());
        } else {
            // LONG1: minimal two's-complement little-endian byte string.
            let mut b = v.to_le_bytes().to_vec();
            let fill = if v < 0 { 0xffu8 } else { 0x00u8 };
            let sign_ok = |last: u8| if v < 0 { last & 0x80 != 0 } else { last & 0x80 == 0 };
            while b.len() > 1 && b[b.len() - 1] == fill && sign_ok(b[b.len() - 2]) {
                b.pop();
            }
            self.buf.push(0x8a); // LONG1
            self.buf.push(b.len() as u8);
            self.buf.extend_from_slice(&b);
        }
    }

    fn none(&mut self) {
        self.buf.push(b'N');
    }

    /// `save_tuple`: <=3 items use the fixed TUPLE1/2/3 opcodes, otherwise MARK..TUPLE.
    /// The empty tuple uses EMPTY_TUPLE and is *not* memoized.
    fn tuple_start(&mut self, n: usize) {
        if n > 3 {
            self.buf.push(b'('); // MARK
        }
    }

    fn tuple_end(&mut self, n: usize) {
        match n {
            0 => {
                self.buf.push(b')'); // EMPTY_TUPLE, no MEMOIZE
                return;
            }
            1 => self.buf.push(0x85),
            2 => self.buf.push(0x86),
            3 => self.buf.push(0x87),
            _ => self.buf.push(b't'),
        }
        self.buf.push(0x94); // MEMOIZE
    }

    fn key(&mut self, k: &Key) {
        match k {
            Key::Str(s) => self.str(s),
            Key::Int(i) => self.int(*i),
            Key::Bytes(b) => self.bytes(b),
            Key::None => self.none(),
            Key::Tuple(items) => {
                self.tuple_start(items.len());
                for it in items {
                    self.key(it);
                }
                self.tuple_end(items.len());
            }
        }
    }
}

/// `hash_block_tokens` input: `(parent_block_hash, tuple(curr_block_token_ids), extra_keys)`.
pub fn pickle_block_hash_input(
    parent: &[u8],
    tokens: &[i64],
    extra_keys: Option<&[Key]>,
) -> Vec<u8> {
    let mut p = Pickler::new(64 + tokens.len() * 3);
    // The hashed object is the 3-tuple (parent, tokens, extra_keys).
    p.tuple_start(3);
    p.bytes(parent);
    p.tuple_start(tokens.len());
    for &t in tokens {
        p.int(t);
    }
    p.tuple_end(tokens.len());
    match extra_keys {
        None => p.none(),
        Some(keys) => {
            p.tuple_start(keys.len());
            for k in keys {
                p.key(k);
            }
            p.tuple_end(keys.len());
        }
    }
    p.tuple_end(3);
    p.finish()
}

/// vLLM's `sha256(obj)` = `sha256(pickle.dumps(obj, protocol=5)).digest()`.
pub fn sha256_of(bytes: &[u8]) -> Digest32 {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// `hash_block_tokens` (kv_cache_utils.py:577).
///
/// `parent` is `None` for the first block, in which case `NONE_HASH` is substituted.
/// Note vLLM uses `if not parent_block_hash`, so an *empty* parent also falls back.
pub fn hash_block_tokens(
    none_hash: &Digest32,
    parent: Option<&[u8]>,
    tokens: &[i64],
    extra_keys: Option<&[Key]>,
) -> Digest32 {
    let parent = match parent {
        Some(p) if !p.is_empty() => p,
        _ => &none_hash[..],
    };
    sha256_of(&pickle_block_hash_input(parent, tokens, extra_keys))
}

/// `init_none_hash` (kv_cache_utils.py:99) for the `sha256` (non-CBOR) function:
/// unset `PYTHONHASHSEED` -> 32 random bytes; set -> `sha256(seed_string)`.
///
/// The random branch is intentionally NOT reproduced here: the live NONE_HASH is
/// always passed in from Python so shadow/authority modes share vLLM's value
/// whatever it is. This helper only covers the deterministic branch, for tests.
pub fn none_hash_from_seed(seed: &str) -> Digest32 {
    let mut p = Pickler::new(32 + seed.len());
    p.str(seed);
    sha256_of(&p.finish())
}

/// `get_request_block_hasher`'s inner loop (kv_cache_utils.py:687) with no MM inputs.
///
/// `extra_keys` is applied to every hashed block (LoRA / cache-salt shape); pass
/// `None` for the plain text-only case that the served workload uses.
pub fn hash_request_tokens(
    none_hash: &Digest32,
    hash_block_size: usize,
    token_ids: &[i64],
    extra_keys: Option<&[Key]>,
) -> Vec<Digest32> {
    let mut out = Vec::with_capacity(token_ids.len() / hash_block_size.max(1));
    let mut prev: Option<Digest32> = None;
    let mut start = 0usize;
    while start + hash_block_size <= token_ids.len() {
        let end = start + hash_block_size;
        let h = hash_block_tokens(
            none_hash,
            prev.as_ref().map(|p| &p[..]),
            &token_ids[start..end],
            extra_keys,
        );
        out.push(h);
        prev = Some(h);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vectors captured from CPython 3.12:
    ///   pickle.dumps((parent, tokens, extra), protocol=5)
    #[test]
    fn pickle_matches_cpython() {
        let p32 = [0u8; 32];
        let got = pickle_block_hash_input(&p32, &[1, 2, 3], None);
        let mut want = vec![0x80, 0x05, 0x95, 0x2f, 0, 0, 0, 0, 0, 0, 0, b'C', 32];
        want.extend_from_slice(&p32);
        want.extend_from_slice(&[
            0x94, b'K', 1, b'K', 2, b'K', 3, 0x87, 0x94, b'N', 0x87, 0x94, b'.',
        ]);
        assert_eq!(got, want);

        // Empty parent bytes, single token -> TUPLE1.
        let got = pickle_block_hash_input(&[], &[1], None);
        let want: Vec<u8> = vec![
            0x80, 0x05, 0x95, 0x0b, 0, 0, 0, 0, 0, 0, 0, b'C', 0, 0x94, b'K', 1, 0x85, 0x94,
            b'N', 0x87, 0x94, b'.',
        ];
        assert_eq!(got, want);

        // >3 tokens -> MARK .. TUPLE, and the int-width ladder.
        let got = pickle_block_hash_input(&[0xbb; 32], &[300, 70000, 5_000_000_000, -3], None);
        let want_tail: Vec<u8> = vec![
            0x94, b'(', b'M', 0x2c, 0x01, b'J', 0x70, 0x11, 0x01, 0x00, 0x8a, 0x05, 0x00, 0xf2,
            0x05, 0x2a, 0x01, b'J', 0xfd, 0xff, 0xff, 0xff, b't', 0x94, b'N', 0x87, 0x94, b'.',
        ];
        assert_eq!(&got[got.len() - want_tail.len()..], &want_tail[..]);
        assert_eq!(&got[..11], &[0x80, 0x05, 0x95, 0x3e, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(got.len(), 73);
    }

    #[test]
    fn pickle_extra_keys() {
        // extra_keys = ('lora', 'x')  -> two SHORT_BINUNICODE + TUPLE2
        let got = pickle_block_hash_input(
            &[0xbb; 32],
            &[300, 70000, 5_000_000_000, -3],
            Some(&[Key::Str("lora".into()), Key::Str("x".into())]),
        );
        let want_tail: Vec<u8> = vec![
            b't', 0x94, 0x8c, 4, b'l', b'o', b'r', b'a', 0x94, 0x8c, 1, b'x', 0x94, 0x86, 0x94,
            0x87, 0x94, b'.',
        ];
        assert_eq!(&got[got.len() - want_tail.len()..], &want_tail[..]);
    }

    #[test]
    fn chain_is_prefix_dependent() {
        let nh = none_hash_from_seed("0");
        let a = hash_request_tokens(&nh, 4, &[1, 2, 3, 4, 5, 6, 7, 8], None);
        let b = hash_request_tokens(&nh, 4, &[1, 2, 3, 4, 5, 6, 7, 9], None);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], b[0], "shared prefix block must hash identically");
        assert_ne!(a[1], b[1]);
        // Partial trailing block is not hashed.
        let c = hash_request_tokens(&nh, 4, &[1, 2, 3, 4, 5], None);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0], a[0]);
    }

    #[test]
    fn none_hash_substituted_for_first_block() {
        let nh = none_hash_from_seed("seed");
        let explicit = hash_block_tokens(&nh, Some(&nh[..]), &[7, 7], None);
        let implicit = hash_block_tokens(&nh, None, &[7, 7], None);
        assert_eq!(explicit, implicit);
    }
}
