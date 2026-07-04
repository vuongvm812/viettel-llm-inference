//! Output buffer trait for [`Consumer::peek_range`](crate::lossy::Consumer::peek_range).
//!
//! Mirrors cyclotrace's `Sink` shape so a peek-snapshot can write into a `Vec`,
//! a fixed-capacity `arrayvec::ArrayVec`, or any user-supplied buffer without
//! the ring needing to know the concrete container.
//!
//! `peek_range` reserves up-front via [`SpmcSink::reserve`] and rolls back to
//! the entry length via [`SpmcSink::truncate`] when a torn slot is detected
//! mid-snapshot, so the caller's buffer is never left half-populated with
//! mixed-snapshot values.
//!
//! Only [`Vec<T>`] is provided in this crate. The `ArrayVec`/`heapless`
//! impls cyclotrace ships are intentionally deferred until a call site asks.

/// Append-only output buffer used by [`Consumer::peek_range`](crate::lossy::Consumer::peek_range).
///
/// The contract:
///
/// * `push(item, i)` appends `item` at logical index `i` (0 = oldest of the
///   snapshot). The implementor MAY ignore `i` (e.g. `Vec::push`), but a
///   fixed-position buffer MAY use it to write directly into slot `i`.
/// * `reserve(additional)` is a capacity hint; the implementor MAY ignore it
///   or panic if it cannot grow (e.g. an array-backed sink with insufficient
///   room).
/// * `truncate(len)` is called by `peek_range` to roll back to the buffer's
///   entry length when the snapshot tears. Implementors must drop appended
///   items beyond `len`.
/// * `len()` / `remain()` describe current length and remaining capacity so
///   `peek_range` can size its reservation correctly.
pub trait SpmcSink {
    /// The item type the snapshot writes.
    type Item;

    /// Append `item` at logical snapshot index `i` (0 = oldest in the snapshot).
    fn push(&mut self, item: Self::Item, i: usize);

    /// Reserve room for `additional` more items.
    fn reserve(&mut self, additional: usize);

    /// Truncate to `len`. Items at indices `>= len` are dropped.
    fn truncate(&mut self, len: usize);

    /// Current length.
    fn len(&self) -> usize;

    /// Remaining capacity. May report `usize::MAX` for unbounded sinks.
    fn remain(&self) -> usize;

    /// True iff [`Self::len`] is zero.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> SpmcSink for Vec<T> {
    type Item = T;

    #[inline]
    fn push(&mut self, item: T, _i: usize) {
        Vec::push(self, item);
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }

    #[inline]
    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len);
    }

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn remain(&self) -> usize {
        self.capacity() - self.len()
    }
}
