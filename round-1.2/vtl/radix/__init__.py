"""Token-granular SGLang RadixAttention memory model, ported to run inside vLLM.

Off-box-testable foundation (this package) + on-box integration frontier (INTEGRATION.md).
The tree core lives one level up in ``vtl.radix_tree``; this package adds the SGLang memory
primitives and the scheduler-facing flow that make the tree authoritative over KV.
"""

from vtl.radix.radix_cache import TokenRadixCache
from vtl.radix.req_to_token_pool import ReqToTokenPool
from vtl.radix.token_allocator import TokenToKVPoolAllocator

__all__ = ["TokenRadixCache", "ReqToTokenPool", "TokenToKVPoolAllocator"]
