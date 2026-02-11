# 64-bit Segregated Free List Allocator

A 64-bit dynamic memory allocator implemented in C as part of a systems programming project. The allocator supports prompt coalescing, segregated free lists, and block splitting to balance throughput and memory utilization.

## Architecture

The allocator uses a **segregated free list** design with **explicit free lists** within each size class.

- **Word Size**: 8 bytes (64-bit).
- **Alignment**: 16 bytes (double word alignment).
- **Minimum Block Size**: 16 bytes.
- **Header/Footer**: Each block maintains an 8-byte header and footer containing size and allocation status.

### Segregated Free Lists
The allocator maintains an array of 14 distinct free lists (`segragatedlist`), each dedicated to a specific range of block sizes. This approach mimics a "best-fit" search strategy by segregating blocks into power-of-two size classes (e.g., 32, 64, 128... up to >65536 bytes).

- **O(1) Insertion**: Freed blocks are inserted at the head of the appropriate bucket (LIFO policy).
- **Efficient Search**: Allocation requests scan the appropriate segregation bucket. If a fit is not found, the search continues to the next larger size class.

### Allocation Strategy
- **Best Fit Approximation**: Within a bucket, the allocator searches for a suitable free block. To balance throughput with utilization, the search is limited to the first 10 fit candidates ("first-10-fit").
- **Block Splitting**: If a selected block is significantly larger than the request, it is split to minimize internal fragmentation. The remainder is returned to the appropriate free list immediately.

### Coalescing
The allocator implements **immediate coalescing** with boundary tag coalescing.
- When a block is freed, the allocator checks the allocation status of its immediate physical neighbors (using the footer of the previous block and header of the next).
- Contiguous free blocks are merged instantly to form larger free blocks, reducing external fragmentation.

## Design Decisions & Trade-offs

1.  **Immediate vs. Deferred Coalescing**:
    - *Decision*: Immediate coalescing was chosen to maximize memory reuse and minimize external fragmentation.
    - *Trade-off*: Slightly higher cost per `free` operation compared to deferred coalescing, but prevents heap fragmentation from degrading performance over time.

2.  **Segregated Lists vs. Single Free List**:
    - *Decision*: Segregated lists provide substantially faster allocation times than a single implicit or explicit list.
    - *Trade-off*: More complex implementation and slightly higher memory overhead for the list heads (array of pointers).

3.  **Segregated List Bucket Count**:
    - *Decision*: 14 buckets were chosen to provide fine-grained separation for common small sizes while grouping larger sizes.
    - *Trade-off*: Matches the workload distribution of typical systems programs where small allocations dominate.

## API

The allocator exposes a standard interface prefixed with `mm_` to avoid conflicts with the system libc:

```c
bool mm_init(void);
void *mm_malloc(size_t size);
void mm_free(void *ptr);
void *mm_realloc(void *ptr, size_t size);
void *mm_calloc(size_t nmemb, size_t size);
```

## Building

To compile the object files:
```bash
make
```

To run the embedded demonstration:
```bash
make demo
```
