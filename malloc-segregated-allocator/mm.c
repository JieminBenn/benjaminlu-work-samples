/**
 * @file mm.c
 * @brief A 64-bit struct-based implicit free list memory allocator
 *
 * This allocator implements a segregated free list with immediate verification
 * and coalescing. It uses 16-byte alignment and supports 64-bit architectures.
 *
 * @author Benjamin Lu <blu2@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"

/*
 * Debugging Macros
 * Enabled by compiling with -DDEBUG
 */
#ifdef DEBUG
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printf(...) ((void)printf(__VA_ARGS__))
#else
#define dbg_discard_expr_(...) ((void)((0) && printf(__VA_ARGS__)))
#define dbg_requires(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_assert(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_ensures(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_printf(...) dbg_discard_expr_(__VA_ARGS__)
#endif

/* Basic Types and Constants */
typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = dsize;

/**
 * Chunksize: size of heap extension
 * (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 12);

/**
 * Mask to determine if the least significant bit is allocated.
 */
static const word_t alloc_mask = 0x1;

/**
 * Mask used to get the block size.
 */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Represents the header and payload of one block in the heap */
typedef struct block block_t;
struct block {
  /** @brief Header contains size + allocation flag */
  word_t header;
  union {
    char payload[0];
    struct {
      struct block *next; // pointer to next block
      struct block *prev; // pointer to prev block
    };
  };
};

/* Function Prototypes */
bool mm_init(void);
void *mm_malloc(size_t size);
void mm_free(void *ptr);
void *mm_realloc(void *ptr, size_t size);
void *mm_calloc(size_t nmemb, size_t size);
bool mm_checkheap(int line);
static block_t *extend_heap(size_t size);
static block_t *coalesce_block(block_t *block);
static block_t *find_fit(size_t asize);
static void split_block(block_t *block, size_t asize);

/* Global variables */

/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

// Pointer to the free list, points to first element
static block_t *freeblockpointer = NULL;

// creating the segregated list with length 14
static block_t *segragatedlist[14];
/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) { return (x > y) ? x : y; }

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
  return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool prevallocated,
                   bool isprevmini) {
  word_t word = size;
  if (alloc) { // the least significant bit represents whether this block is
               // allocated
    word |= alloc_mask;
  }
  if (prevallocated) { // this bit represents whether the previous block is
                       // allocated
    word |= 0x2;
  }
  if (isprevmini) { // this bit represents whether the previous block is a
                    // mini block
    word |= 0x4;
  }
  return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) { return (word & size_mask); }

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) { return extract_size(block->header); }

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
  return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
  dbg_requires(get_size(block) != 0);
  return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
  dbg_requires(get_size(block) != 0 &&
               "Called header_to_footer on the epilogue block");
  return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
  size_t size = extract_size(*footer);
  dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
  return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
  size_t asize = get_size(block);
  return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) { return (bool)(word & alloc_mask); }

// Returns whether the previous block is a mini block given header value.
static bool extract_prevmini(word_t word) { return (bool)(word & 0x4); }

// Returns whether the previous block is a mini block, based on its header.
static bool get_prevmini(block_t *block) {
  return extract_prevmini(block->header);
}

// Returns the allocation status of the previous block give header value
static bool extract_prevalloc(word_t word) { return (bool)(word & 0x2); }

// Returns the allocation status of previous block, based on its header.
static bool get_prevalloc(block_t *block) {
  return extract_prevalloc(block->header);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) { return extract_alloc(block->header); }

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block) {
  dbg_requires(block != NULL);
  dbg_requires((char *)block == (char *)mem_heap_hi() - 7);
  block->header = pack(0, true, true, false);
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
  dbg_requires(block != NULL);
  dbg_requires(get_size(block) != 0 &&
               "Called find_next on the last block in the heap");
  return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes both a header and footer, where the location of the
 * footer is computed in relation to the header.
 *
 * TODO: Are there any preconditions or postconditions?
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 */
static void write_block(block_t *block, size_t size, bool alloc, bool prevalloc,
                        bool isprevmini) {
  dbg_requires(block != NULL);
  dbg_requires(size > 0);
  block->header = pack(size, alloc, prevalloc, isprevmini);
  // If the block isnt allocated and isnt a mini block, it will have footer
  if (!alloc && (size > 16)) {
    word_t *footerp = header_to_footer(block);
    *footerp = pack(size, alloc, prevalloc, isprevmini);
  }
  // sets the allocation bit and mini block bits for the next block if not
  // epilogue
  block_t *next = find_next(block);
  if (get_size(find_next(block)) > 0) {
    size_t nextsize = get_size(next);
    size_t nextalloc = get_alloc(next);
    bool ismini = (size == min_block_size);
    next->header = pack(nextsize, nextalloc, alloc, ismini);
  }
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
  // Compute previous footer position as one word before the header
  return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
  dbg_requires(block != NULL);

  word_t *footerp = find_prev_footer(block);

  // Return NULL if called on first block in the heap
  if (extract_size(*footerp) == 0) {
    return NULL;
  }
  return footer_to_header(footerp);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

// This function gets the index or bucket of the segmented list.
// The segmented list allows blocks of different sizes to be put into different
// buckets, so when we search free blocks it would be more efficient
static int getSegmentedListIndex(size_t size) {
  if (size <= min_block_size) {
    return 0;
  } // The first bucket will be filled with mini blocks
  else if (size <= 32) {
    return 1;
  } else if (size <= 64) {
    return 2;
  } else if (size <= 128) {
    return 3;
  } else if (size <= 256) {
    return 4;
  } else if (size <= 512) {
    return 5;
  } else if (size <= 1024) {
    return 6;
  } else if (size <= 2048) {
    return 7;
  } else if (size <= 4096) {
    return 8;
  } else if (size <= 8192) {
    return 9;
  } else if (size <= 16384) {
    return 10;
  } else if (size <= 32768) {
    return 11;
  } else if (size <= 65536) {
    return 12;
  } else {
    return 13;
  }
  // Any block with size greater than 65536 will go to bucket 13
}

// Updates the free list by adding freed blocks or newly added free blocks to
// the free list
static void updatefreelist_add(block_t *block) {
  // gets the index of the bucket the block should be put in
  int index = getSegmentedListIndex(get_size(block));
  // We add newly freed blocks to the front of the list
  block->next = segragatedlist[index];
  // if a mini block is added, it will not use prev
  if (index != 0) {
    block->prev = NULL;
    if (segragatedlist[index] != NULL) {
      segragatedlist[index]->prev = block;
    }
  }
  segragatedlist[index] = block; // The start of this bucket is now this block
}

// Updates the free list by deleting allocated blocks in the free list
static void updatefreelist_delete(block_t *block) {
  dbg_requires(block != NULL);

  // gets the index of the bucket the block should be put in
  int index = getSegmentedListIndex(get_size(block));

  if (index == 0) { // for mini blocks, if the block is the first value,
    // then we just delete it
    if (segragatedlist[index] == block) {
      segragatedlist[index] = block->next;
    } else { // else, we will need to loop through the entire list to find
             // the target block
      block_t *curr = segragatedlist[0];
      block_t *prev = NULL;
      while (curr != NULL) {
        if (curr == block) {
          prev->next = curr->next;
          break;
        }
        prev = curr;
        curr = curr->next;
      }
    }
  }
  // for non-mini blocks, we can access the prev pointer
  else {
    // If this block is the only block in the bucket, set pointer to NULL
    if (block->prev == NULL && block->next == NULL) {
      segragatedlist[index] = NULL;
    }
    // Connect the prev and next blocks
    if (block->prev != NULL) {
      block->prev->next = block->next;
    } else {
      segragatedlist[index] = block->next;
    }
    if (block->next != NULL) {
      block->next->prev = block->prev;
    }
    block->prev = NULL; // Set the previous block to NULL
  }

  block->next = NULL; // Set the next block to NULL
}

/**
 * @brief Coalesces the current block with its free neighbors.
 *
 * Checks the allocation status of the previous and next blocks.
 * If either or both are free, they are merged into a single larger free block.
 * The coalesced block is then added to the appropriate free list.
 *
 * @param[in] block Pointer to the block to coalesce.
 * @return Pointer to the coalesced block.
 * @pre block must not be NULL and must be free.
 */
static block_t *coalesce_block(block_t *block) {
  assert(!get_alloc(block));
  block_t *next = find_next(block); // gets the next block

  // we find the status of previous and current blocks
  bool isprevallocated = get_prevalloc(block);
  bool isnextallocated = get_alloc(next);
  bool isprevmini = get_prevmini(block);

  size_t size = get_size(block); // get size of block

  // Case 1 (next allocated and prev allocated)
  if (isnextallocated && isprevallocated) {
    updatefreelist_add(block); // add to free list
    write_block(block, size, false, isprevallocated, isprevmini);
    return block;
  }

  // Case 2 (next not allocated and prev allocated)
  if (!isnextallocated && isprevallocated) {
    size += get_size(next);
    updatefreelist_delete(next); // delete next from free list as it is combined
    write_block(block, size, false, isprevallocated, isprevmini);
    updatefreelist_add(block);
    return block;
  }

  block_t *prev; // if previous block is mini block, we won't be able to use
                 // find_prev so we subtract by min_block_size
  if (!isprevmini) {
    prev = find_prev(block);
  } else {
    prev = (block_t *)((char *)block - min_block_size);
  }

  if (prev == NULL) {
    isprevallocated = true; // means it is the first block
  }

  // Information on whether the previous block of the previous block is a mini
  // block
  bool prevprevmini = get_prevmini(prev);

  // Case 3 (next allocated and prev not allocated)
  if (isnextallocated && !isprevallocated) {
    // printf("case 3\n");
    size += get_size(prev);
    updatefreelist_delete(prev); // delete prev from free list
    write_block(prev, size, false, get_prevalloc(prev), prevprevmini);
    updatefreelist_add(prev); // add new prev to free list
    return prev;
  }

  // Case 4 (next not allocated and prev not allocated)
  if (!isnextallocated && !isprevallocated) {
    // printf("case 4\n");
    size += get_size(next) + get_size(prev);
    updatefreelist_delete(prev);
    updatefreelist_delete(next); // delete prev and next from free list
    write_block(prev, size, false, get_prevalloc(prev), prevprevmini);
    updatefreelist_add(prev); // add new prev to free list
    return prev;
  }
  return block;
}

/**
 * @brief Extends the heap by requesting more memory from the system.
 *
 * Extends the heap by the specified size (rounded up to alignment).
 * Initializes the new free block and the new epilogue header.
 * Coalesces the new block with the previous block if it was free.
 *
 * @param[in] size The amount of memory to extend the heap by.
 * @return Pointer to the newly allocated (and potentially coalesced) free
 * block.
 * @pre size > 0
 */
static block_t *extend_heap(size_t size) {
  dbg_requires(size > 0);
  void *bp;

  // Allocate an even number of words to maintain alignment
  size = round_up(size, dsize);
  if ((bp = mem_sbrk((intptr_t)size)) == (void *)-1) {
    return NULL;
  }

  // Initialize free block header/footer
  block_t *block = payload_to_header(bp);
  write_block(block, size, false, get_prevalloc(block), get_prevmini(block));

  // Create new epilogue header
  block_t *block_next = find_next(block);
  write_epilogue(block_next);

  // Coalesce in case the previous block was free
  block = coalesce_block(block);

  dbg_ensures(mm_checkheap(__LINE__));
  return block;
}

/**
 * @brief Splits a block into two if the size is large enough.
 *
 * Takes an allocated block and splits it if the remaining size
 * is greater than or equal to the minimum block size.
 * The first part remains allocated, and the second part is marked as free
 * and added to the free list.
 *
 * @param[in] block The block to split.
 * @param[in] asize The adjusted size required for the allocated block.
 * @pre block assumes block is allocated.
 */
static void split_block(block_t *block, size_t asize) {
  dbg_requires(get_alloc(block));
  dbg_requires(asize >= min_block_size);

  size_t block_size = get_size(block); // get size of block
  if ((block_size - asize) >= min_block_size) {
    // If it can be split, then we split block and move part of it to free
    // list
    block_t *block_next;
    write_block(block, asize, true, get_prevalloc(block), get_prevmini(block));
    block_next = find_next(block);
    bool isMini = false; // check if current is a mini block
    if (asize == min_block_size) {
      isMini = true;
    }
    write_block(block_next, block_size - asize, false, true, isMini);
    updatefreelist_add(block_next);
  }

  dbg_ensures(get_alloc(block));
  dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief Finds a free block that fits the requested size.
 *
 * Uses a "first-10-fit" strategy within the segregated free lists.
 * Searches appropriate bucket and subsequent larger buckets for a suitable
 * block.
 *
 * @param[in] asize The adjusted size of the block needed.
 * @return Pointer to a fitting block, or NULL if no fit is found.
 */
static block_t *find_fit(size_t asize) {
  // Using find better fit
  dbg_requires(asize >= min_block_size);

  // gets the index of bucket
  int index = getSegmentedListIndex(asize);
  block_t *block;

  block_t *bestfit = NULL;
  size_t count = 0;

  // If current size is 16, then it is a mini-block and we look into
  // the 0th bucket in the segregated list.
  if (index == 0) {
    for (block = segragatedlist[0]; block != NULL; block = block->next) {
      if (asize <= min_block_size) {
        return block; // return if there is fit
        // If there is no fit, then we move on to the other buckets
      }
    }
  }

  // loops through the buckets that can contain the block with size asize.
  for (int i = index; i < 14; i++) {
    // for each bucket, iterate through the bucket to find a fit
    for (block = segragatedlist[i]; block != NULL; block = block->next) {
      // if we find a fit, we continue to look for the next 10 fits
      // and compare which one is the best fit (smallest size)
      if (asize <= get_size(block)) {
        if (count >= 10) {
          break;
        }
        if (bestfit == NULL) {
          bestfit = block;
        } else if (get_size(block) < get_size(bestfit)) {
          bestfit = block;
        }
        count++;
      }
    }
  }
  // When we find 10 fits, we return the best fit
  return bestfit; // If bestfit is NULL, then no fit found
}

/**
 * @brief Checks the heap for consistency and errors.
 *
 * Verifies the heap structure, including:
 * - Prologue and epilogue blocks.
 * - Block alignment and boundaries.
 * - Header and footer consistency.
 * - Coalescing (no consecutive free blocks).
 * - Free list consistency (pointers, counts, bucket ranges).
 *
 * @param[in] line The line number where validation was called (for debugging).
 * @return true if the heap is consistent, false otherwise.
 */
bool mm_checkheap(int line) {

  size_t numfreeblocks =
      0; // variables used to check consistency in free blocks
  size_t numfreelist = 0;

  if (heap_start == NULL) {
    dbg_printf("Heap start is NULL: (called at line %d)\n", line);
    return false;
  }

  // Check if there is prologue and if its correct
  block_t *prologue = (block_t *)((char *)heap_start - wsize);
  if (prologue == NULL || get_alloc(prologue) != 1 || get_size(prologue) != 0) {
    dbg_printf("Prologue error: (called at line %d)\n", line);
    return false;
  }

  block_t *start = heap_start;
  block_t *current;
  for (block_t *i = start; get_size(i) > 0; i = find_next(i)) {
    // Storing the current block in current for further uses after loop
    //(checking the epilogue)
    current = i;

    // Here we keep count of the number of free blocks in the heap for later
    // comparing with the number of blocks on free list.
    if (!get_alloc(i)) {
      numfreeblocks++;
    }

    // Check for alignment (make sure payload can be divided by dsize)
    if ((uintptr_t)header_to_payload(i) % dsize != 0) {
      dbg_printf("Alignment error: (called at line %d)\n", line);
      return false;
    }
    // Check for boundaries, make sure everything is in bound
    if ((void *)i < mem_heap_lo() || (void *)i > mem_heap_hi()) {
      dbg_printf("Not within boundaries: (called at line %d)\n", line);
      return false;
    }
    // Check each block's header and footer, making sure they are consitent
    if (i->header != *header_to_footer(i)) {
      dbg_printf("Header and Footer does not match: (called at line %d)\n",
                 line);
      return false;
    }

    // making sure everything has a legal size
    if (i->header < min_block_size) {
      dbg_printf("Illegal size: (called at line %d)\n", line);
      return false;
    }

    // Check previous/next allocate/free bit consistency
    block_t *next = find_next(i);
    if (i != heap_start) {
      block_t *prev = find_prev(i);
      if ((prev != NULL && get_size(prev) > 0 && !get_alloc(prev) &&
           !get_alloc(i))) {
        dbg_printf("Has consecutive free blocks: (called at line %d)\n", line);
        return false;
      }
    }

    // Check coalescing: no consecutive free blocks in the heap.
    if ((next != NULL && get_size(next) > 0 && !get_alloc(next) &&
         !get_alloc(i))) {
      dbg_printf("Has consecutive free blocks: (called at line %d)\n", line);
      return false;
    }
  }

  // Check epilogue
  if (!get_alloc(find_next(current)) || get_size(find_next(current)) != 0) {
    dbg_printf("Epilogue error: (called at line %d)\n", line);
    return false;
  }

  // Checking free list (explicit)

  // here we save a variable so that we can check if the pointers for next and
  // prev are pointing towards the correct block
  block_t *previousfreeblock = NULL;

  for (block_t *block = freeblockpointer; block != NULL; block = block->next) {
    numfreelist++; // counting number of blocks in free list

    // All free list pointers are between mem_heap_lo() and mem_heap_high()
    if ((void *)block < mem_heap_lo() || (void *)block > mem_heap_hi()) {
      dbg_printf("Free list pointers outside of bounds: (called at line %d)\n",
                 line);
      return false;
    }

    // All blocks in free list are freed (not allocated)
    if (get_alloc(block)) {
      dbg_printf("allocated block in free list: (called at line %d)\n", line);
      return false;
    }
    // All next/previous pointers are consistent
    if (previousfreeblock != NULL && previousfreeblock != block->prev) {
      dbg_printf("next/previous pointers inconsistent: (called at line %d)\n",
                 line);
      return false;
    }

    // Store the previous block so we can check if the blocks point
    // correctly to each other
    previousfreeblock = block;
  }

  // Check if free blocks in heap match number of blocks in free list
  if (numfreelist != numfreeblocks) {
    dbg_printf("Free list not matching true free blocks: (called at line %d)\n",
               line);
    return false;
  }

  // Segregated list
  // checking all blocks in each list bucket fall within bucket size range
  // (segregated list)
  for (int i = 0; i < 14; i++) {
    for (block_t *block = segragatedlist[i]; block != NULL;
         block = block->next) {
      if (i != getSegmentedListIndex(get_size(block))) {
        dbg_printf("Bucket size and block size mismatch: (called at "
                   "line %d)\n",
                   line);
        return false;
      }
    }
  }

  return true;
}

/**
 * @brief Initializes the memory allocator.
 *
 * Sets up the segregated free lists and the initial heap with prologue
 * and epilogue blocks. Extends the heap with an initial free block.
 *
 * @return true if initialization was successful, false otherwise.
 */
bool mm_init(void) {
  // Create the initial empty heap

  // Initialize free list pointer
  freeblockpointer = NULL;

  // Initialize segregated list
  for (int i = 0; i < 14; i++) {
    segragatedlist[i] = NULL;
  }

  // initialize heap
  word_t *start = (word_t *)(mem_sbrk(2 * wsize));

  if (start == (void *)-1) {
    return false;
  }

  start[0] = pack(0, true, true, false); // Heap prologue (block footer)
  start[1] = pack(0, true, true, false); // Heap epilogue (block header)

  // Heap starts with first "block header", currently the epilogue
  heap_start = (block_t *)&(start[1]);
  // Extend the empty heap with a free block of chunksize bytes
  if (extend_heap(chunksize) == NULL) {
    return false;
  }

  dbg_ensures(mm_checkheap(__LINE__));
  return true;
}

/**
 * @brief Allocates a block of memory of the specified size.
 *
 * Aligns the requested size, searches for a fitting free block,
 * and splits it if necessary. If no fit is found, extends the heap.
 *
 * @param[in] size The size of the memory block to allocate.
 * @return Pointer to the allocated memory block, or NULL on failure.
 */
void *mm_malloc(size_t size) {
  dbg_requires(mm_checkheap(__LINE__));
  size_t asize;      // Adjusted block size
  size_t extendsize; // Amount to extend heap if no fit is found
  block_t *block;
  void *bp = NULL;
  // Initialize heap if it isn't initialized
  if (heap_start == NULL) {
    if (!(mm_init())) {
      dbg_printf("Problem initializing heap. Likely due to sbrk");
      return NULL;
    }
  }
  // Ignore spurious request
  if (size == 0) {
    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
  }

  // Adjust block size to include overhead and to meet alignment requirements
  asize = round_up(size + wsize, dsize);

  // Search the free list for a fit
  block = find_fit(asize);
  // If no fit is found, request more memory, and then and place the block
  if (block == NULL) {
    // Always request at least chunksize
    extendsize = max(asize, chunksize);
    block = extend_heap(extendsize);
    // extend_heap returns an error
    if (block == NULL) {
      return bp;
    }
  }

  // The block should be marked as free
  dbg_assert(!get_alloc(block));

  // Mark block as allocated
  size_t block_size = get_size(block);
  write_block(block, block_size, true, get_prevalloc(block),
              get_prevmini(block));

  // When a block is allocated, delete from free list
  updatefreelist_delete(block);

  // Try to split the block if too large
  split_block(block, asize);

  bp = header_to_payload(block);

  dbg_ensures(mm_checkheap(__LINE__));
  return bp;
}

/**
 * @brief Frees a previously allocated block of memory.
 *
 * Marks the block as free and attempts to coalesce it with neighboring free
 * blocks.
 *
 * @param[in] bp Pointer to the block to free.
 */
void mm_free(void *bp) {
  dbg_requires(mm_checkheap(__LINE__));
  if (bp == NULL) {
    return;
  }

  block_t *block = payload_to_header(bp);
  size_t size = get_size(block);

  // The block should be marked as allocated
  dbg_assert(get_alloc(block));

  // Mark the block as free
  write_block(block, size, false, get_prevalloc(block), get_prevmini(block));

  // Try to coalesce the block with its neighbors
  coalesce_block(block);

  dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief Reallocates a block of memory to a new size.
 *
 * If the new size is larger, allocates a new block and copies data.
 * If size is 0, frees the block.
 *
 * @param[in] ptr Pointer to the existing block.
 * @param[in] size The new size requested.
 * @return Pointer to the reallocated block, or NULL on failure.
 */
void *mm_realloc(void *ptr, size_t size) {
  block_t *block = payload_to_header(ptr);
  size_t copysize;
  void *newptr;

  // If size == 0, then free block and return NULL
  if (size == 0) {
    mm_free(ptr);
    return NULL;
  }

  // If ptr is NULL, then equivalent to malloc
  if (ptr == NULL) {
    return mm_malloc(size);
  }

  // Otherwise, proceed with reallocation
  newptr = mm_malloc(size);

  // If malloc fails, the original block is left untouched
  if (newptr == NULL) {
    return NULL;
  }

  // Copy the old data
  copysize = get_payload_size(block); // gets size of old payload
  if (size < copysize) {
    copysize = size;
  }
  memcpy(newptr, ptr, copysize);

  // Free the old block
  mm_free(ptr);

  return newptr;
}

/**
 * @brief Allocates memory for an array of elements and initializes them to
 * zero.
 *
 * @param[in] elements Number of elements.
 * @param[in] size Size of each element.
 * @return Pointer to the allocated and zero-initialized memory, or NULL on
 * failure.
 */
void *mm_calloc(size_t elements, size_t size) {
  void *bp;
  size_t asize = elements * size;

  if (elements == 0) {
    return NULL;
  }
  if (asize / elements != size) {
    // Multiplication overflowed
    return NULL;
  }

  bp = mm_malloc(asize);
  if (bp == NULL) {
    return NULL;
  }

  // Initialize all bits to 0
  memset(bp, 0, asize);

  return bp;
}

#ifdef DEMO
int main(void) {
  mem_init(false); // Initialize memory emulation
  printf("Initializing allocator...\n");
  if (!mm_init()) {
    fprintf(stderr, "mm_init failed!\n");
    return 1;
  }
  printf("Allocator initialized.\n");

  printf("Allocating 100 bytes...\n");
  void *p = mm_malloc(100);
  if (!p) {
    fprintf(stderr, "mm_malloc failed!\n");
    return 1;
  }
  printf("Allocated at %p\n", p);

  printf("Freeing %p...\n", p);
  mm_free(p);
  printf("Freed.\n");

  printf("Demo complete.\n");
  return 0;
}
#endif
