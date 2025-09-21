// runtime_all.cc  —  freestanding SE runtime (AArch64)
// - abort via syscall
// - memcpy/memmove/memset
// - bump allocator: __simple_alloc/__simple_free
// - malloc/calloc/realloc/free (+posix_memalign/aligned_alloc) with C ABI
// - C++ global new/delete routed to bump allocator

#include <cstddef>
#include <cstdint>
#include <limits.h> // SIZE_MAX

#ifndef _RUNTIME_H_
#define _RUNTIME_H_

// ======================= abort (AArch64 Linux) =======================
extern "C" __attribute__((noreturn)) void abort(void) {
  register long x0 __asm__("x0") = 134; // exit code for abort
  register long x8 __asm__("x8") = 93;  // SYS_exit
  __asm__ volatile("svc #0" : : "r"(x0), "r"(x8) : "memory");
  __builtin_unreachable();
}

// ======================= minimal mem* =======================
extern "C" void *memcpy(void *dst, const void *src, std::size_t n) {
  auto *d = static_cast<uint8_t *>(dst);
  auto *s = static_cast<const uint8_t *>(src);
  for (std::size_t i = 0; i < n; i++)
    d[i] = s[i];
  return dst;
}

extern "C" void *memmove(void *dst, const void *src, std::size_t n) {
  auto *d = static_cast<uint8_t *>(dst);
  auto *s = static_cast<const uint8_t *>(src);
  if (d == s || n == 0)
    return dst;
  if (d < s) {
    for (std::size_t i = 0; i < n; i++)
      d[i] = s[i];
  } else {
    for (std::size_t i = n; i != 0; i--)
      d[i - 1] = s[i - 1];
  }
  return dst;
}

extern "C" void *memset(void *dst, int c, std::size_t n) {
  auto *d = static_cast<uint8_t *>(dst);
  uint8_t v = static_cast<uint8_t>(c);
  for (std::size_t i = 0; i < n; i++)
    d[i] = v;
  return dst;
}

// ======================= bump allocator =======================
#ifndef SIMPLE_HEAP_SIZE
#define SIMPLE_HEAP_SIZE (128u * 1024u * 1024u)
#endif

static uint8_t __heap_big[SIMPLE_HEAP_SIZE] __attribute__((aligned(16)));
static std::size_t __off_big = 0;

static inline std::size_t align_up(std::size_t v, std::size_t a) {
  return (v + (a - 1)) & ~(a - 1);
}

// C-visible allocator entry points (stable symbols)
extern "C" void *__simple_alloc(unsigned long size) {
  std::size_t s = align_up(static_cast<std::size_t>(size), 16);
  std::size_t n = __off_big + s;
  if (n > SIMPLE_HEAP_SIZE)
    abort();
  void *p = &__heap_big[__off_big];
  __off_big = n;
  return p;
}

extern "C" void __simple_free(void *p) { (void)p; /* no-op */ }

// ======================= malloc family (C ABI) =======================
extern "C" void *malloc(std::size_t size) {
  return __simple_alloc(static_cast<unsigned long>(size));
}

extern "C" void free(void *ptr) {
  __simple_free(ptr); // no-op for bump allocator
}

extern "C" void *calloc(std::size_t nmemb, std::size_t size) {
  if (size != 0 && nmemb > (SIZE_MAX / size))
    abort();
  std::size_t total = nmemb * size;
  void *p = malloc(total);
  if (p)
    memset(p, 0, total);
  return p;
}

extern "C" void *realloc(void *ptr, std::size_t new_size) {
  if (ptr == nullptr)
    return malloc(new_size);
  if (new_size == 0) {
    free(ptr);
    return nullptr;
  }
  // Bump allocator can't shrink/expand in place: return fresh block.
  void *np = malloc(new_size);
  if (!np)
    return nullptr;
  // No copy (unknown old size). Adjust if you track sizes.
  return np;
}

extern "C" int posix_memalign(void **memptr, std::size_t alignment,
                              std::size_t size) {
  if (!memptr)
    return 22; // EINVAL
  if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0)
    return 22;
  std::size_t extra = alignment - 1 + sizeof(void *);
  void *raw = malloc(size + extra);
  if (!raw) {
    *memptr = nullptr;
    return 12;
  } // ENOMEM
  uintptr_t base = (uintptr_t)raw + sizeof(void *);
  uintptr_t aligned = (base + (alignment - 1)) & ~(uintptr_t)(alignment - 1);
  ((void **)aligned)[-1] = raw; // stored for real frees; harmless here
  *memptr = (void *)aligned;
  return 0;
}

extern "C" void *aligned_alloc(std::size_t alignment, std::size_t size) {
  void *p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0)
    return nullptr;
  return p;
}

// ======================= C++ global new/delete =======================
void *operator new(std::size_t size) noexcept { return __simple_alloc(size); }
void operator delete(void *p) noexcept { __simple_free(p); }
void *operator new[](std::size_t size) noexcept { return __simple_alloc(size); }
void operator delete[](void *p) noexcept { __simple_free(p); }
void operator delete(void *p, std::size_t) noexcept { __simple_free(p); }
void operator delete[](void *p, std::size_t) noexcept { __simple_free(p); }

// (Optional) tiny C++ runtime stubs you may want in freestanding builds:
// extern "C" void __cxa_pure_virtual(void) { abort(); }

#endif