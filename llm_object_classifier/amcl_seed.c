/* AMCL Jazzy seeds drand48 from wall time. Override seeding only, equally for A/B. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
void srand48(long ignored) {
    (void)ignored;
    void (*original)(long) = dlsym(RTLD_NEXT, "srand48");
    const char *value = getenv("LILOCBENCH_SEED");
    original(value ? strtol(value, NULL, 10) : 42);
}
