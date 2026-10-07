/*
 * SPARC VIS1 run-time feature detection
 *
 * This file is part of libjpeg-turbo-retro and is intentionally limited to
 * VIS1.  Later VIS revisions are not required by this backend.
 */

#include "../jsimdint.h"

#if defined(HAVE_GETAUXVAL)
#include <sys/auxv.h>
#ifndef HWCAP_SPARC_VIS
#define HWCAP_SPARC_VIS  0x00002000UL
#endif
#elif defined(__sun)
#include <sys/types.h>
#include <sys/auxv.h>
#include <sys/auxv_SPARC.h>
#endif


HIDDEN unsigned int
jpeg_simd_cpu_support(void)
{
#if defined(HAVE_GETAUXVAL)
  unsigned long hwcap = getauxval(AT_HWCAP);

  if (hwcap & HWCAP_SPARC_VIS)
    return JSIMD_VIS;
#elif defined(__sun)
  uint_t hwcap = 0;

  if (getisax(&hwcap, 1) > 0 && (hwcap & AV_SPARC_VIS))
    return JSIMD_VIS;
#endif

  /*
   * Do not infer VIS support from the SPARC ISA level.  Old SPARC systems
   * without VIS must remain able to run scalar code.  Unsupported systems can
   * still opt in explicitly with JSIMD_FORCEVIS1=1.
   */
  return JSIMD_NONE;
}
