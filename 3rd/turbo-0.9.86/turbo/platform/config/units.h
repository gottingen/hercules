/*-----------------------------------------------------------------------------
 * eaunits.h
 *
 * Copyright (c) Electronic Arts Inc. All rights reserved.
 * Copyright (c) Jeff.Li. All rights reserved.
 *---------------------------------------------------------------------------*/


#ifndef TURBO_PLATFORM_CONFIG_UNITS_H_
#define TURBO_PLATFORM_CONFIG_UNITS_H_


// Defining common SI unit macros.
//
// The mebibyte is a multiple of the unit byte for digital information. Technically a
// megabyte (MB) is a power of ten, while a mebibyte (MiB) is a power of two,
// appropriate for binary machines. Many Linux distributions use the unit, but it is
// not widely acknowledged within the industry or media.
// Reference: https://en.wikipedia.org/wiki/Mebibyte
//
// Examples:
// 	auto size1 = TURBO_KILOBYTE(16);
// 	auto size2 = TURBO_MEGABYTE(128);
// 	auto size3 = TURBO_MEBIBYTE(8);
// 	auto size4 = TURBO_GIBIBYTE(8);

// define byte for completeness
#define TURBO_BYTE(x) (x)

// Decimal SI units
#define TURBO_KILOBYTE(x) (size_t(x) * 1000)
#define TURBO_MEGABYTE(x) (size_t(x) * 1000 * 1000)
#define TURBO_GIGABYTE(x) (size_t(x) * 1000 * 1000 * 1000)
#define TURBO_TERABYTE(x) (size_t(x) * 1000 * 1000 * 1000 * 1000)
#define TURBO_PETABYTE(x) (size_t(x) * 1000 * 1000 * 1000 * 1000 * 1000)
#define TURBO_EXABYTE(x)  (size_t(x) * 1000 * 1000 * 1000 * 1000 * 1000 * 1000)

// Binary SI units
#define TURBO_KIBIBYTE(x) (size_t(x) * 1024)
#define TURBO_MEBIBYTE(x) (size_t(x) * 1024 * 1024)
#define TURBO_GIBIBYTE(x) (size_t(x) * 1024 * 1024 * 1024)
#define TURBO_TEBIBYTE(x) (size_t(x) * 1024 * 1024 * 1024 * 1024)
#define TURBO_PEBIBYTE(x) (size_t(x) * 1024 * 1024 * 1024 * 1024 * 1024)
#define TURBO_EXBIBYTE(x) (size_t(x) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024)

#endif  // TURBO_PLATFORM_CONFIG_UNITS_H_




