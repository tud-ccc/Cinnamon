/* 
 * Copyright EPFL 2021
 * Joshua Klein
 * 
 * This file contains wrappers for in-line assembly access to the AIMC core via
 * custom instructions.  The custom instruction definitions are included and
 * match those found in the gem5 model.
 *
 */

#ifndef __AIMC_INTRINSICS_HH__
#define __AIMC_INTRINSICS_HH__

#include <stdint.h>

/* CM Core Process (MVM)
 * Instruction format: |____Opcode___|__rm__|_X|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0000_0001_000|0_0000|_0|000_00|00_000|0_0000|
 * Hex layout:         |__0____1____0|____0_|__|_0____|0____0|____0_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Arguments: None.
 */
inline void
aimcProcess(int tid = 0)
{
    __asm__ volatile(
        ".long 0x01000000;"
        :
        :
        :
    );

    return;
}

/* CM Core Input Memory Queue
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0010_0001_000|0_1000|_1|000_00|00_000|0_0000|
 * Hex layout:         |__2____1____0|____8_|__|_8____|0____0|____0_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rm = QUEUE_MAX input values packed as <val7, val6, ..., val0> for
 *         queueing.
 */
inline void
aimcQueue(uint64_t rm, int tid = 0)
{
    __asm__ volatile(
        "MOV X9, %[input_j];"
        ".long 0x21088000;"
        :
        : [input_j] "r" (rm)
        : "x9"
    );

    return;
}

/* CM Core Output Memory Dequeue
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0010_0001_000|0_0000|_0|000_00|00_000|0_1010|
 * Hex layout:         |__2____1____0|____0_|__|_0____|0____0|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = QUEUE_MAX output values packed as <val7, val6, ..., val0>.
 */
inline uint64_t
aimcDequeue(int tid = 0)
{
    uint64_t res;

    __asm__ volatile(
        ".long 0x2100000A;"
        "MOV %[output], X10;"
        : [output] "=r" (res)
        :
        : "x10"
    );

    return res;
}

/* CM Core Parameter Read
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0100_0001_000|0_1000|_1|000_00|01_001|0_1010|
 * Hex layout:         |__4____1____0|____8_|__|_8____|1____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Parameter value.
 * -- rm = Parameter x index.
 * -- rn = Parameter y index.
 */
inline uint64_t
aimcParamRead(uint64_t rm, uint64_t rn, uint64_t ra, int tid = 0)
{
    uint64_t res;

    __asm__ volatile(
        "MOV X9, %[input_j];"
        "MOV X7, %[input_k];"
        ".long 0x4108812A;"
        "MOV %[output], X10;"
        : [output] "=r" (res)
        : [input_j] "r" (rm), [input_k] "r" (rn)
        : "x7", "x9", "x10"
    );

    return res;
}

/* CM Core Parameter Write
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0100_0001_000|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__4____1____0|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Success code.
 * -- rm = Parameter x index.
 * -- ra = Parameter value.
 * -- rn = Parameter y index.
 */
inline uint64_t
aimcParamWrite(uint64_t rm, uint64_t rn, uint64_t ra, int tid = 0)
{
    uint64_t res;

    __asm__ volatile(
        "MOV X8, %[input_i];"
        "MOV X9, %[input_j];"
        "MOV X7, %[input_k];"
        ".long 0x41081D2A;"
        "MOV %[output], X10;"
        : [output] "=r" (res)
        : [input_i] "r" (ra), [input_j] "r" (rm), [input_k] "r" (rn)
        : "x7", "x8", "x9", "x10"
    );

    return res;
}

#endif // __AIMC_INTRINSICS_HH__
