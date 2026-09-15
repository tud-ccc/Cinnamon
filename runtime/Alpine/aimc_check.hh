/*
 * Copyright EPFL 2021
 * Joshua Klein
 *
 * This file contains classes functions for which we can emulate and check
 * the code in the associated gem5 model.  This model reflects an updated
 * model optimized for queueing and dequeueing: we assume that the control unit
 * of the AIMC core has two internal counters that can be used for the
 * addressing of the input and output memories.
 *
 * The workflow of the core model is as follows:
 * - 1. Load your input vector into the input memory of the AIMC core model:
 * - - a. Scale your input values to 8-bit and apply a to make the values
 *        unsigned.
 * - - b. Write up to scaled inputs to register R, such that it is packed as
 *        <<input 3>, <input 2>, <input 1>, <input 0>> (note this example is
 *        for use with a 32-bit register).
 * - - c. Call the CM_QUEUE custom instruction so that it takes the values in
 *        R, and places them in the input memory.  The first time this
 *        instruction is called, the AIMC control unit's input memory counter
 *        is set to 0, and so this value will be incremeneted by 4 in order to
 *        mean that 4 values have been "queued" in the input memory.  The 4
 *        packed values will reside in the first four places of the AIMC core's
 *        input memory.
 * - - d. Call the CM_QUEUE custom instrution enough times so that the entire
 *        input vector resides in the input memory.
 * - 2. Perform the MVM in the AIMC CORE MODEL: Call the CM_PROCESS custom
 *      instruction once.  This will perform A * B = C where A is the matrix
 *      held in the AIMC core, B is the vector in the input memory, and C is
 *      the output vector placed in the output memory.  The CM_PROCESS custom
 *      also resets both the input and output memory control unit queueing
 *      counters as well as the input memory.
 * - 3. Read the output vector from the output memory of the AIMC core model:
 * - - a. Call the CM_DEQUEUE custom instruction to read from the output memory
 *        into register R.  Assuming a 32-bit register, R will contain 4 and
 *        packed unsigned 8-bit integers from the first 4 output memory cells.
 *        The output memory control unit dequeueing counter will also increment
 *        by 4.
 * - - b. Repeat the CM_DEQUEUE instruction call until the entire output memory
 *        vector is retrieved.  Note that the output values will need to be
 *        scaled back up from 8-bit.
 *
 */

#ifndef __AIMC_CHECK_HH__
#define __AIMC_CHECK_HH__

#include <stdint.h>

//////////////////////////
// Core-Local AIMC Core //
//////////////////////////
// Allow overriding the emulated AIMC array size at compile time to speed up
// SE runs. Defaults remain 4000x4000 if not specified. Provide runtime knobs
// too, so a client can reconfigure via the C runtime (see alpine_rt.cc).
#ifndef AIMC_HEIGHT
#define AIMC_HEIGHT 2000
#endif
#ifndef AIMC_WIDTH
#define AIMC_WIDTH 2000
#endif
extern int gAimcCfgHeight;
extern int gAimcCfgWidth;

struct AnalogComputationalMemoryCore {
  // Constructor.
  AnalogComputationalMemoryCore()
      : crossbarHeight(gAimcCfgHeight), crossbarWidth(gAimcCfgWidth),
        crossbar(new int8_t[crossbarWidth * crossbarHeight]),
        inputMemory(new int8_t[crossbarHeight]),
        outputMemory(new int8_t[crossbarWidth]), inputMemoryCounter(0),
        outputMemoryCounter(0), vectorization(4) {
    for (int i = 0; i < crossbarHeight; i++) {
      inputMemory[i] = 0;
      for (int j = 0; j < crossbarWidth; j++) {
        crossbar[(i * crossbarWidth) + j] = 0;
      }
    }

    for (int i = 0; i < crossbarWidth; i++) {
      outputMemory[i] = 0;
    }
  }

  AnalogComputationalMemoryCore(int height, int width)
      : crossbarHeight(height > 0 ? height : gAimcCfgHeight),
        crossbarWidth(width > 0 ? width : gAimcCfgWidth),
        crossbar(new int8_t[crossbarWidth * crossbarHeight]),
        inputMemory(new int8_t[crossbarHeight]),
        outputMemory(new int8_t[crossbarWidth]), inputMemoryCounter(0),
        outputMemoryCounter(0), vectorization(4) {
    for (int i = 0; i < crossbarHeight; i++) {
      inputMemory[i] = 0;
      for (int j = 0; j < crossbarWidth; j++) {
        crossbar[(i * crossbarWidth) + j] = 0;
      }
    }

    for (int i = 0; i < crossbarWidth; i++) {
      outputMemory[i] = 0;
    }
  }

  const int crossbarHeight; // Height of input memory too.
  const int crossbarWidth;  // Width of output memory too.
  int8_t *crossbar;         // Parameters crossbar.
  int8_t *inputMemory;      // Input memory (pre-DAC).
  int8_t *outputMemory;     // Output memory (post-ADC).
  int inputMemoryCounter;   // Index into input memory.
  int outputMemoryCounter;  // Index into output memory.
  const int vectorization;  // How many values do we queue/dequeue?
};

////////////////
// AIMC Cores //
////////////////
class AnalogComputationalMemory {
private:
  AnalogComputationalMemoryCore **cores;
  int coreCount;

public:
  // All aimc cores (per thread).
  AnalogComputationalMemory() {
    coreCount = 1;
    cores = new AnalogComputationalMemoryCore *[coreCount];
    cores[0] = new AnalogComputationalMemoryCore();
  }

  AnalogComputationalMemory(int numOfCores) {
    coreCount = (numOfCores > 0) ? numOfCores : 1;
    cores = new AnalogComputationalMemoryCore *[coreCount];
    for (int i = 0; i < coreCount; i++) {
      cores[i] = new AnalogComputationalMemoryCore();
    }
  }

  AnalogComputationalMemory(int numOfCores, int height, int width) {
    coreCount = (numOfCores > 0) ? numOfCores : 1;
    cores = new AnalogComputationalMemoryCore *[coreCount];
    for (int i = 0; i < coreCount; i++) {
      cores[i] = new AnalogComputationalMemoryCore(height, width);
    }
  }

  ~AnalogComputationalMemory() {
    for (int i = 0; i < coreCount; i++) {
      AnalogComputationalMemoryCore *core = cores[i];
      delete[] core->inputMemory;
      delete[] core->outputMemory;
      delete[] core->crossbar;
      delete core;
    }
    delete[] cores;
  }

  // Debug helpers (no-op in freestanding build to avoid iostream)
  void printInputMemory(int, int = 10) {}
  void printOutputMemory(int, int = 10) {}
  void printCrossbar(int, int = 10) {}

  // AIMC operations.
  void aimcProcess(int tid = 0) {
    int height = cores[tid]->crossbarHeight;
    int width = cores[tid]->crossbarWidth;

    for (int i = 0; i < width; i++) {
      double acc = 0.0;

      for (int j = 0; j < height; j++) {
        acc += (cores[tid]->inputMemory[j]) *
               cores[tid]->crossbar[(j * width) + i];
      }

      if (acc > 127)
        cores[tid]->outputMemory[i] = 127;
      else if (acc < -128)
        cores[tid]->outputMemory[i] = -128;
      else
        cores[tid]->outputMemory[i] = (int8_t)acc;
    }

    // Refresh input memory and queue counters.
    for (int i = 0; i < height; i++) {
      cores[tid]->inputMemory[i] = 0;
    }

    cores[tid]->inputMemoryCounter = 0;
    cores[tid]->outputMemoryCounter = 0;

    return;
  }

  void aimcQueue(int tid, uint32_t val) {
    int idx = cores[tid]->inputMemoryCounter;
    int length = cores[tid]->vectorization;

    if ((idx + length) <= cores[tid]->crossbarHeight) {
      for (int i = 0; i < length; i++) {
        int8_t currVal = (val >> (8 * i)) & 0xff;
        cores[tid]->inputMemory[idx + i] = currVal;
      }
    }

    cores[tid]->inputMemoryCounter += length;

    return;
  }

  uint32_t aimcDequeue(int tid) {
    uint32_t result = 0;
    int idx = cores[tid]->outputMemoryCounter;
    int length = cores[tid]->vectorization;

    if ((idx + length) <= cores[tid]->crossbarWidth) {
      for (int i = length - 1; i > -1; i--) {
        result <<= 8;
        result |= 0xff & (cores[tid]->outputMemory[idx + i]);
      }
    }

    cores[tid]->outputMemoryCounter += length;

    return result;
  }

  // Debug peek helpers (non-mutating) for SE runs
  int8_t peekInput(int tid, int idx) const {
    if (tid < 0 || tid >= coreCount)
      return 0;
    if (idx < 0 || idx >= cores[tid]->crossbarHeight)
      return 0;
    return cores[tid]->inputMemory[idx];
  }
  int8_t peekOutput(int tid, int idx) const {
    if (tid < 0 || tid >= coreCount)
      return 0;
    if (idx < 0 || idx >= cores[tid]->crossbarWidth)
      return 0;
    return cores[tid]->outputMemory[idx];
  }

  // Direct vector I/O helpers (checker-only convenience)
  void writeInputVector(int tid, const int8_t *src, int len) {
    if (tid < 0 || tid >= coreCount)
      return;
    if (len < 0)
      return;
    int n =
        (len <= cores[tid]->crossbarHeight) ? len : cores[tid]->crossbarHeight;
    for (int i = 0; i < n; ++i) {
      cores[tid]->inputMemory[i] = src[i];
    }
    cores[tid]->inputMemoryCounter = n;
  }
  void readOutputVector(int tid, int8_t *dst, int len) const {
    if (tid < 0 || tid >= coreCount)
      return;
    if (len < 0)
      return;
    int n =
        (len <= cores[tid]->crossbarWidth) ? len : cores[tid]->crossbarWidth;
    for (int i = 0; i < n; ++i) {
      dst[i] = cores[tid]->outputMemory[i];
    }
  }

  int8_t aimcParamRead(int tid, int x, int y) {
    int8_t result = 0;
    int height = cores[tid]->crossbarHeight;
    int width = cores[tid]->crossbarWidth;
    int idx = (y * width) + x; // 1D index

    if (idx < width * height) {
      result = cores[tid]->crossbar[idx];
    }

    return result;
  }

  void aimcParamWrite(int tid, int x, int y, int8_t val) {
    int height = cores[tid]->crossbarHeight;
    int width = cores[tid]->crossbarWidth;
    int idx = (y * width) + x; // 1D index

    if (idx < width * height) {
      cores[tid]->crossbar[idx] = val;
    }

    return;
  }
};

// Singleton accessor and runtime setter (defined in aimc_state.cc).
class AnalogComputationalMemory;
AnalogComputationalMemory &getAimc();
void aimcSetArrayDimsRuntime(int height, int width);

//////////////////////////
// Intrinsics Emulation //
//////////////////////////

/* CM Core Process (MVM)
 * Instruction format: |____Opcode___|__rm__|_X|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0000_0001_100|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__0____1____8|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Arguments: None.
 */
inline uint64_t aimcProcess(int tid = 0) {
  getAimc().aimcProcess(tid);
  return 0;
}

/* CM Core Input Memory Queue
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0010_0001_100|0_1000|_1|001_11|01_001|0_1010|
 * Hex layout:         |__2____1____8|____8_|__|_9____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rm = QUEUE_MAX input values packed as <val7, val6, ..., val0> for
 *         queueing.
 */
inline uint64_t aimcQueue(uint64_t rm, int tid = 0) {
  getAimc().aimcQueue(tid, rm);
  return 0;
}

/* CM Core Output Memory Dequeue
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0010_0001_100|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__2____1____8|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = QUEUE_MAX output values packed as <val7, val6, ..., val0>.
 */
inline uint64_t aimcDequeue(int tid = 0) { return getAimc().aimcDequeue(tid); }

/* CM Core Parameter Read
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0100_0001_100|0_1000|_1|001_11|01_001|0_1010|
 * Hex layout:         |__4____1____8|____8_|__|_9____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Parameter value.
 * -- rm = Parameter x index.
 * -- rn = Parameter y index.
 */
inline uint64_t aimcParamRead(uint64_t rm, uint64_t rn, uint64_t ra,
                              int tid = 0) {
  return getAimc().aimcParamRead(tid, rm, rn);
}

/* CM Core Parameter Write
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0100_0001_100|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__4____1____8|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Success code.
 * -- rm = Parameter x index.
 * -- ra = Parameter value.
 * -- rn = Parameter y index.
 */
inline uint64_t aimcParamWrite(uint64_t rm, uint64_t rn, uint64_t ra,
                               int tid = 0) {
  getAimc().aimcParamWrite(tid, rm, rn, ra);
  return 0;
}

#endif // __AIMC_CHECK_HH__
