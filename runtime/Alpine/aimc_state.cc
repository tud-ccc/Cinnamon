// Single definition for AIMC checker state shared across all TUs.
#include "aimc_check.hh"

// Provide unique storage, not per-TU statics.
int gAimcCfgHeight = AIMC_HEIGHT;
int gAimcCfgWidth = AIMC_WIDTH;
static AnalogComputationalMemory *g_aimc_ptr = nullptr;

AnalogComputationalMemory &getAimc() {
  if (!g_aimc_ptr) {
    g_aimc_ptr =
        new AnalogComputationalMemory(8, gAimcCfgHeight, gAimcCfgWidth);
  }
  return *g_aimc_ptr;
}

void aimcSetArrayDimsRuntime(int height, int width) {
  if (height > 0)
    gAimcCfgHeight = height;
  if (width > 0)
    gAimcCfgWidth = width;
  if (g_aimc_ptr) {
    delete g_aimc_ptr;
    g_aimc_ptr =
        new AnalogComputationalMemory(8, gAimcCfgHeight, gAimcCfgWidth);
  }
}
