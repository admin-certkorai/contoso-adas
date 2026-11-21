/* ecu_config.h - build-time configuration for the DMS-1 driver-monitoring ECU.
 *
 * Target: ARM64 (aarch64) Cortex-A automotive SoC on embedded Linux.
 * Owner:  ADAS Perception Firmware team.
 */
#ifndef ECU_CONFIG_H
#define ECU_CONFIG_H

/* ECU / model identity reported over the vehicle diagnostic bus. */
#define ECU_ID              "DMS-1"
#define ECU_HW_REV          "revC"
#define MODEL_VERSION       "dms-mlp-v1.1.0"

/* Inference cadence on the camera pipeline (30 fps sensor). */
#define FRAME_PERIOD_MS     33

/* Decision threshold: driver-state confidence below this raises a HMI warning
 * chime + escalates to the forward-collision-warning arbiter. */
#define WARN_CONFIDENCE     0.60f

/* OTA and diagnostic access. The per-ECU diagnostic key is provisioned into the HSM at end of
 * line and read at runtime; no key is compiled into the firmware. */
#define OTA_UPDATE_ENDPOINT "https://ota.fleet.internal/dms1/manifest"

#endif /* ECU_CONFIG_H */
