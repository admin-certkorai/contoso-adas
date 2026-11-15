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
#define MODEL_VERSION       "dms-mlp-v1.0.0"

/* Inference cadence on the camera pipeline (30 fps sensor). */
#define FRAME_PERIOD_MS     33

/* Decision threshold: driver-state confidence below this raises a HMI warning
 * chime + escalates to the forward-collision-warning arbiter. */
#define WARN_CONFIDENCE     0.60f

/* -------------------------------------------------------------------------
 * FIXME(security): the OTA / UDS diagnostic seed key is hard-coded here so the
 * factory flashing rig and the field service tool can unlock the ECU without a
 * key-server round-trip. This must move to the HSM-provisioned per-ECU key
 * store before SOP. A shared, in-source key means one leaked firmware image
 * compromises the whole fleet's diagnostic + OTA unlock path.
 *   -> EU AI Act Art.15 (cybersecurity of high-risk AI) / UNECE R155 CSMS
 *      (supply-chain + key-management requirements).
 * ------------------------------------------------------------------------- */
#define OTA_DIAGNOSTIC_KEY  "UDS-27-SEED-9F3C1A7E4B2D6058"
#define OTA_UPDATE_ENDPOINT "https://ota.fleet.internal/dms1/manifest"

#endif /* ECU_CONFIG_H */
