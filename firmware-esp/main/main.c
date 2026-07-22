#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tmf8829.h"
#include "tmf8829_image.h"

#define LIGHTRANGER_INT_GPIO GPIO_NUM_4
#define LIGHTRANGER_EN_GPIO  GPIO_NUM_5
#define LIGHTRANGER_I2C_HZ   1000000U
/*
 * Poll cadence for the result INT. This MUST be >= one FreeRTOS tick, otherwise
 * pdMS_TO_TICKS() rounds to 0 and vTaskDelay() never blocks -- the main task then
 * starves the priority-0 IDLE task and the task watchdog fires after 5 s. At the
 * configured 100 Hz tick, one tick is 10 ms, so 10 ms is the smallest real delay.
 * 10 ms poll keeps up with the ~40 Hz (25 ms) frame cadence below; the INT stays
 * asserted until the FIFO is read, so a slightly coarse poll never drops a frame.
 */
#define RESULT_POLL_MS       10U

static const char *TAG = "TMF8829_APP";
static tmf8829Driver s_sensor;

/*
 * Target rate. This value is only a FLOOR on the frame interval: the sensor
 * cannot go faster than its integration time (set by KILO_ITERATIONS in the
 * loaded 32x32 config) plus the fixed per-frame I2C readout + ASCII format cost.
 * If those exceed 25 ms the device simply runs slower than 40 Hz -- asking for a
 * shorter period never forces it, so lowering this is a safe experiment.
 *
 *   25 ms -> 40 Hz target      22 ms -> ~45 Hz      33 ms -> 30 Hz (known-good)
 *
 * MEASURE the achieved frames/sec at the host after flashing. If it plateaus
 * below the target, the readout/format cost is the wall (transmit less, e.g.
 * binary instead of CSV). If integration time is the limit, lowering
 * KILO_ITERATIONS raises the ceiling but trades away SNR / max-range /
 * confidence. A non-zero period also keeps the sensor in autonomous cyclic mode,
 * which the driver expects (it sets cyclicRunning); period 0 free-running is the
 * theoretical max but gives a variable rate and starved the idle task -> wdt.
 */
#define MEASUREMENT_PERIOD_MS 25U

#define TMF8829_CFG_INDEX(reg) \
    ((uint16_t)((reg) - TMF8829_CFG_PERIOD_MS_LSB))

static esp_err_t tmf8829_start_32x32(void)
{
    int8_t status;

    configurePins(&s_sensor);
    i2cOpen(&s_sensor, LIGHTRANGER_I2C_HZ);

    tmf8829Initialise(&s_sensor);
    tmf8829SetLogLevel(&s_sensor, TMF8829_LOG_LEVEL_RESULTS);

    /* Force a clean hardware reset and return to the default I2C address. */
    tmf8829Disable(&s_sensor);
    delayInMicroseconds(CAP_DISCHARGE_TIME_MS * 1000U);

    tmf8829Enable(&s_sensor);
    delayInMicroseconds(ENABLE_TIME_MS * 1000U);

    tmf8829ClkCorrection(&s_sensor, 1);
    tmf8829PowerUp(&s_sensor);

    if (!tmf8829IsCpuReady(&s_sensor, CPU_READY_TIME_MS)) {
        ESP_LOGE(TAG, "TMF8829 CPU did not become ready. Check 3V3, GND, EN, SDA/SCL, and COMM SEL jumpers.");
        return ESP_ERR_TIMEOUT;
    }

    /* We are using I2C, so disable the SPI bootloader interface. */
    status = tmf8829BootloaderCmdSpiOff(&s_sensor);
    if (status != BL_SUCCESS_OK) {
        ESP_LOGE(TAG, "Failed to switch the TMF8829 bootloader to I2C mode: %d", status);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Downloading TMF8829 RAM firmware (%" PRIu32 " bytes)...",
             (uint32_t)tmf8829_image_length);

    status = tmf8829DownloadFirmware(
        &s_sensor,
        (uint32_t)tmf8829_image_start,
        tmf8829_image,
        (int32_t)tmf8829_image_length,
        1 /* use FIFO download */
    );
    if (status != BL_SUCCESS_OK) {
        ESP_LOGE(TAG, "Firmware download failed: %d", status);
        return ESP_FAIL;
    }

    status = tmf8829ReadDeviceInfo(&s_sensor);
    if (status != APP_SUCCESS_OK) {
        ESP_LOGE(TAG, "Could not read device information: %d", status);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG,
             "Firmware %u.%u.%u.%u | chip %u.%u | serial 0x%08" PRIX32,
             s_sensor.device.appVersion[0],
             s_sensor.device.appVersion[1],
             s_sensor.device.appVersion[2],
             s_sensor.device.appVersion[3],
             s_sensor.device.chipVersion[0],
             s_sensor.device.chipVersion[1],
             (uint32_t)s_sensor.device.deviceSerialNumber);

    /* Load the official 32x32 default measurement configuration. */
    status = tmf8829Command(
        &s_sensor,
        TMF8829_CMD_STAT__cmd_stat__CMD_LOAD_CFG_32X32
    );

    status = tmf8829GetConfiguration(&s_sensor);
    if (status != APP_SUCCESS_OK) {
        ESP_LOGE(TAG, "Could not read TMF8829 configuration: %d", status);
        return ESP_FAIL;
    }

    tmf8829SetUint16(
        MEASUREMENT_PERIOD_MS,
        &s_sensor.config[
            TMF8829_CFG_INDEX(TMF8829_CFG_PERIOD_MS_LSB)
        ]
    );

    status = tmf8829SetConfiguration(&s_sensor);
    if (status != APP_SUCCESS_OK) {
        ESP_LOGE(TAG, "Could not write TMF8829 configuration: %d", status);
        return ESP_FAIL;
    }



    if (status != APP_SUCCESS_OK) {
        ESP_LOGE(TAG, "Could not load the 32x32 configuration: %d", status);
        return ESP_FAIL;
    }

    tmf8829ClrAndEnableInterrupts(&s_sensor, TMF8829_APP_INT_RESULTS);

    status = tmf8829StartMeasurement(&s_sensor);
    if (status != APP_SUCCESS_OK) {
        ESP_LOGE(TAG, "Could not start measurement: %d", status);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "32x32 ranging started (40 Hz target). Raw official-driver result frames follow.");
    return ESP_OK;
}

void app_main(void)
{
    ESP_LOGI(TAG, "RingFusion LightRanger 14 / TMF8829 bring-up");
    ESP_LOGI(TAG, "Pins: SDA=GPIO6, SCL=GPIO7, INT=GPIO4, EN=GPIO5");

    ESP_ERROR_CHECK(tmf8829_start_32x32());

    for (;;) {
        /* INT is active-low. Polling keeps the first port simple and reliable. */
        if (gpio_get_level(LIGHTRANGER_INT_GPIO) == 0) {
            const uint8_t irq = tmf8829GetAndClrInterrupts(
                &s_sensor,
                TMF8829_APP_INT_RESULTS
            );

            if ((irq & TMF8829_APP_INT_RESULTS) != 0U) {
                const int8_t status = tmf8829ReadResults(&s_sensor);
                if (status != APP_SUCCESS_OK) {
                    ESP_LOGW(TAG, "Result interrupt was set, but result read returned %d", status);
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(RESULT_POLL_MS));
    }
}
