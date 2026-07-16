#include "tmf8829_shim.h"
#include "tmf8829.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"

static const char *TAG = "TMF8829_SHIM";
static i2c_master_bus_handle_t s_bus = NULL;
static i2c_master_dev_handle_t s_dev = NULL;
static uint8_t s_dev_addr = 0xFF;
static uint32_t s_i2c_hz = 400000U;
static void (*s_irq_handler)(void) = NULL;
static bool s_result_line_open = false;
static bool s_hist_line_open = false;

static int8_t map_i2c_error(esp_err_t err)
{
    if (err == ESP_OK) {
        return I2C_SUCCESS;
    }
    if (err == ESP_ERR_TIMEOUT) {
        return I2C_ERR_TIMEOUT;
    }
    if (err == ESP_ERR_INVALID_SIZE || err == ESP_ERR_NO_MEM) {
        return I2C_ERR_DATA_TOO_LONG;
    }
    return I2C_ERR_OTHER;
}

static esp_err_t ensure_device(uint8_t slave_addr)
{
    if (s_bus == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (s_dev != NULL && s_dev_addr == slave_addr) {
        return ESP_OK;
    }

    if (s_dev != NULL) {
        ESP_ERROR_CHECK_WITHOUT_ABORT(i2c_master_bus_rm_device(s_dev));
        s_dev = NULL;
        s_dev_addr = 0xFF;
    }

    const i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = slave_addr,
        .scl_speed_hz = s_i2c_hz,
        .scl_wait_us = 20000,
    };

    esp_err_t err = i2c_master_bus_add_device(s_bus, &dev_cfg, &s_dev);
    if (err == ESP_OK) {
        s_dev_addr = slave_addr;
    }
    return err;
}

void delayInMicroseconds(uint32_t wait)
{
    esp_rom_delay_us(wait);
}

uint32_t getSysTick(void)
{
    return (uint32_t)esp_timer_get_time();
}

uint8_t readProgramMemoryByte(uint32_t address)
{
    return *(const uint8_t *)(uintptr_t)address;
}

void enablePinHigh(void *dptr)
{
    (void)dptr;
    gpio_set_level((gpio_num_t)ENABLE_PIN, 1);
}

void enablePinLow(void *dptr)
{
    (void)dptr;
    gpio_set_level((gpio_num_t)ENABLE_PIN, 0);
}

void configurePins(void *dptr)
{
    (void)dptr;

    const gpio_config_t en_cfg = {
        .pin_bit_mask = 1ULL << ENABLE_PIN,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&en_cfg));
    gpio_set_level((gpio_num_t)ENABLE_PIN, 0);

    const gpio_config_t int_cfg = {
        .pin_bit_mask = 1ULL << INTERRUPT_PIN,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ESP_ERROR_CHECK(gpio_config(&int_cfg));
}

void i2cOpen(void *dptr, uint32_t i2cClockSpeedInHz)
{
    (void)dptr;

    if (s_bus != NULL) {
        return;
    }

    s_i2c_hz = i2cClockSpeedInHz;

    const i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = GPIO_NUM_6,
        .scl_io_num = GPIO_NUM_7,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .intr_priority = 0,
        .trans_queue_depth = 0,
        .flags.enable_internal_pullup = false,
        .flags.allow_pd = false,
    };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &s_bus));
    ESP_LOGI(TAG, "I2C initialized: SDA=GPIO6 SCL=GPIO7 clock=%" PRIu32 " Hz", s_i2c_hz);
}

void i2cClose(void *dptr)
{
    (void)dptr;

    if (s_dev != NULL) {
        ESP_ERROR_CHECK_WITHOUT_ABORT(i2c_master_bus_rm_device(s_dev));
        s_dev = NULL;
        s_dev_addr = 0xFF;
    }
    if (s_bus != NULL) {
        ESP_ERROR_CHECK_WITHOUT_ABORT(i2c_del_master_bus(s_bus));
        s_bus = NULL;
    }
}

void printChar(char c) { putchar((unsigned char)c); }
void printInt(int32_t i) { printf("%" PRId32, i); }
void printUint(uint32_t i) { printf("%" PRIu32, i); }
void printUintHex(uint32_t i) { printf("%" PRIX32, i); }
void printStr(char *str) { fputs(str != NULL ? str : "(null)", stdout); }
void printConstStr(const char *str) { fputs(str != NULL ? str : "(null)", stdout); }
void printLn(void) { putchar('\n'); fflush(stdout); }

int8_t txReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
             uint16_t toTx, const uint8_t *txData)
{
    return i2cTxReg(dptr, slaveAddr, regAddr, toTx, txData);
}

int8_t rxReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
             uint16_t toRx, uint8_t *rxData)
{
    return i2cRxReg(dptr, slaveAddr, regAddr, toRx, rxData);
}

int8_t i2cTxReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
                uint16_t toTx, const uint8_t *txData)
{
    (void)dptr;

    esp_err_t err = ensure_device(slaveAddr);
    if (err != ESP_OK) {
        return map_i2c_error(err);
    }

    uint16_t remaining = toTx;
    const uint8_t *src = txData;
    uint8_t current_reg = regAddr;

    do {
        const uint16_t chunk = remaining > ESP_IDF_I2C_CHUNK_SIZE
                                 ? ESP_IDF_I2C_CHUNK_SIZE
                                 : remaining;
        uint8_t frame[ESP_IDF_I2C_CHUNK_SIZE + 1U];
        frame[0] = current_reg;
        if (chunk > 0U) {
            memcpy(&frame[1], src, chunk);
        }

        err = i2c_master_transmit(s_dev, frame, (size_t)chunk + 1U, 1000);
        if (err != ESP_OK) {
            return map_i2c_error(err);
        }

        remaining = (uint16_t)(remaining - chunk);
        src += chunk;

        if (((uint16_t)current_reg + chunk) >= 0xFFU) {
            current_reg = 0xFFU;
        } else {
            current_reg = (uint8_t)(current_reg + chunk);
        }
    } while (remaining > 0U);

    return I2C_SUCCESS;
}

int8_t i2cRxReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
                uint16_t toRx, uint8_t *rxData)
{
    (void)dptr;

    if (toRx == 0U) {
        return I2C_SUCCESS;
    }

    esp_err_t err = ensure_device(slaveAddr);
    if (err != ESP_OK) {
        return map_i2c_error(err);
    }

    uint16_t remaining = toRx;
    uint8_t *dst = rxData;
    uint16_t chunk = remaining > ESP_IDF_I2C_CHUNK_SIZE
                       ? ESP_IDF_I2C_CHUNK_SIZE
                       : remaining;

    err = i2c_master_transmit_receive(s_dev, &regAddr, 1U, dst, chunk, 1000);
    if (err != ESP_OK) {
        return map_i2c_error(err);
    }

    remaining = (uint16_t)(remaining - chunk);
    dst += chunk;

    while (remaining > 0U) {
        chunk = remaining > ESP_IDF_I2C_CHUNK_SIZE
                  ? ESP_IDF_I2C_CHUNK_SIZE
                  : remaining;
        err = i2c_master_receive(s_dev, dst, chunk, 1000);
        if (err != ESP_OK) {
            return map_i2c_error(err);
        }
        remaining = (uint16_t)(remaining - chunk);
        dst += chunk;
    }

    return I2C_SUCCESS;
}

int8_t i2cTxRx(void *dptr, uint8_t slaveAddr, uint16_t toTx,
               const uint8_t *txData, uint16_t toRx, uint8_t *rxData)
{
    (void)dptr;

    if (toTx == 0U && toRx == 0U) {
        return I2C_SUCCESS;
    }

    esp_err_t err = ensure_device(slaveAddr);
    if (err != ESP_OK) {
        return map_i2c_error(err);
    }

    if (toTx > 0U && toRx > 0U) {
        err = i2c_master_transmit_receive(s_dev, txData, toTx, rxData, toRx, 1000);
    } else if (toTx > 0U) {
        err = i2c_master_transmit(s_dev, txData, toTx, 1000);
    } else {
        err = i2c_master_receive(s_dev, rxData, toRx, 1000);
    }

    return map_i2c_error(err);
}

void inputOpen(uint32_t baudrate) { (void)baudrate; }
void inputClose(void) {}
int8_t inputGetKey(char *c)
{
    if (c != NULL) {
        *c = 0;
    }
    return 0;
}

void pinOutput(uint8_t pin)
{
    gpio_set_direction((gpio_num_t)pin, GPIO_MODE_OUTPUT);
}

void pinInput(uint8_t pin)
{
    gpio_set_direction((gpio_num_t)pin, GPIO_MODE_INPUT);
}

static void IRAM_ATTR gpio_isr_adapter(void *arg)
{
    (void)arg;
    if (s_irq_handler != NULL) {
        s_irq_handler();
    }
}

void setInterruptHandler(void (*handler)(void))
{
    s_irq_handler = handler;
    esp_err_t err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_ERROR_CHECK(err);
    }
    ESP_ERROR_CHECK(gpio_isr_handler_add((gpio_num_t)INTERRUPT_PIN, gpio_isr_adapter, NULL));
}

void clrInterruptHandler(void)
{
    ESP_ERROR_CHECK_WITHOUT_ABORT(gpio_isr_handler_remove((gpio_num_t)INTERRUPT_PIN));
    s_irq_handler = NULL;
}

void disableInterrupts(void) {}
void enableInterrupts(void) {}

static void print_raw_bytes(const uint8_t *data, uint16_t len)
{
    for (uint16_t i = 0; i < len; ++i) {
        printf("%u", (unsigned)data[i]);
        if (i + 1U < len) {
            putchar(',');
        }
    }
}

void handleReceivedFrameHeaderData(void *dptr, uint8_t *data)
{
    (void)dptr;
    if (s_result_line_open) {
        putchar('\n');
    }
    fputs("TMF8829_FRAME_HEADER,", stdout);
    print_raw_bytes(data, TMF8829_PRE_HEADER_SIZE + TMF8829_FRAME_HEADER_SIZE);
    fputs(",PAYLOAD,", stdout);
    s_result_line_open = true;
}

void handleReceivedResultData(void *dptr, uint8_t *data, uint16_t size)
{
    (void)dptr;
    if (!s_result_line_open) {
        fputs("TMF8829_RESULT,", stdout);
        s_result_line_open = true;
    }
    print_raw_bytes(data, size);
    putchar(',');
}

void handleReceivedResultDataEnd(void *dptr)
{
    (void)dptr;
    if (s_result_line_open) {
        putchar('\n');
        fflush(stdout);
        s_result_line_open = false;
    }
}

void handleReceivedHistogramData(void *dptr, uint8_t *data, uint16_t size)
{
    (void)dptr;
    if (!s_hist_line_open) {
        fputs("TMF8829_HISTOGRAM,", stdout);
        s_hist_line_open = true;
    }
    print_raw_bytes(data, size);
    putchar(',');
}

void handleReceivedHistogramDataEnd(void *dptr)
{
    (void)dptr;
    if (s_hist_line_open) {
        putchar('\n');
        fflush(stdout);
        s_hist_line_open = false;
    }
}

void printResultHeader(void *dptr, uint8_t *data, uint8_t len)
{
    (void)dptr;
    print_raw_bytes(data, len);
}

void printResults(void *dptr, uint8_t *data, uint16_t len)
{
    (void)dptr;
    print_raw_bytes(data, len);
}

void printHistogram(void *dptr, uint8_t *data, uint16_t len)
{
    (void)dptr;
    print_raw_bytes(data, len);
}
