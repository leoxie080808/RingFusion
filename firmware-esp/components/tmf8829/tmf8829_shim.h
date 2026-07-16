#ifndef TMF8829_SHIM_H
#define TMF8829_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ESP-IDF port configuration for the Waveshare ESP32-C6 board. */
#define DATA_BUFFER_SIZE 8192
#define ESP_IDF_I2C_CHUNK_SIZE 128

#define ENABLE_PIN 5
#define INTERRUPT_PIN 4
#define USE_INTERRUPT_TO_TRIGGER_READ 0
#define TRIGGER_INTERRUPT_PIN INTERRUPT_PIN

#define HOST_TICKS_PER_1000_US 1000
#define TMF8829_TICKS_PER_1000_US 125

#ifndef PROGMEM
#define PROGMEM
#endif

#ifndef F
#define F(str_literal) (str_literal)
#endif

#define PTR_TO_UINT(ptr) ((intptr_t)(ptr))

#define PRINT_CHAR(c) printChar((c))
#define PRINT_INT(i) printInt((i))
#define PRINT_UINT(i) printUint((i))
#define PRINT_UINT_HEX(i) printUintHex((i))
#define PRINT_STR(str) printStr((char *)(str))
#define PRINT_CONST_STR(str) printConstStr((const char *)(str))
#define PRINT_LN() printLn()
#define SEPARATOR ','

#define UNSUPPORTED_BUS_ERROR -2

#define I2C_SUCCESS 0
#define I2C_ERR_DATA_TOO_LONG -1
#define I2C_ERR_SLAVE_ADDR_NAK -2
#define I2C_ERR_DATA_NAK -3
#define I2C_ERR_OTHER -4
#define I2C_ERR_TIMEOUT -5

void delayInMicroseconds(uint32_t wait);
uint32_t getSysTick(void);
uint8_t readProgramMemoryByte(uint32_t address);

void enablePinHigh(void *dptr);
void enablePinLow(void *dptr);
void configurePins(void *dptr);

void i2cOpen(void *dptr, uint32_t i2cClockSpeedInHz);
void i2cClose(void *dptr);

void printChar(char c);
void printInt(int32_t i);
void printUint(uint32_t i);
void printUintHex(uint32_t i);
void printStr(char *str);
void printConstStr(const char *str);
void printLn(void);

int8_t txReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
             uint16_t toTx, const uint8_t *txData);
int8_t rxReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
             uint16_t toRx, uint8_t *rxData);

int8_t i2cTxReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
                uint16_t toTx, const uint8_t *txData);
int8_t i2cRxReg(void *dptr, uint8_t slaveAddr, uint8_t regAddr,
                uint16_t toRx, uint8_t *rxData);
int8_t i2cTxRx(void *dptr, uint8_t slaveAddr, uint16_t toTx,
               const uint8_t *txData, uint16_t toRx, uint8_t *rxData);

/* Functions used by the upstream example/application callbacks. */
void inputOpen(uint32_t baudrate);
void inputClose(void);
int8_t inputGetKey(char *c);
void pinOutput(uint8_t pin);
void pinInput(uint8_t pin);
void setInterruptHandler(void (*handler)(void));
void clrInterruptHandler(void);
void disableInterrupts(void);
void enableInterrupts(void);

void handleReceivedFrameHeaderData(void *dptr, uint8_t *data);
void handleReceivedResultData(void *dptr, uint8_t *data, uint16_t size);
void handleReceivedHistogramData(void *dptr, uint8_t *data, uint16_t size);
void handleReceivedResultDataEnd(void *dptr);
void handleReceivedHistogramDataEnd(void *dptr);

void printResultHeader(void *dptr, uint8_t *data, uint8_t len);
void printResults(void *dptr, uint8_t *data, uint16_t len);
void printHistogram(void *dptr, uint8_t *data, uint16_t len);

#ifdef __cplusplus
}
#endif

#endif /* TMF8829_SHIM_H */
