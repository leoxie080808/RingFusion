# RingFusion LightRanger 14 / ESP32-C6 ESP-IDF project

Target folder:

`C:\Users\xiele\Documents\RingFusion\firmware-esp`

Hardware mapping:

- LightRanger 14 SDA -> ESP32-C6 GPIO6
- LightRanger 14 SCL -> ESP32-C6 GPIO7
- LightRanger 14 INT -> ESP32-C6 GPIO4
- LightRanger 14 EN -> ESP32-C6 GPIO5
- 3V3 -> 3V3
- GND -> GND

The board is based on the ams OSRAM TMF8829. The project uses the official portable
TMF8829 C driver and firmware image, with an ESP-IDF hardware shim.

## Install the project

1. Extract this folder anywhere.
2. In ordinary PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\tools\copy_to_ringfusion.ps1
& "C:\Users\xiele\Documents\RingFusion\firmware-esp\tools\install_official_tmf8829_driver.ps1"
```

## Build from ESP-IDF PowerShell

```powershell
cd C:\Users\xiele\Documents\RingFusion\firmware-esp
idf.py set-target esp32c6
idf.py build
```

## Open for editing in VS Code

```powershell
code C:\Users\xiele\Documents\RingFusion\firmware-esp
```

Use the ESP-IDF VS Code extension only for IntelliSense/editor support if desired. The
actual build can remain in the dedicated ESP-IDF PowerShell.

## Flash and monitor

Find the ESP32-C6 serial port in Device Manager, then run, for example:

```powershell
idf.py -p COM7 flash monitor
```

Exit the monitor with `Ctrl+]`.

## Expected startup

The program should:

1. reset and enable the TMF8829;
2. find it at its default 7-bit I2C address;
3. download the official RAM firmware;
4. load the official 8x8 configuration;
5. start ranging;
6. print raw official-driver result frames over the ESP-IDF monitor.

The current callback intentionally prints the complete raw result packet. The next layer
can decode this into a clean 8x8 array of distance and confidence values after hardware
bring-up is confirmed.

## Hardware checks

- The LightRanger 14 COMM SEL jumpers must be in the I2C position.
- Use 3.3 V only.
- Keep I2C wiring short during initial testing.
- The Click board already includes external I2C pull-up resistors, so the ESP-IDF port does
  not enable the ESP32's weak internal I2C pull-ups.
