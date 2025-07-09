#include <Wire.h>
#include "SparkFun_ISM330DHCX.h"
#include <SoftwareSerial.h>

SparkFun_ISM330DHCX myISM;
sfe_ism_data_t accelData;

SoftwareSerial softSerial(3, 1); // RX = 3, TX = 1

float vx = 0.0, vy = 0.0;
float x_pos = 0.0, y_pos = 0.0;
float kalmanGain = 0.6;
unsigned long lastTime = 0;

// === Accelerometer Calibration (from Magneto) ===
void calibrateAccelerometer(float raw[3], float calibrated[3]) {
  float b[3] = {11.723946, -14.593920, -2.679943};
  float A_inv[3][3] = {
    { 1.004786,  0.01983,   -0.001099 },
    { 0.01983,   1.002748,  -0.000478 },
    { -0.001099, -0.000478, 1.000420 }
  };

  float temp[3];
  for (int i = 0; i < 3; i++) temp[i] = raw[i] - b[i];

  for (int i = 0; i < 3; i++) {
    calibrated[i] = 0;
    for (int j = 0; j < 3; j++) {
      calibrated[i] += A_inv[i][j] * temp[j];
    }
  }
}

bool isStationary(float ax, float ay) {
  return (abs(ax) < 0.03 && abs(ay) < 0.2);
}

void setup() {
  Wire.begin();
  Serial.begin(115200);
  softSerial.begin(9600);

  if (!myISM.begin()) {
    Serial.println("IMU not detected.");
    while (1);
  }

  myISM.deviceReset();
  while (!myISM.getDeviceReset()) delay(1);

  myISM.setDeviceConfig();
  myISM.setBlockDataUpdate();
  myISM.setAccelDataRate(ISM_XL_ODR_104Hz);
  myISM.setAccelFullScale(ISM_2g);
  myISM.setGyroDataRate(ISM_GY_ODR_104Hz);
  myISM.setGyroFullScale(ISM_250dps);
  myISM.setAccelFilterLP2();
  myISM.setAccelSlopeFilter(ISM_LP_ODR_DIV_100);
  myISM.setGyroFilterLP1();
  myISM.setGyroLP1Bandwidth(ISM_MEDIUM);

  lastTime = millis();
}

void loop() {
  if (myISM.checkStatus()) {
    myISM.getAccel(&accelData);

    // Raw accelerometer readings
    float raw_ax = accelData.xData ;
    float raw_ay = accelData.yData ;
    float raw_az = accelData.zData ;

    float raw[3] = {raw_ax, raw_ay, raw_az};
    float calibrated[3];

    calibrateAccelerometer(raw, calibrated); // Apply calibration

    float ax = calibrated[0] * 0.001;
    float ay = calibrated[1] * 0.001;
    float az = calibrated[2] * 0.001;

    // Apply deadband filtering
    float deadbandX = 0.025;
    float deadbandY = 0.045;

    if (abs(ax) < deadbandX) ax = 0.0;
    if (abs(ay) < deadbandY) ay = 0.0;

    // Time delta for integration
    unsigned long currentTime = millis();
    float dt = (currentTime - lastTime) / 1000.0;
    lastTime = currentTime;

    // Integrate acceleration to velocity
    vx += ax * dt;
    vy += ay * dt;

    // ZUPT: reset velocity if stationary (based on raw data)
    if (isStationary(ax, ay)) {
      vx = 0;
      vy = 0;
    }

    // Integrate velocity to position
    float raw_x = x_pos + vx * dt;
    float raw_y = y_pos + vy * dt;

    // Kalman-like smoothing
    x_pos = kalmanGain * raw_x + (1 - kalmanGain) * x_pos;
    y_pos = kalmanGain * raw_y + (1 - kalmanGain) * y_pos;

    // ===== Output via Software Serial ===== //
    softSerial.print("X:");   softSerial.print(x_pos, 3);
    softSerial.print(",Y:");  softSerial.print(y_pos, 3);
    softSerial.print(",vx:"); softSerial.print(vx, 3);
    softSerial.print(",vy:"); softSerial.print(vy, 3);
    softSerial.print(",ax:"); softSerial.print(ax, 3);
    softSerial.print(",ay:"); softSerial.print(ay, 3);
    softSerial.print(",az:"); softSerial.print(az, 3);
    softSerial.print(",raw_ax:"); softSerial.print(raw_ax, 3);
    softSerial.print(",raw_ay:"); softSerial.print(raw_ay, 3);
    softSerial.print(",raw_az:"); softSerial.println(raw_az, 3);

    // ===== Debug Output to Serial Monitor ===== //
    Serial.print("X: "); Serial.print(x_pos, 3);
    Serial.print(" | Y: "); Serial.print(y_pos, 3);
    Serial.print(" | vx: "); Serial.print(vx, 3);
    Serial.print(" | vy: "); Serial.print(vy, 3);
    Serial.print(" | ax: "); Serial.print(ax, 3);
    Serial.print(" | ay: "); Serial.print(ay, 3);
    Serial.print(" | az: "); Serial.print(az, 3);
    Serial.print(" | raw_ax: "); Serial.print(raw_ax, 3);
    Serial.print(" | raw_ay: "); Serial.print(raw_ay, 3);
    Serial.print(" | raw_az: "); Serial.println(raw_az, 3);
  }

  delay(100); // 10 Hz
}
