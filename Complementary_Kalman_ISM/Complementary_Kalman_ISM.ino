#include <Wire.h>
#include "SparkFun_ISM330DHCX.h"

SparkFun_ISM330DHCX myISM;
sfe_ism_data_t accelData;
sfe_ism_data_t gyroData;

float dt = 0.02;
float gyroCalX, gyroCalY, gyroCalZ;
float accelCalX, accelCalY, accelCalZ;

float gyroPitch = 0, accelPitch = 0, compPitch = 0, predictedPitch = 0;
float gyroRoll = 0, accelRoll = 0, compRoll = 0, predictedRoll = 0;

float Q = 0.1, R = 5;
float P00 = 0.1, P11 = 0.1, P01 = 0.1;
float Kk0, Kk1;

unsigned long timer, currentTime;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!myISM.begin()) {
    Serial.println("IMU not detected. Check wiring.");
    while (1);
  }

  myISM.setAccelDataRate(ISM_XL_ODR_104Hz);
  myISM.setAccelFullScale(ISM_4g);
  myISM.setGyroDataRate(ISM_GY_ODR_104Hz);
  myISM.setGyroFullScale(ISM_2000dps);

  // Calibration
  long sumGyroX = 0, sumGyroY = 0, sumGyroZ = 0;
  long sumAccelX = 0, sumAccelY = 0, sumAccelZ = 0;

  for (int i = 0; i < 100; i++) {
    myISM.getGyro(&gyroData);
    myISM.getAccel(&accelData);

    sumGyroX += gyroData.xData;
    sumGyroY += gyroData.yData;
    sumGyroZ += gyroData.zData;

    sumAccelX += accelData.xData;
    sumAccelY += accelData.yData;
    sumAccelZ += accelData.zData;

    delay(10);
  }

  gyroCalX = sumGyroX / 100.0;
  gyroCalY = sumGyroY / 100.0;
  gyroCalZ = sumGyroZ / 100.0;

  accelCalX = sumAccelX / 100.0;
  accelCalY = sumAccelY / 100.0;
  accelCalZ = (sumAccelZ / 100.0) - 1.0; // 1g gravity compensation (assuming ±4g range = 1g → ~1.0)
}

void loop() {
  timer = millis();

  myISM.getGyro(&gyroData);
  myISM.getAccel(&accelData);

  // Accelerometer-based angles
  accelPitch = atan2((accelData.yData - accelCalY), (accelData.zData - accelCalZ)) * 180 / PI;
  accelRoll = atan2((accelData.xData - accelCalX), (accelData.zData - accelCalZ)) * 180 / PI;

  // Gyroscope-based integration
  gyroPitch += ((gyroData.xData - gyroCalX) * dt);
  gyroRoll -= ((gyroData.yData - gyroCalY) * dt);

  // Complementary filter
  float alpha = 0.98;
  compPitch = alpha * (compPitch + (gyroData.xData - gyroCalX) * dt) + (1 - alpha) * accelPitch;
  compRoll = alpha * (compRoll - (gyroData.yData - gyroCalY) * dt) + (1 - alpha) * accelRoll;

  // Kalman Filter Time Update
  predictedPitch += ((gyroData.xData - gyroCalX) * dt);
  predictedRoll -= ((gyroData.yData - gyroCalY) * dt);

  P00 += dt * (2 * P01 + dt * P11);
  P01 += dt * P11;
  P00 += dt * Q;
  P11 += dt * Q;

  // Kalman Filter Measurement Update
  Kk0 = P00 / (P00 + R);
  Kk1 = P01 / (P01 + R);

  predictedPitch += (accelPitch - predictedPitch) * Kk0;
  predictedRoll += (accelRoll - predictedRoll) * Kk0;

  P00 *= (1 - Kk0);
  P01 *= (1 - Kk1);
  P11 -= Kk1 * P01;

  currentTime = millis();
  Serial.print("Time:"); Serial.print(currentTime); 
  Serial.print(" Gyro Pitch:"); Serial.print(gyroPitch); 
  Serial.print(" Accel Pitch:"); Serial.print(accelPitch);
  Serial.print(" Comple Pitch:"); Serial.print(compPitch); 
  Serial.print(" Predicted Pitch:"); Serial.print(predictedPitch); 

  Serial.print(" Gyro Roll:"); Serial.print(gyroRoll); 
  Serial.print(" Accel Roll:"); Serial.print(accelRoll); 
  Serial.print(" Comple Roll:"); Serial.print(compRoll); 
  Serial.print(" Predicted Roll:"); Serial.println(predictedRoll); 

  // Delay to maintain ~50Hz loop
  timer = millis() - timer;
  delay((dt * 1000) - timer);
}
