#include <Wire.h>
#include <QMC5883LCompass.h>

QMC5883LCompass compass;

// === Paste Your MATLAB Calibration Values Here ===
#define CALIBRATION__MAGN_USE_EXTENDED true

const float magn_ellipsoid_center[3] = { -154.600706, 184.157974, 323.076601 };
const float magn_ellipsoid_transform[3][3] = {
  { 0.980417, -0.026740, -0.009220 },
  { -0.026740, 0.961075, -0.026725 },
  { -0.009220, -0.026725, 0.912801 }
};

// === Compensation Buffer ===
float compensated[3];

void setup() {
  Serial.begin(115200);
  Wire.begin(4, 5); // SDA = GPIO4 (D2), SCL = GPIO5 (D1) on ESP8266
  compass.init();
  Serial.println("Heading (°)");
}

void loop() {
  compass.read();

  // Raw values from magnetometer
  float raw[3] = {
    (float)compass.getX(),
    (float)compass.getY(),
    (float)compass.getZ()
  };

  // Step 1: Subtract center offset
  float centered[3];
  for (int i = 0; i < 3; i++) {
    centered[i] = raw[i] - magn_ellipsoid_center[i];
  }

  // Step 2: Apply 3x3 transformation matrix
  for (int i = 0; i < 3; i++) {
    compensated[i] =
      magn_ellipsoid_transform[i][0] * centered[0] +
      magn_ellipsoid_transform[i][1] * centered[1] +
      magn_ellipsoid_transform[i][2] * centered[2];
  }

  // Step 3: Calculate heading in degrees
  float heading = atan2(compensated[1], compensated[0]) * 180.0 / PI;
  if (heading < 0) heading += 360.0;

  Serial.println(heading);

  delay(100); // 10 Hz update
}
