#include <Wire.h>
#include <ESP8266WiFi.h>
#include "SparkFun_ISM330DHCX.h"

const char* ssid = "POWER BB 2";
const char* password = "shardul0473";

WiFiServer server(80);
SparkFun_ISM330DHCX myISM;
sfe_ism_data_t accelData;

// Motor Pins
#define M1_IN1 14  // GPIO14 (D5)
#define M1_IN2 12  // GPIO12 (D6)
#define M2_IN1 13  // GPIO13 (D7)
#define M2_IN2 15  // GPIO15 (D8)

// Positioning variables
float vx = 0.0, vy = 0.0;
float x_pos = 0.0, y_pos = 0.0;
float prev_ax = 0.0, prev_ay = 0.0;
float kalmanGain = 0.6;
unsigned long lastTime = 0;

// Deadband thresholds (modifiable from UI)
float deadbandX = 0.036;
float deadbandY = 0.047;

void calibrateAccelerometer(float raw[3], float calibrated[3]) {
  float b[3] = {11.723946, -14.593920, -2.679943};
  float A_inv[3][3] = {
    {1.004786,  0.01983,   -0.001099},
    {0.01983,   1.002748,  -0.000478},
    {-0.001099, -0.000478, 1.000420}
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
  return (abs(ax) < deadbandX && abs(ay) < deadbandY);
}

void stopMotors() {
  digitalWrite(M1_IN1, LOW); digitalWrite(M1_IN2, LOW);
  digitalWrite(M2_IN1, LOW); digitalWrite(M2_IN2, LOW);
}
void forward() {
  digitalWrite(M1_IN1, HIGH); digitalWrite(M1_IN2, LOW);
  digitalWrite(M2_IN1, HIGH); digitalWrite(M2_IN2, LOW);
}
void backward() {
  digitalWrite(M1_IN1, LOW); digitalWrite(M1_IN2, HIGH);
  digitalWrite(M2_IN1, LOW); digitalWrite(M2_IN2, HIGH);
}
void left() {
  digitalWrite(M1_IN1, LOW); digitalWrite(M1_IN2, HIGH);
  digitalWrite(M2_IN1, HIGH); digitalWrite(M2_IN2, LOW);
}
void right() {
  digitalWrite(M1_IN1, HIGH); digitalWrite(M1_IN2, LOW);
  digitalWrite(M2_IN1, LOW); digitalWrite(M2_IN2, HIGH);
}

String htmlPage() {
  return R"rawliteral(
<html><head>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<style>
  body { font-family: sans-serif; text-align: center; margin-top: 30px; }
  button { width: 120px; height: 50px; margin: 10px; font-size: 18px; }
  input { width: 60px; font-size: 18px; }
</style>
</head><body>
<h2>IMU-Based Car Controller</h2>
<button onclick="location.href='/forward'">Forward</button><br>
<button onclick="location.href='/left'">Left</button>
<button onclick="location.href='/stop'">Stop</button>
<button onclick="location.href='/right'">Right</button><br>
<button onclick="location.href='/backward'">Backward</button><br><br>

<h3>Position and Sensor Data:</h3>
<div id='imuData'>Loading...</div><br>

<button onclick="fetch('/reset')">Reset Position</button><br><br>

<h3>Set Deadband Thresholds</h3>
X: <input id="dbx" type="number" step="0.001" value="0.036">
Y: <input id="dby" type="number" step="0.001" value="0.047"><br>
<button onclick="setDeadband()">Set Deadband</button>

<script>
setInterval(() => {
  fetch('/imu')
    .then(res => res.text())
    .then(val => document.getElementById('imuData').innerHTML = val);
}, 500);

function setDeadband() {
  const x = document.getElementById('dbx').value;
  const y = document.getElementById('dby').value;
  fetch(`/set_deadband?x=${x}&y=${y}`);
}
</script>
</body></html>
)rawliteral";
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  pinMode(M1_IN1, OUTPUT); pinMode(M1_IN2, OUTPUT);
  pinMode(M2_IN1, OUTPUT); pinMode(M2_IN2, OUTPUT);
  stopMotors();

  WiFi.begin(ssid, password);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nConnected to WiFi!");
  Serial.println(WiFi.localIP());
  server.begin();

  if (!myISM.begin()) {
    Serial.println("IMU not detected");
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

void handleIMU(WiFiClient& client) {
  myISM.getAccel(&accelData);
  float raw_ax = accelData.xData;
  float raw_ay = accelData.yData;
  float raw_az = accelData.zData;

  float raw[3] = {raw_ax, raw_ay, raw_az};
  float calibrated[3];
  calibrateAccelerometer(raw, calibrated);

  float ax = calibrated[0] * 0.001;
  float ay = calibrated[1] * 0.001;
  float az = calibrated[2] * 0.001;
  /*
  if (abs(ax) < deadbandX) ax = 0.0;
  if (abs(ay) < deadbandY) ay = 0.0;
  */
  unsigned long currentTime = micros();
  float dt = (currentTime - lastTime) / 1000000.0;
  lastTime = currentTime;

  vx += 0.5 * (prev_ax + ax) * dt;
  vy += 0.5 * (prev_ay + ay) * dt;


  if (isStationary(ax, ay)) {
    vx = 0;
    vy = 0;
  }

  float raw_x = x_pos + vx * dt;
  float raw_y = y_pos + vy * dt;

  x_pos = kalmanGain * raw_x + (1 - kalmanGain) * x_pos;
  y_pos = kalmanGain * raw_y + (1 - kalmanGain) * y_pos;

  String html = "";
  html += "<b>X:</b> " + String(x_pos, 3) + " cm<br>";
  html += "<b>Y:</b> " + String(y_pos, 3) + " cm<br>";
  html += "<b>vx:</b> " + String(vx, 3) + " | ";
  html += "<b>vy:</b> " + String(vy, 3) + "<br>";
  html += "<b>ax:</b> " + String(ax, 3) + " | ";
  html += "<b>ay:</b> " + String(ay, 3) + " | ";
  html += "<b>az:</b> " + String(az, 3) + "<br>";
  html += "<b>raw_ax:</b> " + String(raw_ax, 3) + " | ";
  html += "<b>raw_ay:</b> " + String(raw_ay, 3) + " | ";
  html += "<b>raw_az:</b> " + String(raw_az, 3);

  client.print("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
  client.print(html);

  prev_ax = ax;
  prev_ay = ay;

}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;

  while (client.connected() && !client.available()) delay(1);
  String request = client.readStringUntil('\r');
  client.flush();
  Serial.println(request);

  if (request.indexOf("/forward") != -1) forward();
  else if (request.indexOf("/backward") != -1) backward();
  else if (request.indexOf("/left") != -1) left();
  else if (request.indexOf("/right") != -1) right();
  else if (request.indexOf("/stop") != -1) stopMotors();
  else if (request.indexOf("/imu") != -1) {
    handleIMU(client);
    return;
  }
  else if (request.indexOf("/reset") != -1) {
    vx = 0; vy = 0;
    x_pos = 0; y_pos = 0;
    client.print("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n");
    client.print("Position reset");
    return;
  }
  else if (request.indexOf("/set_deadband") != -1) {
    int xIndex = request.indexOf("x=");
    int yIndex = request.indexOf("&y=");
    if (xIndex != -1 && yIndex != -1) {
      String xStr = request.substring(xIndex + 2, yIndex);
      String yStr = request.substring(yIndex + 3);
      deadbandX = xStr.toFloat();
      deadbandY = yStr.toFloat();
      Serial.printf("Deadbands Updated → X: %.3f, Y: %.3f\n", deadbandX, deadbandY);
    }
    client.print("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n");
    client.print("Deadband updated");
    return;
  }

  client.print("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
  client.print(htmlPage());
}
