#include "SerialTransfer.h"
SerialTransfer myTransfer;

int ECGpin = A0;
int EMGpin = A5;
int GSRpin = A2;

struct Data {
  unsigned long Timestamp;
  uint16_t ECG;
  uint16_t EMG;
  uint16_t GSR;
  unsigned long label;
} testData;

void setup() {
  Serial.begin(57600);
  myTransfer.begin(Serial);
  testData.Timestamp = 0;
  testData.ECG = 0;
  testData.EMG = 0;
  testData.GSR = 0;
  testData.label = 1;
}

void loop() {
  // use this variable to keep track of how many
  // bytes we’re stuffing in the transmit buffer
  uint16_t sendSize = 0;
  testData.EMG = analogRead(EMGpin);
  testData.ECG = analogRead(ECGpin);
  testData.GSR = analogRead(GSRpin);
  ///////////////////////////////////////// Stuff buffer with struct
  sendSize = myTransfer.txObj(testData, sendSize);
  ///////////////////////////////////////// Send buffer
  myTransfer.sendData(sendSize);
  delay(10);
  testData.Timestamp += 1;

}
